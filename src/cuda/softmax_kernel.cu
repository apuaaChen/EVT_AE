#include <cuda.h>
#include <torch/extension.h>
#include <cuda_fp16.h>
#define TORCH_ASSERT_ONLY_METHOD_OPERATORS
#include <ATen/core/Tensor.h>
#include <ATen/cuda/CUDAContext.h>
#include <ATen/Dispatch.h>
#include <ATen/TensorUtils.h>
#include <ATen/TensorOperators.h>
#include <ATen/WrapDimUtils.h>
#include <c10/macros/Macros.h>

#include <ATen/AccumulateType.h>
#include <ATen/cuda/NumericLimits.cuh>
#include <type_traits>

#include <ATen/native/cuda/Loops.cuh>
#include <ATen/native/cuda/MemoryAccess.cuh>
#include <ATen/native/cuda/PersistentSoftmax.cuh>

#ifndef AT_PER_OPERATOR_HEADERS
#include <ATen/Functions.h>
#include <ATen/NativeFunctions.h>
#else
#include <ATen/ops/_masked_softmax_native.h>
#include <ATen/ops/_log_softmax_native.h>
#include <ATen/ops/_log_softmax_backward_data_native.h>
#include <ATen/ops/_softmax_native.h>
#include <ATen/ops/_softmax_backward_data_native.h>
#include <ATen/ops/softmax.h>
#include <ATen/ops/_softmax_backward_data.h>
#endif

namespace at {
namespace extend {

constexpr int ALIGN_BYTES = 16;
const int max_threads = 1024;

template<typename T, typename AccumT, typename OutT>
struct SoftMaxForwardEpilogue {
  __device__ __forceinline__ SoftMaxForwardEpilogue(AccumT max_input, AccumT sum, float bias)
    : max_input(max_input)
    , sum(sum)
    , bias(bias) {}

  __device__ __forceinline__ OutT operator()(T input) const {
    return static_cast<OutT>(std::exp(input - max_input) / sum + bias);
  }

  const AccumT max_input;
  const AccumT sum;
  const float bias;
};


template<>
struct SoftMaxForwardEpilogue<__half, float, __half> {
  __device__ __forceinline__ SoftMaxForwardEpilogue(float max_input, float sum, float bias)
    : max_input(max_input)
    , sum(sum) 
    , bias(bias) {}

  __device__ __forceinline__ __half operator()(__half input) const {
    return __float2half(std::exp(__half2float(input) - max_input) / sum + bias);
  }

  const float max_input;
  const float sum;
  const float bias;
};

inline dim3 SoftMax_getBlockSize(int ILP, uint64_t dim_size) {
  uint64_t block_size = 1;
  uint64_t max_block_size = std::min(dim_size / ILP, static_cast<uint64_t>(max_threads));

  // In the vectorized case we want to trade off allowing more of the buffers to be accessed
  // in a vectorized way against wanting a larger block size to get better utilisation.
  // In general with ILP you can have (ILP-1)/ILP of the buffer accessed vectorised, at the risk
  // of having a very small block size. We choose to keep >= 1/2 of the buffer vectorised while
  // allowing a larger block size.
  if (ILP > 1) {
    max_block_size /= 2;
  }

  while (block_size < (max_block_size)) block_size *= 2;
  // Launch at least a single warp - the kernel assumes that.
  block_size = std::max(block_size, static_cast<uint64_t>(at::cuda::warp_size()));
  return dim3(block_size);
}

template<typename T>
struct Add {
  __device__ __forceinline__ T operator()(T a, T b) const {
    return a + b;
  }
};

template<typename T>
struct Max {
  __device__ __forceinline__ T operator()(T a, T b) const {
    return a < b ? b : a;
  }
};


////////////////////////////////////////////////////////////////////////////////
// Regular kernel (fast when dim_size is large; requires inner_size == 1)
////////////////////////////////////////////////////////////////////////////////


template <typename T, typename AccumT>
struct MaxFloat
{
  __device__ __forceinline__ AccumT operator()(AccumT max, T v) const {
    return ::max(max, (AccumT)v);
  }
};

template <>
struct MaxFloat<__half, float>
{
  __device__ __forceinline__ float operator()(float max, __half v) const {
    return ::max(max, __half2float(v));
  }
};

template<typename T, typename AccumT>
struct AddFloat
{
  __device__ __forceinline__ AccumT operator()(AccumT sum, T v) const {
    return sum + v;
  }
};

template<typename T, typename AccumT>
struct SumExpFloat
{
  __device__ __forceinline__ SumExpFloat(AccumT v)
    : max_k(v) {}

  __device__ __forceinline__ AccumT operator()(AccumT sum, T v) const {
    return sum + std::exp(v - max_k);
  }

  const AccumT max_k;
};

template<>
struct SumExpFloat<__half, float>
{
  __device__ __forceinline__ SumExpFloat(float v)
    : max_k(v) {}

  __device__ __forceinline__ float operator()(float sum, __half v) const {
    return sum + std::exp(__half2float(v) - max_k);
  }

  const float max_k;
};

template <template<typename> class Reduction, typename AccumT>
__device__ __forceinline__ AccumT
blockReduce(AccumT* smem, AccumT val,
            const Reduction<AccumT>& r,
            AccumT defaultVal)
{
  // To avoid RaW races from chaining blockReduce calls together, we need a sync here
  __syncthreads();

  smem[threadIdx.x] = val;

  __syncthreads();

  AccumT warpVal = defaultVal;

  // First warp will perform per-warp reductions for the remaining warps
  uint32_t mask = (((uint64_t)1) << (blockDim.x / C10_WARP_SIZE)) - 1;
  if (threadIdx.x < C10_WARP_SIZE) {
    int lane = threadIdx.x % C10_WARP_SIZE;
    if (lane < blockDim.x / C10_WARP_SIZE) {
#pragma unroll
      for (int i = 0; i < C10_WARP_SIZE; ++i) {
        warpVal = r(warpVal, smem[lane * C10_WARP_SIZE + i]);
      }
#if !defined(USE_ROCM)
      __syncwarp(mask);
#endif
      smem[lane] = warpVal;
    }
  }

  __syncthreads();

  // First thread will perform a reduction of the above per-warp reductions
  AccumT blockVal = defaultVal;

  if (threadIdx.x == 0) {
    for (int i = 0; i < blockDim.x / C10_WARP_SIZE; ++i) {
      blockVal = r(blockVal, smem[i]);
    }
    smem[0] = blockVal;
  }

  // Sync and broadcast
  __syncthreads();
  return smem[0];
}

template <template<typename, typename> class Reduction, int ILP, typename T, typename AccumT>
__device__ __forceinline__ AccumT
ilpReduce(int shift,
          T* data,
          int size,
          const Reduction<T, AccumT>& r,
          AccumT defaultVal)
{
  using LoadT = at::native::memory::aligned_vector<T, ILP>;
  AccumT threadVal = defaultVal;
  int offset = threadIdx.x;

  // shift and do 1
  if(shift > 0){
    data -= shift;
    size += shift;
    if(threadIdx.x >= shift){
      threadVal = r(threadVal, data[offset]);
    }
    size -= blockDim.x;
    data += blockDim.x;
  }
  int last = size % (ILP * blockDim.x);

  T v[ILP];
  LoadT* value = reinterpret_cast<LoadT*>(&v);

  for (; offset * ILP < (size - last); offset += blockDim.x) {
    *value = reinterpret_cast<LoadT*>(data)[offset];

    #pragma unroll
    for (int j = 0; j < ILP; ++j) {
      threadVal = r(threadVal, v[j]);
    }
  }

  offset = size - last + threadIdx.x;
  // Epilogue
  for (; offset < size; offset += blockDim.x)
    threadVal = r(threadVal, data[offset]);

  return threadVal;
}

/**
 * This will apply the Epilogue with vectorized reads & writes when input & output have the same shift
 */
template <int ILP, typename scalar_t, typename accum_t, typename outscalar_t, template<typename, typename, typename> class Epilogue>
__device__ __forceinline__ void
WriteFpropResultsVectorized(
             int size,
             const int shift,
             scalar_t *input,
             outscalar_t *output,
             Epilogue<scalar_t, accum_t, outscalar_t> epilogue) {
  using LoadT = at::native::memory::aligned_vector<scalar_t, ILP>;
  using StoreT = at::native::memory::aligned_vector<outscalar_t, ILP>;

  int offset = threadIdx.x;

  // if unaligned, do one value / thread and move on, guaranteeing aligned reads/writes later
  if (shift > 0) {
    input -= shift;
    output -= shift;
    size += shift;

    if (threadIdx.x >= shift) {
      output[offset] = epilogue(input[offset]);
    }
    size -= blockDim.x;
    input += blockDim.x;
    output += blockDim.x;
  }

  const int last = size % (ILP * blockDim.x);

  scalar_t in_v[ILP];
  LoadT* in_value = reinterpret_cast<LoadT*>(&in_v);

  outscalar_t out_v[ILP];
  StoreT* out_value = reinterpret_cast<StoreT*>(&out_v);

  for (; offset * ILP < (size - last); offset += blockDim.x) {
    *in_value = reinterpret_cast<LoadT*>(input)[offset];

    #pragma unroll
    for (int j = 0; j < ILP; ++j) {
      out_v[j] = epilogue(in_v[j]);
    }

    reinterpret_cast<StoreT*>(output)[offset] = *out_value;
  }

  offset = size - last + threadIdx.x;
  // handle the tail
  for (; offset < size; offset += blockDim.x) {
    output[offset] = epilogue(input[offset]);
  }
}


/**
 * This will apply the Epilogue with non-vectrorized reads & writes for the general case
 */
template <int ILP, typename scalar_t, typename accum_t, typename outscalar_t, template<typename, typename, typename> class Epilogue>
__device__ __forceinline__ void
WriteFpropResults(
             int classes,
             scalar_t *input,
             outscalar_t *output,
             Epilogue<scalar_t, accum_t, outscalar_t> epilogue) {
  int offset = threadIdx.x;

  int last = classes % (ILP * blockDim.x);

  // Main bulk of loop with ILP
  for (; offset < classes - last; offset += blockDim.x * ILP) {
    scalar_t tmp[ILP];

    #pragma unroll
    for (int j = 0; j < ILP; ++j) {
      tmp[j] = input[offset + j * blockDim.x];
    }
    #pragma unroll
    for (int j = 0; j < ILP; ++j) {
      output[offset + j * blockDim.x] = epilogue(tmp[j]);
    }
  }

  // Remainder - no ILP
  for (; offset < classes; offset += blockDim.x) {
    output[offset] = epilogue(input[offset]);
  }
}


template <int ILP, typename scalar_t, typename accscalar_t, typename outscalar_t, template <typename, typename, typename> class Epilogue>
__global__ void
cunn_SoftMaxForward(outscalar_t *output, scalar_t *input, int classes, float bias)
{
  extern __shared__ unsigned char smem[];
  auto sdata = reinterpret_cast<accscalar_t*>(smem);

  using LoadT = at::native::memory::aligned_vector<scalar_t, ILP>;
  using StoreT = at::native::memory::aligned_vector<outscalar_t, ILP>;

  // forward pointers to batch[blockIdx.x]
  // each block handles a sample in the mini-batch
  input += blockIdx.x * classes;
  output += blockIdx.x * classes;

  const int shift = ((uint64_t)input) % ALIGN_BYTES / sizeof(scalar_t);
  const int output_shift = ((uint64_t)output) % ALIGN_BYTES / sizeof(outscalar_t);

  // find the max
  accscalar_t threadMax = ilpReduce<MaxFloat, ILP, scalar_t, accscalar_t>(
      shift, input, classes, MaxFloat<scalar_t, accscalar_t>(), -at::numeric_limits<accscalar_t>::max());
  accscalar_t max_k = blockReduce<Max, accscalar_t>(
      sdata, threadMax, Max<accscalar_t>(), -at::numeric_limits<accscalar_t>::max());

  // reduce all values
  accscalar_t threadExp = ilpReduce<SumExpFloat, ILP, scalar_t, accscalar_t>(
      shift, input, classes, SumExpFloat<scalar_t, accscalar_t>(max_k), static_cast<accscalar_t>(0));
  accscalar_t sumAll = blockReduce<Add, accscalar_t>(
      sdata, threadExp, Add<accscalar_t>(), static_cast<accscalar_t>(0));

  Epilogue<scalar_t, accscalar_t, outscalar_t> epilogue(max_k, sumAll, bias);

  if (shift == output_shift) {
    WriteFpropResultsVectorized<ILP, scalar_t, accscalar_t, outscalar_t, Epilogue>(classes, shift, input, output, epilogue);
  } else {
    WriteFpropResults<ILP, scalar_t, accscalar_t, outscalar_t, Epilogue>(classes, input, output, epilogue);
  }
}

}  // namespace extend
}  // namespace at



torch::Tensor softmax_cuda(torch::Tensor input_, const int64_t dim_, float bias){
  using scalar_t = __half;
  auto input = input_.contiguous();

  auto output = torch::empty_like(input);

  if (input.dim() == 0) input = input.view(1);
  int64_t dim = at::maybe_wrap_dim(dim_, input.dim());
  TORCH_CHECK(dim >=0 && dim < input.dim(), "dim must be non-negative and less than input dimensions");
  int64_t outer_size = 1;
  int64_t dim_size = input.size(dim);

  if (input.numel() > 0) {
    int64_t inner_size = 1;
    cudaStream_t stream = at::cuda::getCurrentCUDAStream();
    for (int64_t i = 0; i < dim; ++i)
      outer_size *= input.size(i);
    for (int64_t i = dim + 1; i < input.dim(); ++i)
      inner_size *= input.size(i);

    dim3 grid(outer_size);
    using accscalar_t = float;
    constexpr int ILP = sizeof(float4) / sizeof(scalar_t);
    dim3 block = at::extend::SoftMax_getBlockSize(ILP, dim_size);
    at::extend::cunn_SoftMaxForward<ILP, scalar_t, accscalar_t, scalar_t, at::extend::SoftMaxForwardEpilogue>
        <<<grid, block, block.x * sizeof(accscalar_t), stream>>>(
        (scalar_t*) output.data_ptr(), (scalar_t*)input.data_ptr(), dim_size, bias);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
  }
  return output;
}