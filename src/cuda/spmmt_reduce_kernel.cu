#include <cuda.h>
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <iostream>
#include "cutlass/cutlass.h"
#include "cutlass/gemm/device/gemm_sparse.h"
#include <stdio.h>
#include <vector>
#include "cuda_bf16.h"
#include "helper.h"
#include "mma/default_mma.h"
#include <type_traits>

#include "spmmt/default_sparse_mma_trans_reduce.h"
#include "epilogue/pipelined_transpose_epilogue.h"
#include "epilogue/linear_combination.h"

// Define the Tile Size in different levels

using ThreadblockShape_16 = cutlass::gemm::GemmShape<128, 256, 64>;
using WarpShape_16 = cutlass::gemm::GemmShape<32, 128, 64>;
using InstructionShape_16 = cutlass::gemm::GemmShape<16, 8, 32>;

// Define MMA & Epilogue
using ThreadblockSwizzle = cutlass::gemm::threadblock::GemmBatchedIdentityThreadblockSwizzle;

// DefaultConfigurations for float & bf16
using DefaultConfig = cutlass::gemm::device::DefaultGemmConfiguration<
    cutlass::arch::OpClassTensorOp, cutlass::arch::Sm80, float, float, float, float>;

// Pipeline stages in GEMM
constexpr int NumStages = 3;


// A structure to switch between different configurations
template<typename Element_, typename LayoutB_, bool Trans_>
struct SpMMTConfigure{
    static const bool Trans = Trans_;
    using LayoutB = LayoutB_;
    using Element = Element_;

    using EpilogueOp = cutlass::epilogue::thread::LinearCombination_<
        Element, 128 / cutlass::sizeof_bits<Element>::value, float, Element,
        cutlass::epilogue::thread::ScaleType::OnlyAlphaScaling>;  
    
    using Mma = typename cutlass::gemm::threadblock::DefaultSparseMmaTransReduce<
        Element, cutlass::layout::RowMajor, 128 / cutlass::sizeof_bits<Element>::value,
        Element, LayoutB, 128 / cutlass::sizeof_bits<Element>::value,
        float, cutlass::layout::RowMajor, cutlass::arch::OpClassTensorOp, cutlass::arch::Sm80,
        ThreadblockShape_16, WarpShape_16, InstructionShape_16, NumStages, cutlass::arch::OpMultiplyAdd>::ThreadblockMma;
    
    using Epilogue = typename std::conditional<Trans,
        typename cutlass::epilogue::threadblock::DefaultTransposeEpilogue<
            ThreadblockShape_16, WarpShape_16, EpilogueOp, Mma>::Epilogue,
        typename cutlass::epilogue::threadblock::DefaultEpilogueTensorOp<
            ThreadblockShape_16, typename Mma::Operator, ThreadblockShape_16::kK / WarpShape_16::kK, EpilogueOp,
            EpilogueOp::kCount>::Epilogue>::type;
    
    union SharedStorage {
        typename Mma::SharedStorage main_loop;
        typename Epilogue::SharedStorage epilogue;
    };
};


template<typename Element, typename _Mma, typename _SharedStorage, typename _Epilogue>
__global__ void cutlassSpmmTKernel_reduce_16(
    cutlass::gemm::GemmCoord problem_size,
    cutlass::gemm::GemmCoord grid_tiled_shape,
    typename _Mma::IteratorA::Params params_A,
    Element* __restrict__ ptr_A,
    typename _Mma::IteratorB::Params params_B,
    Element* __restrict__ ptr_B,
    typename _Epilogue::OutputTileIterator::Params params_D,
    Element* __restrict__ ptr_D,
    typename _Mma::IteratorE::Params params_E,
    typename _Mma::ElementE* __restrict__ ptr_E,
    typename _Epilogue::OutputOp::Params output_op_,
    int gemm_k_size)
{
    extern __shared__ int SharedStorageBase[];

    _SharedStorage& shared_storage = *reinterpret_cast<_SharedStorage *>(SharedStorageBase);

    ThreadblockSwizzle threadblock_swizzle;

    cutlass::gemm::GemmCoord threadblock_tile_offset=threadblock_swizzle.get_tile_offset(grid_tiled_shape);

    // Early exit if CTA is out of range
    if (grid_tiled_shape.m() <= threadblock_tile_offset.m() ||
        grid_tiled_shape.n() <= threadblock_tile_offset.n())
    {
        return;
    }

    int batch_idx = threadblock_swizzle.get_batch_idx();

    // Compute initial location in logical coordinates
    cutlass::MatrixCoord tb_offset_A{
        0,
        threadblock_tile_offset.m() * _Mma::Shape::kM / _Mma::kSparse
    };

    cutlass::MatrixCoord tb_offset_B{
        0,
        threadblock_tile_offset.n() * _Mma::Shape::kN
    };

    cutlass::MatrixCoord tb_offset_E{
        0,
        threadblock_tile_offset.m() * _Mma::Shape::kM / _Mma::kSparse / _Mma::kElementsPerElementE
    };

    // Problem size
    int problem_size_k = problem_size.k(); //min(problem_size.k(), (threadblock_tile_offset.k() + 1) * gemm_k_size);

    int gemm_k_iterations = (problem_size_k - tb_offset_B.row() + _Mma::Shape::kK - 1) / _Mma::Shape::kK;

    // Compute position within threadblock
    int thread_idx = threadIdx.x;

    // Construct iterators to A, B, and E operands
    typename _Mma::IteratorA iterator_A(
        params_A,
        //ref_A.data(),
        ptr_A,
        {problem_size_k, problem_size.m() / _Mma::kSparse},
        thread_idx,
        tb_offset_A
    );

    int64_t a_stride = problem_size.m() * problem_size.k() / _Mma::kSparse;
    iterator_A.add_pointer_offset(batch_idx * a_stride);

    typename _Mma::IteratorB iterator_B(
        params_B,
        //ref_B.data(),
        ptr_B,
        {problem_size_k, problem_size.n()},
        thread_idx,
        tb_offset_B
    );

    int64_t b_stride = problem_size.n() * problem_size.k();
    iterator_B.add_pointer_offset(batch_idx * b_stride);

    typename _Mma::IteratorE iterator_E(
        params_E,
        // ref_E.data(),
        ptr_E,
        {problem_size_k,
        problem_size.m() / _Mma::kSparse / _Mma::kElementsPerElementE},
        thread_idx,
        tb_offset_E
    );

    int64_t e_stride = problem_size.m() * problem_size.k() / _Mma::kSparse / _Mma::kElementsPerElementE;
    iterator_E.add_pointer_offset(batch_idx * e_stride);

    // Broadcast the warp_id computed by lane 0 to ensure dependent code
    // is compuled as warp-uniform
    int warp_idx = __shfl_sync(0xffffffff, threadIdx.x / 32, 0);
    int lane_idx = threadIdx.x % 32;

    //
    //  Main loop
    //

    // Construct thread-scoped matrix multiply
    _Mma mma(shared_storage.main_loop, thread_idx, warp_idx, lane_idx);

    typename _Mma::FragmentC accumulators;

    accumulators.clear();

    if (gemm_k_iterations > 0){
        mma(gemm_k_iterations, accumulators, iterator_A, iterator_B, iterator_E, accumulators);
    }

    //
    //  Epilogue
    //

    typename _Epilogue::OutputOp output_op(output_op_);

    threadblock_tile_offset = threadblock_swizzle.get_tile_offset(grid_tiled_shape);

    // (blockIdx.x * TileM, blockIdx.y * TileN)
    cutlass::MatrixCoord threadblock_offset(
        threadblock_tile_offset.m() * _Mma::Shape::kM,
        threadblock_tile_offset.n() * _Mma::Shape::kN
    );

    int block_idx = threadblock_tile_offset.m() + threadblock_tile_offset.n() * grid_tiled_shape.m();
    
    typename _Epilogue::OutputTileIterator iterator_D(
        params_D,
        ptr_D,
        problem_size.mn(),
        thread_idx,
        threadblock_offset
    );

    int64_t d_stride = problem_size.m() * problem_size.n();
    iterator_D.add_pointer_offset(batch_idx * d_stride);

    
    _Epilogue epilogue(
        shared_storage.epilogue,
        thread_idx,
        warp_idx,
        lane_idx
    );

    epilogue(output_op, iterator_D, accumulators, iterator_D);
}


template<typename Config>
std::vector<torch::Tensor> spmmt_reduce_cuda(
    torch::Tensor tensor_a,
    torch::Tensor tensor_b,
    torch::Tensor tensor_e,
    const float alpha_)
{
    const int m = tensor_a.size(-1) * 2;
    int n, k;
    if (std::is_same<typename Config::LayoutB, cutlass::layout::ColumnMajor>::value){
        n = tensor_b.size(-2);
        k = tensor_b.size(-1);
    } else {
        n = tensor_b.size(-1);
        k = tensor_b.size(-2);
    }

    int batch_size = tensor_b.numel() / n / k;

    auto options_val = torch::TensorOptions().dtype(tensor_a.dtype()).device(tensor_b.device());
    torch::Tensor output_matrix;
    if (Config::Trans){
        output_matrix = torch::empty({batch_size, n, m}, options_val);
    } else {
        output_matrix = torch::empty({batch_size, m, n}, options_val);
    }

    auto output_reduce = torch::empty({batch_size, k}, options_val);

    // Create a tuple of problem size for matrix multiplication
    cutlass::gemm::GemmCoord problem_size(m, n, k);

    auto layout_a = cutlass::layout::RowMajor::packed(cutlass::make_Coord(problem_size.k(), problem_size.m()/2));
    auto layout_b = Config::LayoutB::packed(problem_size.kn());
    auto layout_e = Config::Mma::LayoutE::packed(cutlass::make_Coord(problem_size.k(), problem_size.m()/Config::Mma::kSparse / Config::Mma::kElementsPerElementE));
    auto layout_d = cutlass::layout::RowMajor::packed(problem_size.mn());

    typename Config::Element alpha = typename Config::Element(alpha_);
    typename Config::Element beta = typename Config::Element(0.0);
    
    ThreadblockSwizzle threadblock_swizzle;

    cutlass::gemm::GemmCoord grid_tiled_shape = threadblock_swizzle.get_tiled_shape(
        problem_size,
        {ThreadblockShape_16::kM, ThreadblockShape_16::kN, ThreadblockShape_16::kK},
        batch_size
    );

    dim3 grid = threadblock_swizzle.get_grid_shape(grid_tiled_shape);
    dim3 block(Config::Mma::WarpCount::kCount * 32, 1, 1);

    int smem_size = int(sizeof(typename Config::SharedStorage));

    cudaFuncSetAttribute(cutlassSpmmTKernel_reduce_16<typename Config::Element, typename Config::Mma, typename Config::SharedStorage, typename Config::Epilogue>, cudaFuncAttributeMaxDynamicSharedMemorySize, smem_size);
    cudaFuncSetAttribute(cutlassSpmmTKernel_reduce_16<typename Config::Element, typename Config::Mma, typename Config::SharedStorage, typename Config::Epilogue>, cudaFuncAttributePreferredSharedMemoryCarveout, 100);

    int gemm_k_size = ((problem_size.k() + Config::Mma::Shape::kK - 1) / Config::Mma::Shape::kK) * Config::Mma::Shape::kK;

    cutlassSpmmTKernel_reduce_16<typename Config::Element, typename Config::Mma, typename Config::SharedStorage, typename Config::Epilogue><<<grid, block, smem_size>>>(
        problem_size, grid_tiled_shape, 
        layout_a, (typename Config::Element*)tensor_a.data_ptr(),
        layout_b, (typename Config::Element*)tensor_b.data_ptr(),
        layout_d, (typename Config::Element*)output_matrix.data_ptr(),
        layout_e, (typename Config::Mma::ElementE*)tensor_e.data_ptr(),
        {alpha, beta}, gemm_k_size);

    return {output_matrix, output_reduce};
}


std::vector<torch::Tensor> spmmt_reduce_bf16_ntn_cuda(
    torch::Tensor tensor_a,
    torch::Tensor tensor_b,
    torch::Tensor tensor_e,
    const float alpha)
{
    using Config = SpMMTConfigure<cutlass::bfloat16_t, cutlass::layout::ColumnMajor, false>;
    return spmmt_reduce_cuda<Config>(tensor_a, tensor_b, tensor_e, alpha);
}

std::vector<torch::Tensor> spmmt_reduce_bf16_ntt_cuda(
    torch::Tensor tensor_a,
    torch::Tensor tensor_b,
    torch::Tensor tensor_e,
    const float alpha)
{
    using Config = SpMMTConfigure<cutlass::bfloat16_t, cutlass::layout::ColumnMajor, true>;
    return spmmt_reduce_cuda<Config>(tensor_a, tensor_b, tensor_e, alpha);
}

std::vector<torch::Tensor> spmmt_reduce_f16_ntn_cuda(
    torch::Tensor tensor_a,
    torch::Tensor tensor_b,
    torch::Tensor tensor_e,
    const float alpha)
{
    using Config = SpMMTConfigure<cutlass::half_t, cutlass::layout::ColumnMajor, false>;
    return spmmt_reduce_cuda<Config>(tensor_a, tensor_b, tensor_e, alpha);
}

std::vector<torch::Tensor> spmmt_reduce_f16_ntt_cuda(
    torch::Tensor tensor_a,
    torch::Tensor tensor_b,
    torch::Tensor tensor_e,
    const float alpha)
{
    using Config = SpMMTConfigure<cutlass::half_t, cutlass::layout::ColumnMajor, true>;
    return spmmt_reduce_cuda<Config>(tensor_a, tensor_b, tensor_e, alpha);
}


std::vector<torch::Tensor> spmmt_reduce_bf16_nnn_cuda(
    torch::Tensor tensor_a,
    torch::Tensor tensor_b,
    torch::Tensor tensor_e,
    const float alpha)
{
    using Config = SpMMTConfigure<cutlass::bfloat16_t, cutlass::layout::RowMajor, false>;
    return spmmt_reduce_cuda<Config>(tensor_a, tensor_b, tensor_e, alpha);
}

std::vector<torch::Tensor> spmmt_reduce_bf16_nnt_cuda(
    torch::Tensor tensor_a,
    torch::Tensor tensor_b,
    torch::Tensor tensor_e,
    const float alpha)
{
    using Config = SpMMTConfigure<cutlass::bfloat16_t, cutlass::layout::RowMajor, true>;
    return spmmt_reduce_cuda<Config>(tensor_a, tensor_b, tensor_e, alpha);
}

std::vector<torch::Tensor> spmmt_reduce_f16_nnn_cuda(
    torch::Tensor tensor_a,
    torch::Tensor tensor_b,
    torch::Tensor tensor_e,
    const float alpha)
{
    using Config = SpMMTConfigure<cutlass::half_t, cutlass::layout::RowMajor, false>;
    return spmmt_reduce_cuda<Config>(tensor_a, tensor_b, tensor_e, alpha);
}

std::vector<torch::Tensor> spmmt_reduce_f16_nnt_cuda(
    torch::Tensor tensor_a,
    torch::Tensor tensor_b,
    torch::Tensor tensor_e,
    const float alpha)
{
    using Config = SpMMTConfigure<cutlass::half_t, cutlass::layout::RowMajor, true>;
    return spmmt_reduce_cuda<Config>(tensor_a, tensor_b, tensor_e, alpha);
}