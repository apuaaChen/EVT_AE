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
#include <type_traits>

#include "epilogue/default_epilogue_tensor_op.h"
#include "epilogue/pipelined_transpose_epilogue.h"


// Define the Tile Size in different levels

using ThreadblockShape_bf16 = cutlass::gemm::GemmShape<128, 128, 64>;
using WarpShape_bf16 = cutlass::gemm::GemmShape<64, 64, 64>;
using InstructionShape_bf16 = cutlass::gemm::GemmShape<16, 8, 32>;

// Define MMA & Epilogue
using ThreadblockSwizzle = cutlass::gemm::threadblock::GemmIdentityThreadblockSwizzle<>;

// DefaultConfigurations for float & bf16
using DefaultConfig = cutlass::gemm::device::DefaultGemmConfiguration<
    cutlass::arch::OpClassTensorOp, cutlass::arch::Sm80, float, float, float, float>;

// Pipeline stages in GEMM
constexpr int NumStages = 3;

// A structure to switch between different configurations
template<typename Element_, typename LayoutB_, bool Trans_>
struct SpMMConfigure{

    static const bool Trans = Trans_;
    using LayoutB = LayoutB_;
    using Element = Element_;

    using EpilogueOp = cutlass::epilogue::thread::LinearCombination<
    Element, 128 / cutlass::sizeof_bits<Element>::value, float, float,
    cutlass::epilogue::thread::ScaleType::OnlyAlphaScaling>;

    using Mma = typename cutlass::gemm::threadblock::DefaultSparseMma<
    Element, cutlass::layout::RowMajor, 128 / cutlass::sizeof_bits<Element>::value,
    Element, LayoutB, 128 / cutlass::sizeof_bits<Element>::value,
    float, cutlass::layout::RowMajor, cutlass::arch::OpClassTensorOp, cutlass::arch::Sm80,
    ThreadblockShape_bf16, WarpShape_bf16, InstructionShape_bf16, NumStages, cutlass::arch::OpMultiplyAdd>::ThreadblockMma;

    using Epilogue = typename std::conditional<Trans, 
        typename cutlass::epilogue::threadblock::DefaultTransposeEpilogue<
            ThreadblockShape_bf16, WarpShape_bf16, EpilogueOp, Mma>::Epilogue,
        typename cutlass::epilogue::threadblock::DefaultEpilogueTensorOp<
            ThreadblockShape_bf16, typename Mma::Operator, ThreadblockShape_bf16::kK / WarpShape_bf16::kK, EpilogueOp,
            EpilogueOp::kCount>::Epilogue>::type;
    
    union SharedStorage {
        typename Mma::SharedStorage main_loop;
        typename Epilogue::SharedStorage epilogue;
    }; 
};


template<typename Element, typename _Mma, typename _SharedStorage, typename _Epilogue>
__device__ void cutlassSpmmKernel_16_(
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

    // Compute initial location in logical coordinates
    cutlass::MatrixCoord tb_offset_A{
        threadblock_tile_offset.m() * _Mma::Shape::kM,
        threadblock_tile_offset.k() * gemm_k_size / _Mma::kSparse
    };

    cutlass::MatrixCoord tb_offset_B{
        threadblock_tile_offset.k() * gemm_k_size,
        threadblock_tile_offset.n() * _Mma::Shape::kN
    };

    cutlass::MatrixCoord tb_offset_E{
        threadblock_tile_offset.m() * _Mma::Shape::kM,
        threadblock_tile_offset.k() * gemm_k_size / _Mma::kSparse
    };

    // Problem size
    int problem_size_k = min(problem_size.k(), (threadblock_tile_offset.k() + 1) * gemm_k_size);

    int gemm_k_iterations = (problem_size_k - tb_offset_B.row() + _Mma::Shape::kK - 1) / _Mma::Shape::kK;

    // Compute position within threadblock
    int thread_idx = threadIdx.x;

    // Construct iterators to A, B, and E operands
    typename _Mma::IteratorA iterator_A(
        params_A,
        //ref_A.data(),
        ptr_A,
        {problem_size.m(), problem_size_k / _Mma::kSparse},
        thread_idx,
        tb_offset_A
    );

    typename _Mma::IteratorB iterator_B(
        params_B,
        //ref_B.data(),
        ptr_B,
        {problem_size_k, problem_size.n()},
        thread_idx,
        tb_offset_B
    );

    typename _Mma::IteratorE iterator_E(
        params_E,
        // ref_E.data(),
        ptr_E,
        {problem_size.m(),
        problem_size_k / _Mma::kSparse / _Mma::kElementsPerElementE},
        thread_idx,
        tb_offset_E
    );

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

    
    typename _Epilogue::OutputTileIterator iterator_D(
        params_D,
        ptr_D,
        problem_size.mn(),
        thread_idx,
        threadblock_offset
    );

    
    _Epilogue epilogue(
        shared_storage.epilogue,
        thread_idx,
        warp_idx,
        lane_idx
    );

    epilogue(output_op, iterator_D, accumulators, iterator_D);
}


template<typename Element, typename Config>
__global__ void cutlassSpmmKernel_16(
    cutlass::gemm::GemmCoord problem_size,
    cutlass::gemm::GemmCoord grid_tiled_shape,
    typename Config::Mma::IteratorA::Params params_A,
    Element* __restrict__ ptr_A,
    typename Config::Mma::IteratorB::Params params_B,
    Element* __restrict__ ptr_B,
    typename Config::Epilogue::OutputTileIterator::Params params_D,
    Element* __restrict__ ptr_D,
    typename Config::Mma::IteratorE::Params params_E,
    typename Config::Mma::ElementE* __restrict__ ptr_E,
    typename Config::Epilogue::OutputOp::Params output_op_,
    int gemm_k_size)
{
    cutlassSpmmKernel_16_<Element, Config::Mma, Config::SharedStorage, Config::Epilogue>(
        problem_size, grid_tiled_shape,
        params_A, ptr_A, params_B, ptr_B,
        params_D, ptr_D, params_E, ptr_E,
        output_op_, gemm_k_size
    );
}


template<typename Config>
torch::Tensor spmm_cuda(
    torch::Tensor tensor_a,
    torch::Tensor tensor_b,
    torch::Tensor tensor_e)
{
    int m, n, k;
    m = tensor_a.size(0);
    if (std::is_same<typename Config::LayoutB, cutlass::layout::RowMajor>::value){
        n = tensor_b.size(1); 
        k = tensor_b.size(0);   
    } else {
        n = tensor_b.size(0);
        k = tensor_b.size(1);
    }

    auto options_val = torch::TensorOptions().dtype(tensor_a.dtype()).device(tensor_b.device());
    torch::Tensor output_matrix;
    if (Config::Trans){
        output_matrix = torch::empty({n, m}, options_val);
    } else {
        output_matrix = torch::empty({m, n}, options_val);
    }

    // Create a tuple of problem size for matrix multiplication
    cutlass::gemm::GemmCoord problem_size(m, n, k);

    auto layout_a = cutlass::layout::RowMajor::packed(cutlass::make_Coord(problem_size.m(), problem_size.k() / 2));
    auto layout_b = Config::LayoutB::packed(problem_size.kn());
    auto layout_e = Config::Mma::LayoutE::packed(cutlass::make_Coord(problem_size.m(), problem_size.k()/Config::Mma::kSparse / Config::Mma::kElementsPerElementE));
    auto layout_d = cutlass::layout::RowMajor::packed(problem_size.mn());

    typename Config::Element alpha = typename Config::Element(1.0);
    typename Config::Element beta = typename Config::Element(0.0);
    
    ThreadblockSwizzle threadblock_swizzle;

    cutlass::gemm::GemmCoord grid_tiled_shape = threadblock_swizzle.get_tiled_shape(
        problem_size,
        {ThreadblockShape_bf16::kM, ThreadblockShape_bf16::kN, ThreadblockShape_bf16::kK},
        1
    );

    dim3 grid = threadblock_swizzle.get_grid_shape(grid_tiled_shape);
    dim3 block(Config::Mma::WarpCount::kCount * 32, 1, 1);

    int smem_size = int(sizeof(typename Config::SharedStorage));

    cudaFuncSetAttribute(cutlassSpmmKernel_16<typename Config::Element, Config>, cudaFuncAttributeMaxDynamicSharedMemorySize, smem_size);
    cudaFuncSetAttribute(cutlassSpmmKernel_16<typename Config::Element, Config>, cudaFuncAttributePreferredSharedMemoryCarveout, 100);

    int gemm_k_size = ((problem_size.k() + Config::Mma::Shape::kK - 1) / Config::Mma::Shape::kK) * Config::Mma::Shape::kK;

    cutlassSpmmKernel_16<typename Config::Element, Config><<<grid, block, smem_size>>>(
        problem_size, grid_tiled_shape, 
        layout_a, (typename Config::Element*)tensor_a.data_ptr(),
        layout_b, (typename Config::Element*)tensor_b.data_ptr(),
        layout_d, (typename Config::Element*)output_matrix.data_ptr(),
        layout_e, (typename Config::Mma::ElementE*)tensor_e.data_ptr(),
        {alpha, beta}, gemm_k_size);

    return output_matrix;
}


torch::Tensor spmmv2_bf16_nnn_cuda(
    torch::Tensor tensor_a,
    torch::Tensor tensor_b,
    torch::Tensor tensor_e_reordered)
{
    using Config = SpMMConfigure<cutlass::bfloat16_t, cutlass::layout::RowMajor, false>;
    return spmm_cuda<Config>(tensor_a, tensor_b, tensor_e_reordered);
}


torch::Tensor spmmv2_bf16_ntn_cuda(
    torch::Tensor tensor_a,
    torch::Tensor tensor_b,
    torch::Tensor tensor_e_reordered)
{
    using Config = SpMMConfigure<cutlass::bfloat16_t, cutlass::layout::ColumnMajor, false>;
    return spmm_cuda<Config>(tensor_a, tensor_b, tensor_e_reordered);
}


torch::Tensor spmmv2_bf16_ntt_cuda(
    torch::Tensor tensor_a,
    torch::Tensor tensor_b,
    torch::Tensor tensor_e_reordered)
{
    using Config = SpMMConfigure<cutlass::bfloat16_t, cutlass::layout::ColumnMajor, true>;
    return spmm_cuda<Config>(tensor_a, tensor_b, tensor_e_reordered);
}


torch::Tensor spmmv2_f16_nnn_cuda(
    torch::Tensor tensor_a,
    torch::Tensor tensor_b,
    torch::Tensor tensor_e_reordered)
{
    using Config = SpMMConfigure<cutlass::half_t, cutlass::layout::RowMajor, false>;
    return spmm_cuda<Config>(tensor_a, tensor_b, tensor_e_reordered);
}


torch::Tensor spmmv2_f16_ntn_cuda(
    torch::Tensor tensor_a,
    torch::Tensor tensor_b,
    torch::Tensor tensor_e_reordered)
{
    using Config = SpMMConfigure<cutlass::half_t, cutlass::layout::ColumnMajor, false>;
    return spmm_cuda<Config>(tensor_a, tensor_b, tensor_e_reordered);
}


torch::Tensor spmmv2_f16_ntt_cuda(
    torch::Tensor tensor_a,
    torch::Tensor tensor_b,
    torch::Tensor tensor_e_reordered)
{
    using Config = SpMMConfigure<cutlass::half_t, cutlass::layout::ColumnMajor, true>;
    return spmm_cuda<Config>(tensor_a, tensor_b, tensor_e_reordered);
}