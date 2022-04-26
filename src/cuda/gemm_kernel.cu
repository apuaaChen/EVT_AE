#include <cuda.h>
#include <torch/extension.h>
#include <iostream>
#include "cutlass/cutlass.h"
#include "cutlass/gemm/device/gemm.h"
#include <stdio.h>
#include <vector>
#include "cuda_bf16.h"
#include "mma/default_mma.h"


using ThreadblockShape_bf16 = cutlass::gemm::GemmShape<128, 128, 64>;
using WarpShape_bf16 = cutlass::gemm::GemmShape<64, 64, 64>;
using InstructionShape_bf16 = cutlass::gemm::GemmShape<16, 8, 16>;

// Define MMA & Epilogue
using ThreadblockSwizzle = cutlass::gemm::threadblock::GemmBatchedIdentityThreadblockSwizzle;

// DefaultConfigurations for float & bf16
using DefaultConfig = cutlass::gemm::device::DefaultGemmConfiguration<
    cutlass::arch::OpClassTensorOp, cutlass::arch::Sm80, float, float, float, float>;

using EpilogueOp_bf16 = cutlass::epilogue::thread::LinearCombination<
    cutlass::bfloat16_t, 128 / cutlass::sizeof_bits<cutlass::bfloat16_t>::value, float, float,
    cutlass::epilogue::thread::ScaleType::OnlyAlphaScaling>;

// Pipeline stages in GEMM
constexpr int NumStages = 3;

// Mma
// using Mma_bf16_nnn = typename cutlass::gemm::threadblock::DefaultMma<
//     cutlass::bfloat16_t, cutlass::layout::RowMajor, 128 / cutlass::sizeof_bits<cutlass::bfloat16_t>::value,
//     cutlass::bfloat16_t, cutlass::layout::RowMajor, 128 / cutlass::sizeof_bits<cutlass::bfloat16_t>::value,
//     float, cutlass::layout::RowMajor, cutlass::arch::OpClassTensorOp, cutlass::arch::Sm80,
//     ThreadblockShape_bf16, WarpShape_bf16, InstructionShape_bf16, NumStages, cutlass::arch::OpMultiplyAdd>::ThreadblockMma;

using Mma_bf16_nnn = typename cutlass::gemm::threadblock::DefaultMmaV2<
    cutlass::bfloat16_t, cutlass::layout::RowMajor, 128 / cutlass::sizeof_bits<cutlass::bfloat16_t>::value,
    cutlass::bfloat16_t, cutlass::layout::RowMajor, 128 / cutlass::sizeof_bits<cutlass::bfloat16_t>::value,
    float, cutlass::layout::RowMajor, cutlass::arch::OpClassTensorOp, cutlass::arch::Sm80,
    ThreadblockShape_bf16, WarpShape_bf16, InstructionShape_bf16, NumStages, cutlass::arch::OpMultiplyAdd>::ThreadblockMma;

// Epilogue
using Epilogue_bf16_nnn = typename cutlass::epilogue::threadblock::DefaultEpilogueTensorOp<
    ThreadblockShape_bf16, typename Mma_bf16_nnn::Operator, ThreadblockShape_bf16::kK / WarpShape_bf16::kK, EpilogueOp_bf16,
    EpilogueOp_bf16::kCount>::Epilogue;

// Shared Storage
union SharedStorage_bf16_nnn {
    typename Mma_bf16_nnn::SharedStorage main_loop;
    typename Epilogue_bf16_nnn::SharedStorage epilogue;
};



template<typename _Mma, typename _SharedStorage, typename _Epilogue>
__global__ void cutlassGemmKernel_bf16(
    cutlass::gemm::GemmCoord problem_size,
    cutlass::gemm::GemmCoord grid_tiled_shape,
    typename _Mma::IteratorA::Params params_A,
    cutlass::bfloat16_t* __restrict__ ptr_A, int64_t a_stride,
    typename _Mma::IteratorB::Params params_B,
    cutlass::bfloat16_t* __restrict__ ptr_B, int64_t b_stride,
    typename _Epilogue::OutputTileIterator::Params params_D,
    cutlass::bfloat16_t* __restrict__ ptr_D, int64_t d_stride,
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
        0// threadblock_tile_offset.k() * gemm_k_size
    };

    cutlass::MatrixCoord tb_offset_B{
        0,// threadblock_tile_offset.k() * gemm_k_size,
        threadblock_tile_offset.n() * _Mma::Shape::kN
    };

    // Problem size
    int problem_size_k =problem_size.k();// min(problem_size.k(), (threadblock_tile_offset.k() + 1) * gemm_k_size);

    int gemm_k_iterations = (problem_size_k - tb_offset_B.row() + _Mma::Shape::kK - 1) / _Mma::Shape::kK;

    // Compute position within threadblock
    int thread_idx = threadIdx.x;

    int batch_idx = threadblock_swizzle.get_batch_idx();

    // Construct iterators to A, B, and E operands
    typename _Mma::IteratorA iterator_A(
        params_A,
        //ref_A.data(),
        ptr_A,
        {problem_size.m(), problem_size_k},
        thread_idx,
        tb_offset_A
    );

    // int64_t a_stride = problem_size.m() * problem_size.k();
    iterator_A.add_pointer_offset(batch_idx * a_stride);

    typename _Mma::IteratorB iterator_B(
        params_B,
        //ref_B.data(),
        ptr_B,
        {problem_size_k, problem_size.n()},
        thread_idx,
        tb_offset_B
    );

    // int64_t b_stride = problem_size.n() * problem_size.k();
    iterator_B.add_pointer_offset(batch_idx * b_stride);

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
        mma(gemm_k_iterations, accumulators, iterator_A, iterator_B, accumulators);
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
    
    // int64_t d_stride = problem_size.m() * problem_size.n();
    iterator_D.add_pointer_offset(batch_idx * d_stride);
    
    _Epilogue epilogue(
        shared_storage.epilogue,
        thread_idx,
        warp_idx,
        lane_idx
    );

    epilogue(output_op, iterator_D, accumulators, iterator_D);
}


torch::Tensor gemm_bf16_nnn_cuda(
    torch::Tensor tensor_a,
    torch::Tensor tensor_b)
{
    const int m = tensor_a.size(-2);
    const int n = tensor_b.size(-1);
    const int k = tensor_b.size(-2);

    int batch_size = tensor_a.numel() / m / k;

    auto options_val = torch::TensorOptions().dtype(torch::kBFloat16).device(tensor_b.device());
    auto output_matrix = torch::empty({batch_size, m, n}, options_val);

    // Create a tuple of problem size for matrix multiplication
    cutlass::gemm::GemmCoord problem_size(m, n, k);

    auto layout_a = cutlass::layout::RowMajor::packed(cutlass::make_Coord(problem_size.m(), problem_size.k()));
    auto layout_b = cutlass::layout::RowMajor::packed(problem_size.kn());
    auto layout_d = cutlass::layout::RowMajor::packed(problem_size.mn());

    cutlass::bfloat16_t alpha = cutlass::bfloat16_t(1.0);
    cutlass::bfloat16_t beta = cutlass::bfloat16_t(0.0);
    
    ThreadblockSwizzle threadblock_swizzle;

    cutlass::gemm::GemmCoord grid_tiled_shape = threadblock_swizzle.get_tiled_shape(
        problem_size,
        {ThreadblockShape_bf16::kM, ThreadblockShape_bf16::kN, ThreadblockShape_bf16::kK},
        batch_size
    );

    dim3 grid = threadblock_swizzle.get_grid_shape(grid_tiled_shape);
    dim3 block(Mma_bf16_nnn::WarpCount::kCount * 32, 1, 1);

    int smem_size = int(sizeof(SharedStorage_bf16_nnn));

    cudaFuncSetAttribute(cutlassGemmKernel_bf16<Mma_bf16_nnn, SharedStorage_bf16_nnn, Epilogue_bf16_nnn>, cudaFuncAttributeMaxDynamicSharedMemorySize, smem_size);
    cudaFuncSetAttribute(cutlassGemmKernel_bf16<Mma_bf16_nnn, SharedStorage_bf16_nnn, Epilogue_bf16_nnn>, cudaFuncAttributePreferredSharedMemoryCarveout, 100);

    int gemm_k_size = ((problem_size.k() + Mma_bf16_nnn::Shape::kK - 1) / Mma_bf16_nnn::Shape::kK) * Mma_bf16_nnn::Shape::kK;

    cutlassGemmKernel_bf16<Mma_bf16_nnn, SharedStorage_bf16_nnn, Epilogue_bf16_nnn><<<grid, block, smem_size>>>(
        problem_size, grid_tiled_shape, 
        layout_a, (cutlass::bfloat16_t*)tensor_a.data_ptr(), m*k,
        layout_b, (cutlass::bfloat16_t*)tensor_b.data_ptr(), n*k,
        layout_d, (cutlass::bfloat16_t*)output_matrix.data_ptr(), m * n,
        {alpha, beta}, gemm_k_size);

    return output_matrix;
}