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

#include "epilogue/default_epilogue_tensor_op.h"


// Define the Tile Size in different levels

using ThreadblockShape_bf16 = cutlass::gemm::GemmShape<128, 128, 64>;
using WarpShape_bf16 = cutlass::gemm::GemmShape<64, 64, 64>;
using InstructionShape_bf16 = cutlass::gemm::GemmShape<16, 8, 32>;

// Define MMA & Epilogue
using ThreadblockSwizzle = cutlass::gemm::threadblock::GemmIdentityThreadblockSwizzle<>;

// DefaultConfigurations for float & bf16
using DefaultConfig = cutlass::gemm::device::DefaultGemmConfiguration<
    cutlass::arch::OpClassTensorOp, cutlass::arch::Sm80, float, float, float, float>;

using EpilogueOp_bf16 = cutlass::epilogue::thread::LinearCombination<
    cutlass::bfloat16_t, 128 / cutlass::sizeof_bits<cutlass::bfloat16_t>::value, float, float,
    cutlass::epilogue::thread::ScaleType::OnlyAlphaScaling>;

// Pipeline stages in GEMM
constexpr int NumStages = 3;

// Mma

using Mma_bf16_nnn = typename cutlass::gemm::threadblock::DefaultSparseMma<
    cutlass::bfloat16_t, cutlass::layout::RowMajor, 128 / cutlass::sizeof_bits<cutlass::bfloat16_t>::value,
    cutlass::bfloat16_t, cutlass::layout::RowMajor, 128 / cutlass::sizeof_bits<cutlass::bfloat16_t>::value,
    float, cutlass::layout::RowMajor, cutlass::arch::OpClassTensorOp, cutlass::arch::Sm80,
    ThreadblockShape_bf16, WarpShape_bf16, InstructionShape_bf16, NumStages, cutlass::arch::OpMultiplyAdd>::ThreadblockMma;

using Mma_bf16_ntn = typename cutlass::gemm::threadblock::DefaultSparseMma<
    cutlass::bfloat16_t, cutlass::layout::RowMajor, 128 / cutlass::sizeof_bits<cutlass::bfloat16_t>::value,
    cutlass::bfloat16_t, cutlass::layout::ColumnMajor, 128 / cutlass::sizeof_bits<cutlass::bfloat16_t>::value,
    float, cutlass::layout::RowMajor, cutlass::arch::OpClassTensorOp, cutlass::arch::Sm80,
    ThreadblockShape_bf16, WarpShape_bf16, InstructionShape_bf16, NumStages, cutlass::arch::OpMultiplyAdd>::ThreadblockMma;

using Mma_bf16_ntt = typename cutlass::gemm::threadblock::DefaultSparseMma<
    cutlass::bfloat16_t, cutlass::layout::RowMajor, 128 / cutlass::sizeof_bits<cutlass::bfloat16_t>::value,
    cutlass::bfloat16_t, cutlass::layout::ColumnMajor, 128 / cutlass::sizeof_bits<cutlass::bfloat16_t>::value,
    float, cutlass::layout::RowMajor, cutlass::arch::OpClassTensorOp, cutlass::arch::Sm80,
    ThreadblockShape_bf16, WarpShape_bf16, InstructionShape_bf16, NumStages, cutlass::arch::OpMultiplyAdd>::ThreadblockMma;

// Epilogue

using Epilogue_bf16_nnn = typename cutlass::epilogue::threadblock::DefaultEpilogueTensorOp<
    ThreadblockShape_bf16, typename Mma_bf16_nnn::Operator, ThreadblockShape_bf16::kK / WarpShape_bf16::kK, EpilogueOp_bf16,
    EpilogueOp_bf16::kCount>::Epilogue;

using Epilogue_bf16_ntn = typename cutlass::epilogue::threadblock::DefaultEpilogueTensorOp<
    ThreadblockShape_bf16, typename Mma_bf16_ntn::Operator, ThreadblockShape_bf16::kK / WarpShape_bf16::kK, EpilogueOp_bf16,
    EpilogueOp_bf16::kCount>::Epilogue;

using Epilogue_bf16_ntt = typename cutlass::epilogue::threadblock::DefaultEpilogueTensorOpV2<
    ThreadblockShape_bf16, typename Mma_bf16_ntt::Operator, ThreadblockShape_bf16::kK / WarpShape_bf16::kK, EpilogueOp_bf16,
    EpilogueOp_bf16::kCount>::Epilogue;

// Shared Storage

union SharedStorage_bf16_nnn {
    typename Mma_bf16_nnn::SharedStorage main_loop;
    typename Epilogue_bf16_nnn::SharedStorage epilogue;
};

union SharedStorage_bf16_ntn {
    typename Mma_bf16_ntn::SharedStorage main_loop;
    typename Epilogue_bf16_ntn::SharedStorage epilogue;
};

union SharedStorage_bf16_ntt {
    typename Mma_bf16_ntt::SharedStorage main_loop;
    typename Epilogue_bf16_ntt::SharedStorage epilogue;
};


template<typename _Mma, typename _SharedStorage, typename _Epilogue>
__device__ void cutlassSpmmKernel_bf16_(
    cutlass::gemm::GemmCoord problem_size,
    cutlass::gemm::GemmCoord grid_tiled_shape,
    typename _Mma::IteratorA::Params params_A,
    cutlass::bfloat16_t* __restrict__ ptr_A,
    typename _Mma::IteratorB::Params params_B,
    cutlass::bfloat16_t* __restrict__ ptr_B,
    typename _Epilogue::OutputTileIterator::Params params_D,
    cutlass::bfloat16_t* __restrict__ ptr_D,
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

    int block_idx = threadblock_tile_offset.m() + threadblock_tile_offset.n() * grid_tiled_shape.m();
    
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


template<typename _Mma, typename _SharedStorage, typename _Epilogue>
__global__ void cutlassSpmmKernel_bf16_test(
    cutlass::gemm::GemmCoord problem_size,
    cutlass::gemm::GemmCoord grid_tiled_shape,
    typename _Mma::IteratorA::Params params_A,
    cutlass::bfloat16_t* __restrict__ ptr_A,
    typename _Mma::IteratorB::Params params_B,
    cutlass::bfloat16_t* __restrict__ ptr_B,
    typename _Epilogue::OutputTileIterator::Params params_D,
    cutlass::bfloat16_t* __restrict__ ptr_D,
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

    /* Row-major output
    int warp_row_idx = warp_idx % 2;
    int warp_col_idx = warp_idx / 2;
    int quad_idx = lane_idx / 4;
    int idx_in_quad = lane_idx % 4;

    int global_offset = (warp_row_idx * 64 + quad_idx) * problem_size.n() + warp_col_idx * 64 + idx_in_quad * 2;


    float2 *frag_ptr = reinterpret_cast<float2 *>(&accumulators);
    float *global_ptr = reinterpret_cast<float *>(ptr_D + 128 * blockIdx.x * problem_size.n() + 128 * blockIdx.y + global_offset);

    float2 tmp;
    __nv_bfloat16 res[2];

    float* res_vec = reinterpret_cast<float *>(res);

    #pragma unroll
    for (int row=0; row < 8; row ++){
        float *global_ptr_t = global_ptr;
        float2 *frag_ptr_t = frag_ptr;
        #pragma unroll
        for (int col=0; col < 8; col ++){
            tmp = *(frag_ptr_t);
            res[0] = __float2bfloat16(tmp.x);
            res[1] = __float2bfloat16(tmp.y);
            *(global_ptr_t) = *(res_vec);
            frag_ptr_t += 8;
            global_ptr_t += 4;
        }
        global_ptr += problem_size.n() * 4;
        frag_ptr += 1;
    }
    */

    //// Transposed epilogue
    // Step 1: cast results to bfloat16

    int col_quad_idx = lane_idx % 8;
    int row_quad_idx = lane_idx / 8;
    int col_major = col_quad_idx % 4;
    int col_minor = col_quad_idx / 4;
    int quad_idx = lane_idx / 4;
    int warp_row_idx = warp_idx % 2;
    int warp_col_idx = warp_idx / 2;
    int idx_in_quad = lane_idx % 4;

    int sub_warp_idx = lane_idx / 16;
    int sub_lane_idx = lane_idx % 16;

    constexpr int kShmStride = 64 + 8;

    float2 *frag_ptr = reinterpret_cast<float2 *>(&accumulators);

    int shared_store_offset = (warp_col_idx * 8 + col_major * 2 + col_minor) * kShmStride + warp_row_idx * 32 + row_quad_idx;

    int shared_load_offset = (warp_row_idx * 4 + warp_col_idx * 8 + sub_warp_idx) * kShmStride / 4 + sub_lane_idx;

    float *shared_store_ptr = reinterpret_cast<float *>(SharedStorageBase) + shared_store_offset;
    float4 *shared_load_ptr = reinterpret_cast<float4 *>(SharedStorageBase) + shared_load_offset;


    int global_offset = (warp_row_idx * 4 + warp_col_idx * 64 + sub_warp_idx) * problem_size.m() / 8 + sub_lane_idx;
    float4 *global_ptr = reinterpret_cast<float4 *>(ptr_D + 128 * blockIdx.y * problem_size.m() + 128 * blockIdx.x) + global_offset;

    __nv_bfloat16 tmp_share;
    __nv_bfloat16 res[2];

    float* res_vec = reinterpret_cast<float* >(res);

    #pragma unroll
    for (int col=0; col < 8; col ++){
        float2 *frag_ptr_t = frag_ptr;
        float * shared_store_ptr_t = shared_store_ptr;
        __syncthreads();

        #pragma unroll
        for (int row=0; row < 8; row ++){
            if (quad_idx % 2 == 0){
                tmp_share = __float2bfloat16((*(frag_ptr_t)).y);
            } else {
                tmp_share = __float2bfloat16((*(frag_ptr_t)).x);
            }
            tmp_share = __shfl_xor_sync(0xffffffff, tmp_share, 4);
            if (quad_idx % 2 == 0){
                res[0] = __float2bfloat16((*(frag_ptr_t)).x);
                res[1] = tmp_share;
            } else {
                res[0] = tmp_share;
                res[1] = __float2bfloat16((*(frag_ptr_t)).y);
            }
            *(shared_store_ptr_t) = *(res_vec);
            frag_ptr_t += 1;
            shared_store_ptr_t += 4;
        }
        __syncthreads();
        // To this point, the results have been written into the shared memory
        // Then we load them back to registers
        *(global_ptr) = *(shared_load_ptr);

        *(global_ptr + problem_size.m() / 4) = *(shared_load_ptr + kShmStride / 2);
        global_ptr += problem_size.m();
        frag_ptr += 8;
    }



    // typename _Epilogue::OutputOp output_op(output_op_);

    // threadblock_tile_offset = threadblock_swizzle.get_tile_offset(grid_tiled_shape);

    // // (blockIdx.x * TileM, blockIdx.y * TileN)
    // cutlass::MatrixCoord threadblock_offset(
    //     threadblock_tile_offset.m() * _Mma::Shape::kM,
    //     threadblock_tile_offset.n() * _Mma::Shape::kN
    // );

    // int block_idx = threadblock_tile_offset.m() + threadblock_tile_offset.n() * grid_tiled_shape.m();
    
    // typename _Epilogue::OutputTileIterator iterator_D(
    //     params_D,
    //     ptr_D,
    //     problem_size.mn(),
    //     thread_idx,
    //     threadblock_offset
    // );

    
    // _Epilogue epilogue(
    //     shared_storage.epilogue,
    //     thread_idx,
    //     warp_idx,
    //     lane_idx
    // );

    // epilogue(output_op, iterator_D, accumulators, iterator_D);
}

template<typename _Mma, typename _SharedStorage, typename _Epilogue>
__global__ void cutlassSpmmKernel_bf16(
    cutlass::gemm::GemmCoord problem_size,
    cutlass::gemm::GemmCoord grid_tiled_shape,
    typename _Mma::IteratorA::Params params_A,
    cutlass::bfloat16_t* __restrict__ ptr_A,
    typename _Mma::IteratorB::Params params_B,
    cutlass::bfloat16_t* __restrict__ ptr_B,
    typename _Epilogue::OutputTileIterator::Params params_D,
    cutlass::bfloat16_t* __restrict__ ptr_D,
    typename _Mma::IteratorE::Params params_E,
    typename _Mma::ElementE* __restrict__ ptr_E,
    typename _Epilogue::OutputOp::Params output_op_,
    int gemm_k_size)
{
    cutlassSpmmKernel_bf16_<_Mma, _SharedStorage, _Epilogue>(
        problem_size, grid_tiled_shape,
        params_A, ptr_A, params_B, ptr_B,
        params_D, ptr_D, params_E, ptr_E,
        output_op_, gemm_k_size
    );
}


torch::Tensor spmmv2_bf16_nnn_cuda(
    torch::Tensor tensor_a,
    torch::Tensor tensor_b,
    torch::Tensor tensor_e_reordered)
{
    const int m = tensor_a.size(0);
    const int n = tensor_b.size(1);
    const int k = tensor_b.size(0);

    auto options_val = torch::TensorOptions().dtype(torch::kBFloat16).device(tensor_b.device());
    auto output_matrix = torch::empty({m, n}, options_val);

    // Create a tuple of problem size for matrix multiplication
    cutlass::gemm::GemmCoord problem_size(m, n, k);

    auto layout_a = cutlass::layout::RowMajor::packed(cutlass::make_Coord(problem_size.m(), problem_size.k() / 2));
    auto layout_b = cutlass::layout::RowMajor::packed(problem_size.kn());
    auto layout_e = Mma_bf16_nnn::LayoutE::packed(cutlass::make_Coord(problem_size.m(), problem_size.k()/Mma_bf16_nnn::kSparse / Mma_bf16_nnn::kElementsPerElementE));
    auto layout_d = cutlass::layout::RowMajor::packed(problem_size.mn());

    cutlass::bfloat16_t alpha = cutlass::bfloat16_t(1.0);
    cutlass::bfloat16_t beta = cutlass::bfloat16_t(0.0);
    
    ThreadblockSwizzle threadblock_swizzle;

    cutlass::gemm::GemmCoord grid_tiled_shape = threadblock_swizzle.get_tiled_shape(
        problem_size,
        {ThreadblockShape_bf16::kM, ThreadblockShape_bf16::kN, ThreadblockShape_bf16::kK},
        1
    );

    dim3 grid = threadblock_swizzle.get_grid_shape(grid_tiled_shape);
    dim3 block(Mma_bf16_nnn::WarpCount::kCount * 32, 1, 1);

    int smem_size = int(sizeof(SharedStorage_bf16_nnn));

    cudaFuncSetAttribute(cutlassSpmmKernel_bf16<Mma_bf16_nnn, SharedStorage_bf16_nnn, Epilogue_bf16_nnn>, cudaFuncAttributeMaxDynamicSharedMemorySize, smem_size);
    cudaFuncSetAttribute(cutlassSpmmKernel_bf16<Mma_bf16_nnn, SharedStorage_bf16_nnn, Epilogue_bf16_nnn>, cudaFuncAttributePreferredSharedMemoryCarveout, 100);

    int gemm_k_size = ((problem_size.k() + Mma_bf16_nnn::Shape::kK - 1) / Mma_bf16_nnn::Shape::kK) * Mma_bf16_nnn::Shape::kK;

    cutlassSpmmKernel_bf16<Mma_bf16_nnn, SharedStorage_bf16_nnn, Epilogue_bf16_nnn><<<grid, block, smem_size>>>(
        problem_size, grid_tiled_shape, 
        layout_a, (cutlass::bfloat16_t*)tensor_a.data_ptr(),
        layout_b, (cutlass::bfloat16_t*)tensor_b.data_ptr(),
        layout_d, (cutlass::bfloat16_t*)output_matrix.data_ptr(),
        layout_e, (Mma_bf16_nnn::ElementE*)tensor_e_reordered.data_ptr(),
        {alpha, beta}, gemm_k_size);

    return output_matrix;
}


torch::Tensor spmmv2_bf16_ntn_cuda(
    torch::Tensor tensor_a,
    torch::Tensor tensor_b,
    torch::Tensor tensor_e_reordered)
{
    const int m = tensor_a.size(0);
    const int n = tensor_b.size(0);
    const int k = tensor_b.size(1);

    auto options_val = torch::TensorOptions().dtype(torch::kBFloat16).device(tensor_b.device());
    auto output_matrix = torch::empty({m, n}, options_val);

    // Create a tuple of problem size for matrix multiplication
    cutlass::gemm::GemmCoord problem_size(m, n, k);

    auto layout_a = cutlass::layout::RowMajor::packed(cutlass::make_Coord(problem_size.m(), problem_size.k() / 2));
    auto layout_b = cutlass::layout::ColumnMajor::packed(problem_size.kn());
    auto layout_e = Mma_bf16_ntn::LayoutE::packed(cutlass::make_Coord(problem_size.m(), problem_size.k()/Mma_bf16_ntn::kSparse / Mma_bf16_ntn::kElementsPerElementE));
    auto layout_d = cutlass::layout::RowMajor::packed(problem_size.mn());

    cutlass::bfloat16_t alpha = cutlass::bfloat16_t(1.0);
    cutlass::bfloat16_t beta = cutlass::bfloat16_t(0.0);
    
    ThreadblockSwizzle threadblock_swizzle;

    cutlass::gemm::GemmCoord grid_tiled_shape = threadblock_swizzle.get_tiled_shape(
        problem_size,
        {ThreadblockShape_bf16::kM, ThreadblockShape_bf16::kN, ThreadblockShape_bf16::kK},
        1
    );

    dim3 grid = threadblock_swizzle.get_grid_shape(grid_tiled_shape);
    dim3 block(Mma_bf16_ntn::WarpCount::kCount * 32, 1, 1);

    int smem_size = int(sizeof(SharedStorage_bf16_ntn));

    cudaFuncSetAttribute(cutlassSpmmKernel_bf16<Mma_bf16_ntn, SharedStorage_bf16_ntn, Epilogue_bf16_ntn>, cudaFuncAttributeMaxDynamicSharedMemorySize, smem_size);
    cudaFuncSetAttribute(cutlassSpmmKernel_bf16<Mma_bf16_ntn, SharedStorage_bf16_ntn, Epilogue_bf16_ntn>, cudaFuncAttributePreferredSharedMemoryCarveout, 100);

    int gemm_k_size = ((problem_size.k() + Mma_bf16_ntn::Shape::kK - 1) / Mma_bf16_ntn::Shape::kK) * Mma_bf16_ntn::Shape::kK;

    cutlassSpmmKernel_bf16<Mma_bf16_ntn, SharedStorage_bf16_ntn, Epilogue_bf16_ntn><<<grid, block, smem_size>>>(
        problem_size, grid_tiled_shape, 
        layout_a, (cutlass::bfloat16_t*)tensor_a.data_ptr(),
        layout_b, (cutlass::bfloat16_t*)tensor_b.data_ptr(),
        layout_d, (cutlass::bfloat16_t*)output_matrix.data_ptr(),
        layout_e, (Mma_bf16_ntn::ElementE*)tensor_e_reordered.data_ptr(),
        {alpha, beta}, gemm_k_size);

    return output_matrix;
}


torch::Tensor spmmv2_bf16_ntt_cuda(
    torch::Tensor tensor_a,
    torch::Tensor tensor_b,
    torch::Tensor tensor_e_reordered)
{
    const int m = tensor_a.size(0);
    const int n = tensor_b.size(0);
    const int k = tensor_b.size(1);

    auto options_val = torch::TensorOptions().dtype(torch::kBFloat16).device(tensor_b.device());
    auto output_matrix = torch::empty({n, m}, options_val);

    // Create a tuple of problem size for matrix multiplication
    cutlass::gemm::GemmCoord problem_size(m, n, k);

    auto layout_a = cutlass::layout::RowMajor::packed(cutlass::make_Coord(problem_size.m(), problem_size.k() / 2));
    auto layout_b = cutlass::layout::ColumnMajor::packed(problem_size.kn());
    auto layout_e = Mma_bf16_ntt::LayoutE::packed(cutlass::make_Coord(problem_size.m(), problem_size.k()/Mma_bf16_ntt::kSparse / Mma_bf16_ntt::kElementsPerElementE));
    auto layout_d = cutlass::layout::RowMajor::packed(problem_size.mn());

    cutlass::bfloat16_t alpha = cutlass::bfloat16_t(1.0);
    cutlass::bfloat16_t beta = cutlass::bfloat16_t(0.0);
    
    ThreadblockSwizzle threadblock_swizzle;

    cutlass::gemm::GemmCoord grid_tiled_shape = threadblock_swizzle.get_tiled_shape(
        problem_size,
        {ThreadblockShape_bf16::kM, ThreadblockShape_bf16::kN, ThreadblockShape_bf16::kK},
        1
    );

    dim3 grid = threadblock_swizzle.get_grid_shape(grid_tiled_shape);
    dim3 block(Mma_bf16_ntt::WarpCount::kCount * 32, 1, 1);

    int smem_size = int(sizeof(SharedStorage_bf16_ntt));

    cudaFuncSetAttribute(cutlassSpmmKernel_bf16_test<Mma_bf16_ntt, SharedStorage_bf16_ntt, Epilogue_bf16_ntt>, cudaFuncAttributeMaxDynamicSharedMemorySize, smem_size);
    cudaFuncSetAttribute(cutlassSpmmKernel_bf16_test<Mma_bf16_ntt, SharedStorage_bf16_ntt, Epilogue_bf16_ntt>, cudaFuncAttributePreferredSharedMemoryCarveout, 100);

    int gemm_k_size = ((problem_size.k() + Mma_bf16_ntt::Shape::kK - 1) / Mma_bf16_ntt::Shape::kK) * Mma_bf16_ntt::Shape::kK;

    cutlassSpmmKernel_bf16_test<Mma_bf16_ntt, SharedStorage_bf16_ntt, Epilogue_bf16_ntt><<<grid, block, smem_size>>>(
        problem_size, grid_tiled_shape, 
        layout_a, (cutlass::bfloat16_t*)tensor_a.data_ptr(),
        layout_b, (cutlass::bfloat16_t*)tensor_b.data_ptr(),
        layout_d, (cutlass::bfloat16_t*)output_matrix.data_ptr(),
        layout_e, (Mma_bf16_ntt::ElementE*)tensor_e_reordered.data_ptr(),
        {alpha, beta}, gemm_k_size);

    return output_matrix;
}