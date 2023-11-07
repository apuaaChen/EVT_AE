#pragma once


// Define MMA & Epilogue
using ThreadblockSwizzle = cutlass::gemm::threadblock::GemmBatchedIdentityThreadblockSwizzle;

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

    int batch_idx = threadblock_swizzle.get_batch_idx();

    // Compute position within threadblock
    int thread_idx = threadIdx.x;
    // Broadcast the warp_id computed by lane 0 to ensure dependent code
    // is compuled as warp-uniform
    int warp_idx = __shfl_sync(0xffffffff, threadIdx.x / 32, 0);
    int lane_idx = threadIdx.x % 32;


    // Compute initial location in logical coordinates
    cutlass::MatrixCoord tb_offset_A{
        threadblock_tile_offset.m() * _Mma::Shape::kM,
        0
    };

    cutlass::MatrixCoord tb_offset_B{
        0,
        threadblock_tile_offset.n() * _Mma::Shape::kN
    };

    cutlass::MatrixCoord tb_offset_E{
        threadblock_tile_offset.m() * _Mma::Shape::kM,
        0
    };

    // Problem size
    int problem_size_k = problem_size.k(); //min(problem_size.k(), (threadblock_tile_offset.k() + 1) * gemm_k_size);

    int gemm_k_iterations = (problem_size_k - tb_offset_B.row() + _Mma::Shape::kK - 1) / _Mma::Shape::kK;

    
    // Construct iterators to A, B, and E operands
    typename _Mma::IteratorA iterator_A(
        params_A,
        //ref_A.data(),
        ptr_A,
        {problem_size.m(), problem_size_k / _Mma::kSparse},
        thread_idx,
        tb_offset_A
    );

    // int64_t a_stride = problem_size.m() * problem_size.k() / _Mma::kSparse;
    iterator_A.add_pointer_offset(batch_idx * problem_size.m() * problem_size.k() / _Mma::kSparse);

    typename _Mma::IteratorB iterator_B(
        params_B,
        //ref_B.data(),
        ptr_B,
        {problem_size_k, problem_size.n()},
        thread_idx,
        tb_offset_B
    );

    // int64_t b_stride = problem_size.n() * problem_size.k();
    iterator_B.add_pointer_offset(batch_idx * problem_size.n() * problem_size.k());

    typename _Mma::IteratorE iterator_E(
        params_E,
        // ref_E.data(),
        ptr_E,
        {problem_size.m(),
        problem_size_k / _Mma::kSparse / _Mma::kElementsPerElementE},
        thread_idx,
        tb_offset_E
    );

    // int64_t e_stride = problem_size.m() * problem_size.k() / _Mma::kSparse / _Mma::kElementsPerElementE;
    iterator_E.add_pointer_offset(batch_idx * problem_size.m() * problem_size.k() / _Mma::kSparse / _Mma::kElementsPerElementE);

    

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

    int64_t d_stride = problem_size.m() * problem_size.n();
    iterator_D.add_pointer_offset(batch_idx * problem_size.m() * problem_size.n());

    
    _Epilogue epilogue(
        shared_storage.epilogue,
        thread_idx,
        warp_idx,
        lane_idx
    );

    epilogue(output_op, iterator_D, accumulators, iterator_D);
}