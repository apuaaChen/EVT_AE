#pragma once

#include "cutlass/cutlass.h"

#include "softmax/kernel/reduce_apply_universal.h"
#include "softmax/threadblock/softmax_reduction.h"

/////////////////////////////////////////////////////////////////////////////////////////////////

namespace cutlass {
namespace softmax {
namespace kernel {

/////////////////////////////////////////////////////////////////////////////////////////////////

template<
    typename Reduction_,
    typename Epilogue_>
struct SoftmaxUniversalwithEpilogueVisitorBlock :
   cutlass::reduce_apply::kernel::ReductionApplywithEpilogueVisitor<Reduction_, Epilogue_> {
public:
    using Base = cutlass::reduce_apply::kernel::ReductionApplywithEpilogueVisitor<Reduction_, Epilogue_>;

    /// Execute one softmax
    CUTLASS_DEVICE
    void operator()(typename Base::Params const &params, typename Base::SharedStorage &shared_storage) {

        int thread_idx = threadIdx.x;
        cutlass::MatrixCoord threadblock_offset{
            int(blockIdx.x), int(blockIdx.y)
        };

        typename Base::Reduction softmax_reduction(
            params.reduction_params, 
            shared_storage.reduction_storage, 
            thread_idx,
            threadblock_offset
        );

        typename Base::ElementAccumulator row_max;
        typename Base::ElementAccumulator row_sum;

        softmax_reduction(row_max, row_sum);

        gemm::GemmCoord threadblock_tile_offset(
            int(blockIdx.x), int(blockIdx.y), int(blockIdx.z)
        );
     
        /// Epilogue

        typename Base::EpilogueVisitor epilogue_visitor(
            params.epilogue_visitor,
            shared_storage.visitor,
            threadblock_offset, 
            threadblock_tile_offset,
            thread_idx,
            params.reduction_params.problem_size
        );

        typename Base::Epilogue epilogue(
            params.epilogue_params,
            thread_idx,
            threadblock_offset
        );

        // Execute the epilogue operator to update the destination tensor
        epilogue(epilogue_visitor, row_max, row_sum);

    }
};


template<
    typename Reduction_,
    typename Epilogue_>
struct SoftmaxUniversalwithEpilogueVisitorWarp :
    cutlass::reduce_apply::kernel::ReductionApplywithEpilogueVisitor<Reduction_, Epilogue_>{
public:
    using Base = cutlass::reduce_apply::kernel::ReductionApplywithEpilogueVisitor<Reduction_, Epilogue_>;

    /// Execute one softmax
    CUTLASS_DEVICE
    void operator()(typename Base::Params const &params, typename Base::SharedStorage &shared_storage) {

        int thread_idx = threadIdx.x;
        cutlass::MatrixCoord threadblock_offset{
            int(blockIdx.x), int(blockIdx.y)
        };

        typename Base::Reduction softmax_reduction(
            params.reduction_params, 
            shared_storage.reduction_storage, 
            thread_idx,
            threadblock_offset
        );

        typename Base::ElementAccumulator row_max;
        typename Base::ElementAccumulator row_sum;
        typename Base::Reduction::InputFragment input_buffer;

        softmax_reduction(input_buffer, row_max, row_sum);

        gemm::GemmCoord threadblock_tile_offset(
            int(blockIdx.x), int(blockIdx.y), int(blockIdx.z)
        );

        
        /// Epilogue


        typename Base::EpilogueVisitor epilogue_visitor(
            params.epilogue_visitor,
            shared_storage.visitor,
            threadblock_offset, 
            threadblock_tile_offset,
            thread_idx,
            params.reduction_params.problem_size
        );

        typename Base::Epilogue epilogue(
            params.epilogue_params,
            thread_idx,
            threadblock_offset
        );

        // Execute the epilogue operator to update the destination tensor
        epilogue(epilogue_visitor, input_buffer, row_max, row_sum);

    }
};


/////////////////////////////////////////////////////////////////////////////////////////////////

} // namespace kernel
} // namespace softmax
} // namespace cutlass

/////////////////////////////////////////////////////////////////////////////////////////////////