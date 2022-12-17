#pragma once

#include "cutlass/cutlass.h"
#include "softmax/kernel/reduce_apply_universal.h"
#include "softmax/threadblock/layernorm_backward_reduction.h"

/////////////////////////////////////////////////////////////////////////////////////////////////

namespace cutlass {
namespace softmax {
namespace kernel {

/////////////////////////////////////////////////////////////////////////////////////////////////

template<
    typename Reduction_,
    typename Epilogue_>
struct LayerNormBackwardUniversalwithEpilogueVisitorWarp :
    cutlass::reduce_apply::kernel::ReductionApplywithEpilogueVisitor<Reduction_, Epilogue_> {
public:
    using Base = cutlass::reduce_apply::kernel::ReductionApplywithEpilogueVisitor<Reduction_, Epilogue_>;

public:

    /// Execute one softmax backward
    CUTLASS_DEVICE
    void operator()(typename Base::Params const &params, typename Base::SharedStorage &shared_storage) {
        int thread_idx = threadIdx.x;
        cutlass::MatrixCoord threadblock_offset{
            int(blockIdx.x), int(blockIdx.y)
        };

        typename Base::Reduction layernorm_backward_reduction(
            params.reduction_params,
            shared_storage.reduction_storage,
            thread_idx,
            threadblock_offset
        );

        // t1 = sum(gamma_grad_y, dim=1)
        // t2 = sum(gamma_grad_y * x_hat, dim=1)
        typename Base::ElementAccumulator row_sum_t1;
        typename Base::ElementAccumulator row_sum_t2;

        typename Base::Reduction::InputFragment gamma_grad_y_buffer;
        typename Base::Reduction::InputFragment x_hat_buffer;

        layernorm_backward_reduction(gamma_grad_y_buffer, x_hat_buffer, row_sum_t1, row_sum_t2);

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
        epilogue(epilogue_visitor, gamma_grad_y_buffer, x_hat_buffer, row_sum_t1, row_sum_t2);
    }
};

/////////////////////////////////////////////////////////////////////////////////////////////////

} // namespace kernel
} // namespace softmax
} // namespace cutlass

/////////////////////////////////////////////////////////////////////////////////////////////////