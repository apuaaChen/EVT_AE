#pragma once

#include "cutlass/cutlass.h"
#include "softmax/threadblock/layernorm_backward_reduction.h"

/////////////////////////////////////////////////////////////////////////////////////////////////

namespace cutlass {
namespace softmax {
namespace kernel {

/////////////////////////////////////////////////////////////////////////////////////////////////

template<
    typename Reduction_,
    typename Epilogue_>
struct LayerNormBackwardUniversalwithEpilogueVisitorWarp {
public:
    using Reduction = Reduction_;
    using Epilogue = Epilogue_;
    using EpilogueVisitor = typename Epilogue::Visitor;

    using ElementAccumulator = typename Reduction::ElementAccumulator;

    //
    // Structures
    //

    struct Arguments {
        //
        // Data members
        //
        typename Reduction::Arguments reduction_args;
        typename Epilogue::Arguments epilogue_args;
        typename EpilogueVisitor::Arguments epilogue_visitor;
    };

    struct Params {
        //
        // Data members
        //
        typename Reduction::Params reduction_params;
        typename Epilogue::Params epilogue_params;
        typename EpilogueVisitor::Params epilogue_visitor;

        /// Constructs an arguments structure
        Params(
            Arguments const &args
        ):
            reduction_params(args.reduction_args),
            epilogue_params(args.epilogue_args),
            epilogue_visitor(args.epilogue_visitor){ }

    };

    union SharedStorage {
        typename Reduction::SharedStorage reduction_storage;
        typename EpilogueVisitor::SharedStorage visitor;
    }; 

public:

    /// Execute one softmax backward
    CUTLASS_DEVICE
    void operator()(Params const &params, SharedStorage &shared_storage) {
        int thread_idx = threadIdx.x;
        cutlass::MatrixCoord threadblock_offset{
            int(blockIdx.x), int(blockIdx.y)
        };

        Reduction layernorm_backward_reduction(
            params.reduction_params,
            shared_storage.reduction_storage,
            thread_idx,
            threadblock_offset
        );

        // t1 = sum(gamma_grad_y, dim=1)
        // t2 = sum(gamma_grad_y * x_hat, dim=1)
        ElementAccumulator row_sum_t1;
        ElementAccumulator row_sum_t2;

        typename Reduction::InputFragment gamma_grad_y_buffer;
        typename Reduction::InputFragment x_hat_buffer;

        layernorm_backward_reduction(gamma_grad_y_buffer, x_hat_buffer, row_sum_t1, row_sum_t2);

        gemm::GemmCoord threadblock_tile_offset(
            int(blockIdx.x), int(blockIdx.y), int(blockIdx.z)
        );

        /// Epilogue

        EpilogueVisitor epilogue_visitor(
            params.epilogue_visitor,
            shared_storage.visitor,
            threadblock_offset, 
            threadblock_tile_offset,
            thread_idx,
            params.reduction_params.problem_size
        );

        Epilogue epilogue(
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