#pragma once

#include "cutlass/cutlass.h"
#include "softmax/threadblock/softmax_backward_reduction.h"

/////////////////////////////////////////////////////////////////////////////////////////////////

namespace cutlass {
namespace softmax {
namespace kernel {

/////////////////////////////////////////////////////////////////////////////////////////////////

template<
    typename Reduction_,
    typename Epilogue_>
struct SoftmaxBackwardUniversalwithEpilogueVisitorBlock {
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

        Reduction softmax_backward_reduction(
            params.reduction_params,
            shared_storage.reduction_storage,
            thread_idx,
            threadblock_offset
        );

        ElementAccumulator row_sum;

        softmax_backward_reduction(row_sum);

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
        epilogue(epilogue_visitor, row_sum);
    }
};



template<
    typename Reduction_,
    typename Epilogue_>
struct SoftmaxBackwardUniversalwithEpilogueVisitorWarp {
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

        Reduction softmax_backward_reduction(
            params.reduction_params,
            shared_storage.reduction_storage,
            thread_idx,
            threadblock_offset
        );

        ElementAccumulator row_sum;
        typename Reduction::InputFragment o_softmax_buffer;
        typename Reduction::InputFragment grad_softmax_buffer;

        softmax_backward_reduction(o_softmax_buffer, grad_softmax_buffer, row_sum);

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
        epilogue(epilogue_visitor, o_softmax_buffer, grad_softmax_buffer, row_sum);
    }
};


/////////////////////////////////////////////////////////////////////////////////////////////////

} // namespace kernel
} // namespace softmax
} // namespace cutlass

/////////////////////////////////////////////////////////////////////////////////////////////////