#pragma once

#include "cutlass/cutlass.h"

#include "softmax/threadblock/layernorm_reduction.h"

/////////////////////////////////////////////////////////////////////////////////////////////////

namespace cutlass {
namespace softmax {
namespace kernel {

template<
    typename Reduction_,
    typename Epilogue_>
struct LayerNormUniversalwithEpilogueVisitorWarp {
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

    /// Execute one softmax
    CUTLASS_DEVICE
    void operator()(Params const &params, SharedStorage &shared_storage) {

        int thread_idx = threadIdx.x;
        cutlass::MatrixCoord threadblock_offset{
            int(blockIdx.x), int(blockIdx.y)
        };

        Reduction layernorm_reduction(
            params.reduction_params, 
            shared_storage.reduction_storage, 
            thread_idx,
            threadblock_offset
        );

        ElementAccumulator row_m1;
        ElementAccumulator row_m2;
        typename Reduction::InputFragment input_buffer;


        layernorm_reduction(input_buffer, row_m1, row_m2);

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
        epilogue(epilogue_visitor, input_buffer, row_m1, row_m2);

    }
};

/////////////////////////////////////////////////////////////////////////////////////////////////

} // namespace kernel
} // namespace softmax
} // namespace cutlass

/////////////////////////////////////////////////////////////////////////////////////////////////