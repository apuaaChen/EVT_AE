#pragma once

#include "cutlass/cutlass.h"
#include "softmax/threadblock/layernorm_reduction.h"
#include "softmax/epilogue/epilogue.h"
#include "stdio.h"


/////////////////////////////////////////////////////////////////////////////////////////////////

namespace cutlass {
namespace softmax {
namespace kernel {

/////////////////////////////////////////////////////////////////////////////////////////////////

template<
    typename ThreadblockShape_,
    typename WarpCount_,
    typename ElementInput_,
    int AlignmentInput_,
    typename ElementOutput_,
    int AlignmentOutput_,
    typename ElementAccumulator_>
struct LayerNormUniversal {
public:
    static_assert(AlignmentInput_ == AlignmentOutput_, "Input and output tensor should have the same alignemnt");

    using ThreadblockShape = ThreadblockShape_;
    using WarpCount = WarpCount_;
    using ElementInput = ElementInput_;
    using ElementOutput = ElementOutput_;
    using ElementAccumulator = ElementAccumulator_;
    static const int kElementsPerAccess = AlignmentInput_;

    //
    // Iterators
    //

    using Reduction = cutlass::softmax::threadblock::LayerNormWarpReduction<
        ThreadblockShape,
        WarpCount,
        ElementInput,
        kElementsPerAccess,
        ElementAccumulator
    >;

    static_assert(WarpCount::kColumn == 1, "only one warp can be assigned to each row");

    using Epilogue = cutlass::softmax::threadblock::DefaultEpilogueLayerNorm<
        ElementAccumulator,
        ElementOutput,
        kElementsPerAccess,
        ThreadblockShape,
        WarpCount
    >;

    //
    // Structures
    //

    struct Arguments {
        //
        // Data members
        //
        typename Reduction::Arguments reduction_args;
        typename Epilogue::Arguments epilogue_args;
    };

    struct Params {
        //
        // Data members
        //
        typename Reduction::Params reduction_params;
        typename Epilogue::Params epilogue_params;

        //
        // Memebrs
        //
        CUTLASS_HOST_DEVICE
        Params(Arguments const &args):
            reduction_params(args.reduction_args),
            epilogue_params(args.epilogue_args)
        { }

    };

    union SharedStorage {
        typename Reduction::SharedStorage reduction_storage;
    };

public:
    
    /// Execute one layernorm
    CUTLASS_DEVICE
    void operator()(Params const &params, SharedStorage &shared_storage) {
        int thread_idx = threadIdx.x;
        cutlass::MatrixCoord threadblock_offset{
            int(blockIdx.x), int(blockIdx.y)
        };

        Reduction layernorm_reduction(params.reduction_params, shared_storage.reduction_storage, thread_idx, threadblock_offset);

        ElementAccumulator row_m1;
        ElementAccumulator row_m2;

        layernorm_reduction(row_m1, row_m2);

        /// Epilogue
        Epilogue epilogue(params.epilogue_params, thread_idx, threadblock_offset);

        epilogue(row_m1, row_m2);
    }
};

/////////////////////////////////////////////////////////////////////////////////////////////////

} // namespace kernel
} // namespace softmax
} // namespace cutlass

/////////////////////////////////////////////////////////////////////////////////////////////////