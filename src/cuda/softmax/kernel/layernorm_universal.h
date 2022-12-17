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
};

/////////////////////////////////////////////////////////////////////////////////////////////////

} // namespace kernel
} // namespace softmax
} // namespace cutlass

/////////////////////////////////////////////////////////////////////////////////////////////////