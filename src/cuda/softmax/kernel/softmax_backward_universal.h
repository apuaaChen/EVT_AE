#pragma once

#include "cutlass/cutlass.h"
#include "softmax/threadblock/softmax_backward_reduction.h"
#include "softmax/epilogue/epilogue_backward.h"
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
struct SoftmaxBackwardUniversal {
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

    using BlockReduction = cutlass::softmax::threadblock::SoftmaxBackwardBlockReduction<
        ThreadblockShape, 
        WarpCount, 
        ElementInput, 
        kElementsPerAccess, 
        ElementAccumulator
    >;

    using WarpReduction = cutlass::softmax::threadblock::SoftmaxBackwardWarpReduction<
        ThreadblockShape, 
        WarpCount, 
        ElementInput, 
        kElementsPerAccess, 
        ElementAccumulator
    >;

    using Reduction = typename platform::conditional<WarpCount::kColumn == 1,
                                                     WarpReduction,
                                                     BlockReduction>::type;

    using Epilogue = cutlass::softmax::threadblock::DefaultEpilogueSoftmaxBackward<
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