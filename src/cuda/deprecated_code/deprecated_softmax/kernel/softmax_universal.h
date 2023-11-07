#pragma once

#include "cutlass/cutlass.h"
#include "softmax/threadblock/softmax_reduction.h"
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
struct SoftmaxUniversal {
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

    using BlockReduction = cutlass::softmax::threadblock::SoftmaxBlockReduction<
        ThreadblockShape, 
        WarpCount, 
        ElementInput, 
        kElementsPerAccess, 
        ElementAccumulator
    >;

    using WarpReduction = cutlass::softmax::threadblock::SoftmaxWarpReduction<
        ThreadblockShape, 
        WarpCount, 
        ElementInput, 
        kElementsPerAccess, 
        ElementAccumulator
    >;

    using Reduction = typename platform::conditional<WarpCount::kColumn == 1,
                                                     WarpReduction,
                                                     BlockReduction>::type;

    using Epilogue = cutlass::softmax::threadblock::DefaultEpilogueSoftmax<
        ElementAccumulator,
        ElementOutput,
        kElementsPerAccess,
        ThreadblockShape,
        WarpCount,
        typename Reduction::ReductionResult,
        typename Reduction::InputCache
    >;
};

/////////////////////////////////////////////////////////////////////////////////////////////////

} // namespace kernel
} // namespace softmax
} // namespace cutlass

/////////////////////////////////////////////////////////////////////////////////////////////////