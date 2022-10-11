#pragma once

#include "cutlass/cutlass.h"
#include "softmax/kernel/softmax_universal.h"

/////////////////////////////////////////////////////////////////////////////////////////////////

namespace cutlass {
namespace softmax {
namespace kernel {

/////////////////////////////////////////////////////////////////////////////////////////////////

template<
    typename ThreadblockShape,
    typename WarpCount,
    typename ElementInput,
    int AlignmentInput,
    typename ElementOutput,
    int AlignmentOutput,
    typename ElementAccumulator
>
struct DefaultSoftmaxUniversal {
    using SoftmaxKernel = kernel::SoftmaxUniversal<
        ThreadblockShape, WarpCount,
        ElementInput, AlignmentInput, ElementOutput, 
        AlignmentOutput, ElementAccumulator
    >;
};


/////////////////////////////////////////////////////////////////////////////////////////////////

}  // namespace kernel
}  // namespace softmax
}  // namespace cutlass

/////////////////////////////////////////////////////////////////////////////////////////////////