#pragma once

#include "cutlass/cutlass.h"
#include "softmax/threadblock/row_tile_iterator.h"


/////////////////////////////////////////////////////////////////////////////////////////////////

namespace cutlass {
namespace softmax {
namespace threadblock {

/////////////////////////////////////////////////////////////////////////////////////////////////

template<
    typename ElementAccumulator_,
    typename ElementOutput_,
    int AlignmentOutput_,
    typename ThreadblockShape_,
    typename WarpCount_,
    typename ReductionResult_,
    typename InputCache_
>
class DefaultEpilogueSoftmaxBackward {

public:
    using Element = ElementOutput_;
    static const int kElementsPerAccess = AlignmentOutput_;
    using ElementAccumulator = ElementAccumulator_;
    using ThreadblockShape = ThreadblockShape_;
    using WarpCount = WarpCount_;

    using ThreadMap = cutlass::softmax::threadblock::RowThreadMap<
        ThreadblockShape_, WarpCount_, Element, kElementsPerAccess>;
    
    using InputTileIterator = cutlass::softmax::threadblock::RowTileIterator<ThreadMap, Element>;
    using OutputTileIterator = cutlass::softmax::threadblock::RowTileIterator<ThreadMap, Element>;

    using Fragment = typename OutputTileIterator::Fragment;

    using AccumulatorFragment = Array<ElementAccumulator, kElementsPerAccess>;
    using ReductionResult = ReductionResult_;
    using InputCache = InputCache_;

    using Mult = cutlass::multiplies<AccumulatorFragment>;
    using Sub = cutlass::minus<AccumulatorFragment>;

};




template<
    typename ElementAccumulator_,
    typename ElementOutput_,
    int AlignmentOutput_,
    typename ThreadblockShape_,
    typename WarpCount_,
    typename ReductionResult_,
    typename InputCache_
>
class DefaultEpilogueLayerNormBackward {

public:
    using Element = ElementOutput_;
    static const int kElementsPerAccess = AlignmentOutput_;
    using ElementAccumulator = ElementAccumulator_;
    using ThreadblockShape = ThreadblockShape_;
    using WarpCount = WarpCount_;

    using ThreadMap = cutlass::softmax::threadblock::RowThreadMap<
        ThreadblockShape_, WarpCount_, Element, kElementsPerAccess>;
    
    using InputTileIterator = cutlass::softmax::threadblock::RowTileIterator<ThreadMap, Element>;
    using OutputTileIterator = cutlass::softmax::threadblock::RowTileIterator<ThreadMap, Element>;

    using Fragment = typename OutputTileIterator::Fragment;

    using AccumulatorFragment = Array<ElementAccumulator, kElementsPerAccess>;
    using ReductionResult = ReductionResult_;
    using InputCache = InputCache_;

    using Mult = cutlass::multiplies<AccumulatorFragment>;
    using Sub = cutlass::minus<AccumulatorFragment>;

};



/////////////////////////////////////////////////////////////////////////////////////////////////

} // namespace threadblock
} // namespace softmax
} // namespace cutlass

/////////////////////////////////////////////////////////////////////////////////////////////////