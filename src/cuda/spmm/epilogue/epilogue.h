#pragma once

#include "cutlass/cutlass.h"
#include "spmm/threadblock/row_tile_iterator.h"

/////////////////////////////////////////////////////////////////////////////////////////////////

namespace cutlass {
namespace spmm {
namespace threadblock {

/////////////////////////////////////////////////////////////////////////////////////////////////

template<
    typename ElementAccumulator_,
    typename ElementOutput_,
    int ALignmentOutput_,
    typename Accumulator_,
    typename ThreadblockShape_,
    typename ThreadMap_
>
class DefaultEpilogueSpmm {
public:
    using ElementAccumulator = ElementAccumulator_;
    using Element = ElementOutput_;
    static const int kElementsPerAccess = ALignmentOutput_;
    using FragmentAcc = Accumulator_;
    using ThreadMap = ThreadMap_;

    using ThreadMapOutput = cutlass::spmm::threadblock::OutputThreadMap<
        ThreadblockShape_, Element, kElementsPerAccess>;
    
    using OutputTileIterator = cutlass::spmm::threadblock::OutputTileIterator<
        ThreadMapOutput, Element>;

};


/////////////////////////////////////////////////////////////////////////////////////////////////

} // namespace threadblock
} // namespace spmm
} // namespace cutlass

/////////////////////////////////////////////////////////////////////////////////////////////////