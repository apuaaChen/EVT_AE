#pragma once

#include "cutlass/cutlass.h"


/////////////////////////////////////////////////////////////////////////////////////////////////

namespace cutlass {
namespace softmax {
namespace threadblock {

/////////////////////////////////////////////////////////////////////////////////////////////////

template<
    typename ElementOutput_,
    int AlignmentOutput_,
    typename ThreadblockShape_,
    typename WarpCount_
>
class DefaultEpilogueSoftmax {

public:
    using Element = ElementOutput_;
    static const int kElementsPerAccess = AlignmentOutput_;

    using ThreadMap = cutlass::softmax::threadblock::RowThreadMap<
        ThreadblockShape_, WarpCount_, Element, kElementsPerAccess>;
    using OutputTileIterator = cutlass::softmax::threadblock::RowTileIterator<ThreadMap, Element>;

    using Fragment = typename OutputTileIterator::Fragment;

private:

    OutputTileIterator iterator_;

public:
    /// Constructor
    CUTLASS_DEVICE
    DefaultEpilogueSoftmax(
        Element *output_ptr,
        MatrixCoord extent,
        int thread_idx,
        MatrixCoord threadblock_offset = MatrixCoord()
    ):
        iterator_(
            typename OutputTileIterator::Param(extent.column()),
            output_ptr,
            extent,
            thread_idx,
            threadblock_offset
        )
    { }

    CUTLASS_DEVICE
    void store(Fragment const &frag) {
        iterator_.store(frag);
    }

};



/////////////////////////////////////////////////////////////////////////////////////////////////

} // namespace threadblock
} // namespace softmax
} // namespace cutlass

/////////////////////////////////////////////////////////////////////////////////////////////////