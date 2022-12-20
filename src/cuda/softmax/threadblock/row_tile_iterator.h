#pragma once
#include "cutlass/cutlass.h"
#include "stdio.h"

/////////////////////////////////////////////////////////////////////////////////////////////////

namespace cutlass {
namespace softmax {
namespace threadblock {

/////////////////////////////////////////////////////////////////////////////////////////////////

/// Defines the thread map of softmax loader
template <
    typename ThreadblockShape_,
    typename WarpCount_,
    typename Element_,
    int ElementsPerAccess
>
class RowThreadMap {
public:
    using ThreadblockShape = ThreadblockShape_;
    using Element = Element_;
    static const int kElementsPerAccess = ElementsPerAccess;
    static const int kWarpSize = 32;

    using TensorCoord = MatrixCoord;

    using WarpCount = WarpCount_;

    using Shape = cutlass::MatrixShape<
        WarpCount::kRow,
        WarpCount::kColumn * kWarpSize * kElementsPerAccess
    >;

    using ShapeVec = cutlass::MatrixShape<
        WarpCount::kRow,
        WarpCount::kColumn * kWarpSize
    >;

    using Iterations = cutlass::MatrixShape<
        ThreadblockShape::kRow / Shape::kRow,
        (ThreadblockShape::kColumn + Shape::kColumn - 1) / Shape::kColumn
    >;

    using Delta = cutlass::MatrixShape<
        1,
        Shape::kColumn
    >;

    static_assert(WarpCount::kRow == 1, "Currently all the warps should be in the same row");
    static_assert(ThreadblockShape::kRow == 1, "Currently on threadblock only handles a single row");
        

    CUTLASS_HOST_DEVICE
    static TensorCoord initial_offset(int thread_id) {
        return TensorCoord(
            thread_id / ShapeVec::kColumn,
            (thread_id % ShapeVec::kColumn) * kElementsPerAccess
        );
    };
};


// Iterator used to load and store data from global memory in softmax
template <
    typename ThreadMap_,
    typename Element_
>
class RowTileIterator{
public:
    using ThreadMap = ThreadMap_;
    using Shape = typename ThreadMap::Shape;
    using ShapeVec = typename ThreadMap::ShapeVec;
    using Element = Element_;
    using TensorCoord = MatrixCoord;

    static int const kElementsPerAccess = ThreadMap::kElementsPerAccess;
    static int const kAlignBytes = kElementsPerAccess * sizeof(Element);

    /// Fragment object
    using Fragment = Array<Element, kElementsPerAccess>;

    using AccessType = AlignedArray<Element, kElementsPerAccess>;

    using Iterations = typename ThreadMap::Iterations;

    //
    // Structure
    //
    struct Params {
        int ldt;

        CUTLASS_HOST_DEVICE
        Params(int ldt_): ldt(ldt_) { }
    };

private:
    //
    // Data members
    //
    AccessType* pointer_;

    int extent_row_;
    int extent_column_;
    int thread_start_row_;
    int thread_start_column_;

    int column_;
    
public:
    //
    // Methods
    //

    /// Constructor
    CUTLASS_DEVICE
    RowTileIterator(
        Element *pointer,
        MatrixCoord extent,
        int thread_idx,
        TensorCoord threadblock_offset = TensorCoord()
    ) {
        TensorCoord thread_offset = ThreadMap::initial_offset(thread_idx) + threadblock_offset;

        extent_row_ = extent.row();
        extent_column_ = extent.column();

        thread_start_row_ = thread_offset.row();
        thread_start_column_ = thread_offset.column();

        column_ = thread_start_column_;

        // get initial pointer
        pointer_ = reinterpret_cast<AccessType*>(pointer + extent_column_ * thread_start_row_ + thread_start_column_);
    }

    CUTLASS_DEVICE
    RowTileIterator(
        Params const & params, 
        Element *pointer,
        MatrixCoord extent,
        int thread_idx,
        TensorCoord threadblock_offset = TensorCoord()
    ): RowTileIterator(pointer, extent, thread_idx, threadblock_offset) { }

    /// Loads a fragment from memory
    CUTLASS_DEVICE
    void load(Fragment &frag) {
        AccessType *frag_ptr = reinterpret_cast<AccessType *>(&frag);

        if (column_ < extent_column_){
            *frag_ptr = *pointer_;
        } else {
            frag.fill(Element(-1e+6));
        }

        pointer_ += ShapeVec::kColumn;
        column_ += Shape::kColumn;
    }

    CUTLASS_DEVICE
    Fragment load() {
        Fragment frag;
        load(frag);
        return frag;
    }

    /// Store a fragment to memory
    CUTLASS_DEVICE
    void store(Fragment const &frag) {
        AccessType const *frag_ptr = reinterpret_cast<AccessType const *>(&frag);
        if (column_ < extent_column_) {
            *pointer_ = *frag_ptr;
        } 

        pointer_ += ShapeVec::kColumn;
        column_ += Shape::kColumn;
    }
};




/////////////////////////////////////////////////////////////////////////////////////////////////

} // namespace kernel
} // namespace softmax
} // namespace threadblock

/////////////////////////////////////////////////////////////////////////////////////////////////