#pragma once
#include "cutlass/cutlass.h"
#include "stdio.h"

/////////////////////////////////////////////////////////////////////////////////////////////////

namespace cutlass {
namespace spmm {
namespace threadblock {

/////////////////////////////////////////////////////////////////////////////////////////////////

/// Defines the thread map of spmm store
template <
    typename ThreadblockShape_,
    typename Element_,
    int ElementsPerAccess
>
class OutputThreadMap {
public:
    // threadblock tiling size
    // concept: MatrixCoord
    using ThreadblockShape = ThreadblockShape_;
    // output data type
    using Element = Element_;
    // alignment
    static const int kElementsPerAccess = ElementsPerAccess;

    using TensorCoord = MatrixCoord;

    // the shape processed by the threadblock per iteration
    using Shape = cutlass::MatrixShape<
        ThreadblockShape::kRow,
        ThreadblockShape::kColumn
    >;

    // number of threads in rows and columns
    using ShapeVec = cutlass::MatrixShape<
        ThreadblockShape::kRow,
        ThreadblockShape::kColumn / kElementsPerAccess
    >;

    static const int kThreads = ShapeVec::kRow * ShapeVec::kColumn;

    // number of iterations
    using Iterations = cutlass::MatrixShape<
        ThreadblockShape::kRow / Shape::kRow,
        (ThreadblockShape::kColumn + Shape::kColumn - 1) / Shape::kColumn
    >;

    // stride
    using Delta = cutlass::MatrixShape<
        1,
        Shape::kColumn
    >;
      

    // compute initial row and column in the current tile
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
class OutputTileIterator{
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
    OutputTileIterator(
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
    OutputTileIterator(
        Params const & params, 
        Element *pointer,
        MatrixCoord extent,
        int thread_idx,
        TensorCoord threadblock_offset = TensorCoord()
    ): OutputTileIterator(pointer, extent, thread_idx, threadblock_offset) { }

    /// Loads a fragment from memory
    CUTLASS_DEVICE
    void load(Fragment &frag, Element fill_value=Element(-1e+6)) {
        AccessType *frag_ptr = reinterpret_cast<AccessType *>(&frag);

        if (column_ < extent_column_){
            *frag_ptr = *pointer_;
        } else {
            frag.fill(fill_value);
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