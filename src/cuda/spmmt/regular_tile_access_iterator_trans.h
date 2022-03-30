#include "helper.h"
////////////////////////////////////////////////////////////////////////////////

namespace cutlass {
namespace transform {
namespace threadblock {

////////////////////////////////////////////////////////////////////////////////

template <typename Shape, typename Element, typename Layout, int AdvanceRank,
          typename ThreadMap,
          int Alignment =
              sizeof_bits<Element>::value* ThreadMap::kElementsPerAccess / 8>
class RegularTileAccessIteratorTrans;



////////////////////////////////////////////////////////////////////////////////

/// Tile iterator specialized for crosswise arrangements for TensorOps
///
///
/// Satisfies: ForwardTileIteratorConcept |
///            ReadableContiguousTileIteratorConcept |
///            WriteableContiguousTileIteratorConcept
///
template <typename Shape_, typename Element_, int AdvanceRank,
          typename ThreadMap_, int Alignment, int Crosswise>
class RegularTileAccessIteratorTrans<Shape_, Element_,
                                layout::TensorOpMultiplicandCrosswise<
                                    sizeof_bits<Element_>::value, Crosswise>,
                                AdvanceRank, ThreadMap_, Alignment> {
 public:
  static_assert(
      AdvanceRank == 0 || AdvanceRank == 1,
      "Specialization for pitch-linear iterator may along advance along the "
      "contiguous(rank=0) or strided(rank=1) dimension.");

  using Shape = Shape_;
  using Element = Element_;
  using Layout =
      layout::TensorOpMultiplicandCrosswise<sizeof_bits<Element_>::value,
                                            Crosswise>;
  static int const kAdvanceRank = AdvanceRank;
  static int const kAlignment = Alignment;
  static int const kCrosswise = Crosswise;

  using Index = typename Layout::Index;
  using LongIndex = typename Layout::LongIndex;
  using StrideIndex = typename Layout::Stride::Index;

  using TensorRef = TensorRef<Element, Layout>;
  using TensorCoord = typename Layout::TensorCoord;

  using ThreadMap = ThreadMap_;

  static_assert(!(ThreadMap::Delta::kContiguous % kCrosswise),
                "kCrosswise is the smallest unit in the contiguous dimension "
                "for shared memory swizzling.");

  /// Internal details made public to facilitate introspection
  struct Detail {
    /// This iterator is specialized for an access size that is 128 bits in
    /// length.
    static int const kAccessSizeInBits = 128;

    static_assert(sizeof_bits<Element_>::value *
                          ThreadMap::kElementsPerAccess ==
                      kAccessSizeInBits,
                  "This iterator requires a policy whose access size is 128bs");

    /// Number of pointers
    ///
    /// Note:TN kblock32 layouts only needs 1 pointer, but strangely
    /// reducing pointer count hurts perfomrnace
    static int const kPointerCount =
        (ThreadMap::Iterations::kStrided > 1 ? 2 : 1);
  };

  /// Element type per access
  using AccessType = Array<Element, Layout::kElementsPerAccess>;

 private:
  //
  // Data members
  //

  /// Total number of sections.  The memory is divided into stages.  One stage
  /// can store one tile.  Stage is divided into sections.  Interleaved layout
  /// can have multiple sections in a stage.  The rest layout only has one section
  /// in a stage.
  int sections_;

  /// Sections that a stage has
  int sections_per_stage_;

  /// Stride value
  StrideIndex stride_;

  /// Internal pointer to first access of tile
  AccessType *pointer_[Detail::kPointerCount];

  /// Internal byte offset
  Index byte_offset_;

  /// Iteration in the contiguous dimension
  int iteration_contiguous_;

  /// Iteration in the strided dimension
  int iteration_strided_;

 public:
  /// Construct a TileIterator with zero threadblock offset
  CUTLASS_HOST_DEVICE
  RegularTileAccessIteratorTrans(TensorRef ref,  ///< Pointer to start of tensor
                            int thread_id   ///< ID of each participating thread
                            )
      : sections_(ref.stride(0) / kCrosswise),
        sections_per_stage_(Shape::kContiguous / kCrosswise),
        // stride_ = kCrosswise x sections_ x kFactor
        stride_(ref.stride(0) * Layout::kFactor / Layout::kElementsPerAccess),
        byte_offset_(0) {
    layout::PitchLinearCoord thread_offset_base =
        ThreadMap::initial_offset(thread_id);
    print_val(0, 0, 0, Detail::kPointerCount);
    CUTLASS_PRAGMA_UNROLL
    for (int i = 0; i < Detail::kPointerCount; ++i) {
      // This is the offset of a thread within a threadblock tile for a specific
      // pointer (units of elements)
      layout::PitchLinearCoord thread_offset_in_threadblock_tile =
          thread_offset_base +
          layout::PitchLinearCoord{
              0, ThreadMap::Detail::WarpThreadArrangement::kStrided * i};
      // initialize pointer
      pointer_[i] = reinterpret_cast<AccessType *>(ref.data()) +
                    ref.offset(thread_offset_in_threadblock_tile) /
                        Layout::kElementsPerAccess;
    }

    set_iteration_index(0);
  }

  /// Overrides the internal iteration index
  CUTLASS_HOST_DEVICE
  void set_iteration_index(int index) {
    iteration_contiguous_ = index % ThreadMap::Iterations::kContiguous;
    iteration_strided_ = index / ThreadMap::Iterations::kContiguous;
  }

  /// Adds a pointer offset in units of Element
  CUTLASS_HOST_DEVICE
  void add_pointer_offset(LongIndex pointer_offset) {
    byte_offset_ += pointer_offset * sizeof_bits<Element>::value / 8;
  }

  /// Returns a pointer
  CUTLASS_HOST_DEVICE
  AccessType *get() const {
    AccessType *access_ptr = pointer_[iteration_strided_ & 1];
    int stride_idx = (iteration_strided_ & ~1);

    int access_offset =
        stride_idx * ThreadMap::Delta::kStrided * stride_ / Layout::kFactor +
        // kCrosswise elements in the contiguous dimension would span to a
        // shared memory cache line.
        iteration_contiguous_ * (ThreadMap::Delta::kContiguous / kCrosswise) *
            Layout::TileShape::kContiguous;
    char *access_byte_ptr =
        reinterpret_cast<char *>(access_ptr + access_offset);
    return reinterpret_cast<AccessType *>(access_byte_ptr + byte_offset_);
  }

  /// Advances to the next tile in memory.
  CUTLASS_HOST_DEVICE
  RegularTileAccessIteratorTrans &operator++() {
    ++iteration_contiguous_;

    if (iteration_contiguous_ < ThreadMap::Iterations::kContiguous)
      return *this;

    // Enter here only if (iteration_contiguous_ ==
    // ThreadMap::Iteration::kContiguous)
    iteration_contiguous_ = 0;
    ++iteration_strided_;

    if (iteration_strided_ < ThreadMap::Iterations::kStrided) {
      return *this;
    }

    // Enter here only if (iteration_strided_ == ThreadMap::Iteration::kStrided)
    // which means we enter the next section.
    iteration_strided_ = 0;

    return *this;
  }

  /// Advances to the next tile in memory.
  CUTLASS_HOST_DEVICE
  RegularTileAccessIteratorTrans operator++(int) {
    RegularTileAccessIteratorTrans prev(*this);
    this->operator++();

    return prev;
  }

  /// Adds a tile offset
  CUTLASS_DEVICE
  void add_tile_offset(TensorCoord const &coord) {
    add_pointer_offset(coord.contiguous() * sections_per_stage_ * stride_ *
                           ThreadMap::kElementsPerAccess / sections_ +
                       coord.strided() * Shape::kStrided * stride_ *
                           Layout::kElementsPerAccess);
  }
};


/// Tile Iterator specialized for row-major crosswise TensorOp formats.
///
///
/// Satisfies: ForwardTileIteratorConcept |
///            ReadableContiguousTileIteratorConcept |
///            WriteableContiguousTileIteratorConcept
///
template <typename Shape_, typename Element_, int AdvanceRank,
          typename ThreadMap_, int Alignment, int Crosswise>
class RegularTileAccessIteratorTrans<Shape_, Element_,
                                layout::RowMajorTensorOpMultiplicandCrosswise<
                                    sizeof_bits<Element_>::value, Crosswise>,
                                AdvanceRank, ThreadMap_, Alignment> {
 public:
  static_assert(
      AdvanceRank == 0 || AdvanceRank == 1,
      "Specialization for row-major iterator may along advance along the "
      "columns(rank=0) or rows(rank=1) dimension.");

  using Shape = Shape_;
  using Element = Element_;
  using Layout = layout::RowMajorTensorOpMultiplicandCrosswise<
      sizeof_bits<Element_>::value, Crosswise>;
  static int const kAdvanceRank = AdvanceRank;
  static int const kAlignment = Alignment;

  using Index = typename Layout::Index;
  using LongIndex = typename Layout::LongIndex;

  using TensorRef = TensorRef<Element, Layout>;
  using TensorCoord = typename Layout::TensorCoord;

  using ThreadMap = ThreadMap_;

  /// Underlying iterator type
  using UnderlyingIterator = RegularTileAccessIteratorTrans<
      layout::PitchLinearShape<Shape::kColumn, Shape::kRow>, Element,
      layout::TensorOpMultiplicandCrosswise<sizeof_bits<Element_>::value,
                                            Crosswise>,
      (kAdvanceRank == 0 ? 1 : 0), ThreadMap_>;

  using AccessType = typename UnderlyingIterator::AccessType;

 private:
  /// Underlying iterator
  UnderlyingIterator iterator_;

 public:
  /// Construct a TileIterator with zero threadblock offset
  CUTLASS_HOST_DEVICE
  RegularTileAccessIteratorTrans(TensorRef ref,  ///< Pointer to start of tensor
                            int thread_id   ///< ID of each participating thread
                            )
      : iterator_({ref.data(), ref.stride()}, thread_id) {}

  /// Overrides the internal iteration index
  CUTLASS_HOST_DEVICE
  void set_iteration_index(int index) { iterator_.set_iteration_index(index); }

  /// Adds a pointer offset in units of Element
  CUTLASS_HOST_DEVICE
  void add_pointer_offset(LongIndex pointer_offset) {
    iterator_.add_pointer_offset(pointer_offset);
  }

  /// Returns a pointer
  CUTLASS_HOST_DEVICE
  AccessType *get() const {
    return reinterpret_cast<AccessType *>(iterator_.get());
  }

  /// Adds a tile offset
  CUTLASS_DEVICE
  void add_tile_offset(TensorCoord const &coord) {
    iterator_.add_tile_offset({coord.column(), coord.row()});
  }

  /// Advances to the next tile in memory.
  CUTLASS_HOST_DEVICE
  RegularTileAccessIteratorTrans &operator++() {
    ++iterator_;
    return *this;
  }

  /// Advances to the next tile in memory.
  CUTLASS_HOST_DEVICE
  RegularTileAccessIteratorTrans operator++(int) {
    RegularTileAccessIteratorTrans prev(*this);
    ++iterator_;

    return prev;
  }
};


////////////////////////////////////////////////////////////////////////////////

}  // namespace threadblock
}  // namespace transform
}  // namespace cutlass