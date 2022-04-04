#include "helper.h"
namespace cutlass {
namespace transform {
namespace threadblock {

////////////////////////////////////////////////////////////////////////////////

template <typename Shape, typename Element, typename Layout, int AdvanceRank,
          typename ThreadMap,
          int Alignment =
              sizeof_bits<Element>::value* ThreadMap::kElementsPerAccess / 8>
class RegularTileAccessIteratorETrans;



////////////////////////////////////////////////////////////////////////////////

/// Tile iterator specialized for congruous arrangements for TensorOps
///
///
/// Satisfies: ForwardTileIteratorConcept |
///            ReadableContiguousTileIteratorConcept |
///            WriteableContiguousTileIteratorConcept
///
template <typename Shape_, typename Element_, int AdvanceRank,
          typename ThreadMap_, int Alignment>
class RegularTileAccessIteratorETrans<
    Shape_, Element_,
    layout::PitchLinear,
    AdvanceRank, ThreadMap_, Alignment> {
 public:
  static_assert(
      AdvanceRank == 0 || AdvanceRank == 1,
      "Specialization for pitch-linear iterator may along advance along the "
      "contiguous(rank=0) or strided(rank=1) dimension.");

  using Shape = Shape_;
  using Element = Element_;
  using Layout = layout::PitchLinear;
  static int const kAdvanceRank = AdvanceRank;
  static int const kAlignment = Alignment;

  using Index = typename Layout::Index;
  using LongIndex = typename Layout::LongIndex;
  using StrideIndex = typename Layout::Stride::Index;

  using TensorRef = TensorRef<Element, Layout>;
  using TensorCoord = typename Layout::TensorCoord;

  using ThreadMap = ThreadMap_;

  /// Element type per access
  using AccessType = Array<Element, ThreadMap::kElementsPerAccess>;

 private:
  //
  // Data members
  //

  /// Stride value
  StrideIndex stride_;

  /// Internal pointer to first access of tile
  AccessType *pointer_;

  /// Internal byte offset
  Index byte_offset_;

  /// Iteration in the contiguous dimension
  int iteration_contiguous_;

  /// Iteration in the strided dimension
  int iteration_strided_;

 public:
  /// Construct a TileIterator with zero threadblock offset
  CUTLASS_HOST_DEVICE
  RegularTileAccessIteratorETrans(TensorRef ref,  ///< Pointer to start of tensor
                            int thread_id   ///< ID of each participating thread
                            )
      : stride_(ref.stride(0) / ThreadMap::kElementsPerAccess),
        byte_offset_(0) {

    layout::PitchLinearCoord thread_offset_base = ThreadMap::initial_offset(thread_id);
    
    // initialize pointer
    pointer_ = reinterpret_cast<AccessType *>(ref.data() + ref.offset(thread_offset_base));

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
    byte_offset_ += pointer_offset * sizeof(Element);
  }

  /// Returns a pointer
  CUTLASS_DEVICE
  AccessType *get() const {

    AccessType *access_ptr = pointer_;

    int access_offset = iteration_strided_ * ThreadMap::Delta::kStrided * stride_ +
                        iteration_contiguous_ * ThreadMap::Delta::kContiguous /
                            ThreadMap::kElementsPerAccess;

    char *access_byte_ptr =
        reinterpret_cast<char *>(access_ptr + access_offset);

    return reinterpret_cast<AccessType *>(access_byte_ptr + byte_offset_);
  }

  /// Advances to the next tile in memory.
  CUTLASS_HOST_DEVICE
  RegularTileAccessIteratorETrans &operator++() {
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

    // Enter here only if (iteration_stride_ == ThreadMap::Iteration::kStrided)
    // which means we enter the next tile.
    iteration_strided_ = 0;

    return *this;
  }

  /// Advances to the next tile in memory.
  CUTLASS_HOST_DEVICE
  RegularTileAccessIteratorETrans operator++(int) {
    RegularTileAccessIteratorETrans prev(*this);
    this->operator++();

    return prev;
  }

  /// Adds a tile offset in the unit of tile.
  /// In GEMM/Conv implementation, this is used to move in the k dimension in the shared memory.
  /// Below layouts are the shared memory layouts.  Current SM50 SIMT kernels only use col major A and row major B.
  ///   For row major A operand, k dimension is contiguous dimension;
  ///   For col major A operand, k dimension is strided dimension;
  ///   For row major B operand, k dimension is strided dimension;
  ///   For col major B operand, k dimension is contiguous dimension.
  /// Below two classes map col/row major to the pitch linear coordinates used
  /// in this base class.
  CUTLASS_DEVICE
  void add_tile_offset(TensorCoord const &coord) {
    add_pointer_offset(coord.contiguous() * Shape::kContiguous +
                       coord.strided() * Shape::kStrided * stride_ *
                           ThreadMap::kElementsPerAccess);
  }
};


////////////////////////////////////////////////////////////////////////////////

/// Tile iterator specialized for column major layouts
///
///
/// Satisfies: ForwardTileIteratorConcept |
///            ReadableContiguousTileIteratorConcept |
///            WriteableContiguousTileIteratorConcept
///
template <typename Shape_, typename Element_, int AdvanceRank,
          typename ThreadMap_, int Alignment>
class RegularTileAccessIteratorETrans<
    Shape_, Element_,
    layout::ColumnMajor,
    AdvanceRank, ThreadMap_, Alignment> {
 public:
  static_assert(
      AdvanceRank == 0 || AdvanceRank == 1,
      "Specialization for pitch-linear iterator may along advance along the "
      "contiguous(rank=0) or strided(rank=1) dimension.");

  using Shape = Shape_;
  using Element = Element_;
  using Layout = layout::ColumnMajor;
  static int const kAdvanceRank = AdvanceRank;
  static int const kAlignment = Alignment;

  using Index = typename Layout::Index;
  using LongIndex = typename Layout::LongIndex;

  using TensorRef = TensorRef<Element, Layout>;
  using TensorCoord = typename Layout::TensorCoord;

  using ThreadMap = ThreadMap_;

  /// Underlying iterator type
  using UnderlyingIterator = RegularTileAccessIteratorETrans<
      layout::PitchLinearShape<Shape::kRow, Shape::kColumn>, Element,
      layout::PitchLinear,
      (kAdvanceRank == 0 ? 0 : 1), 
      ThreadMap_>;

  using AccessType = typename UnderlyingIterator::AccessType;

 private:

  /// Underlying iterator
  UnderlyingIterator iterator_;

 public:
  /// Construct a TileIterator with zero threadblock offset
  CUTLASS_HOST_DEVICE
  RegularTileAccessIteratorETrans(TensorRef ref,  ///< Pointer to start of tensor
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
    iterator_.add_tile_offset({coord.row(), coord.column()});
  }

  /// Advances to the next tile in memory.
  CUTLASS_HOST_DEVICE
  RegularTileAccessIteratorETrans &operator++() {
    ++iterator_;
    return *this;
  }

  /// Advances to the next tile in memory.
  CUTLASS_HOST_DEVICE
  RegularTileAccessIteratorETrans operator++(int) {
    RegularTileAccessIteratorETrans prev(*this);
    ++iterator_;

    return prev;
  }
};

////////////////////////////////////////////////////////////////////////////////

}  // namespace threadblock
}  // namespace transform
}  // namespace cutlass