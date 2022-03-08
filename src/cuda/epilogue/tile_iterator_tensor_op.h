#include "helper.h"
namespace cutlass {
namespace epilogue {
namespace warp {


/// Template for reading and writing tiles of accumulators to shared memory
template <
  typename WarpShape,     ///< shape of warp-level GEMM (concept: MatrixShape)
  typename OperatorShape, ///< matrix multiply operation shape (concept: gemm::GemmShape)
  typename Element,       ///< data type of element to be written
  typename Layout         ///< target shared memory layout
>
class TileIteratorTensorOpV2;

/// Template for reading and writing tiles of accumulators to shared memory
template <
  typename WarpShape_,     ///< shape of warp-level GEMM (concept: GemmShape)
  typename OperatorShape_, ///< matrix multiply operation shape (concept: gemm::GemmShape)
  typename Element_        ///< data type of element to be written
>
class TileIteratorTensorOpV2<WarpShape_, OperatorShape_, Element_, layout::RowMajor> {
public:

  using WarpShape = WarpShape_;
  using OperatorShape = OperatorShape_;
  using Element = Element_;
  using Layout = layout::RowMajor;

  using TensorLayout = Layout;
  using TensorRef = TensorRef<Element, Layout>;         ///< Tensor Reference object
  using TensorCoord = MatrixCoord;                      ///< Logical coordinate in referenced tensor
  using Index = typename TensorRef::Index;
  using LongIndex = typename TensorRef::LongIndex;

  using Policy = TensorOpPolicy<WarpShape, OperatorShape, Layout>;

  /// Shape of the tile in memory
  using Shape = MatrixShape<
    Policy::kRowsPerIteration,
    WarpShape::kN
  >;

  /// This is the fragment size produced by one access of the iterator.
  using Fragment = Array<
    Element, 
    Policy::OperatorCount::kColumn * Policy::kElementsPerAccess>;

  /// This is the complete warp-level accumulator tile.
  //using AccumulatorTile = typename Operator::FragmentC;

  /// Number of times this iterator can be incremented
  static int const kIterations = Policy::kIterations;

  /// Number of times this iterator can be incremented
  using TileIterations = typename Policy::TileIterations;

  // Internal constants
  struct Detail {
    static int const kLanesInQuad = 4;
  };

  /// Padding quantity
  using Padding = MatrixShape<
    0,
    Detail::kLanesInQuad * Policy::kElementsPerAccess>;

private:

  /// Storage type for accessing memory
  using AccessType = AlignedArray<Element, Policy::kElementsPerAccess>;

  //
  // Data members
  //

  /// Internal pointer to memory
  AccessType *pointer_;

  /// Internal layout object
  Layout layout_;

  /// Thread offset
  MatrixCoord thread_offset_;

public:

  /// Default constructor
  CUTLASS_HOST_DEVICE
  TileIteratorTensorOpV2(): pointer_(nullptr) { }

  /// Constructor from TensorRef
  CUTLASS_HOST_DEVICE
  TileIteratorTensorOpV2(
    TensorRef const &ref,
    unsigned lane_id
  ):
    pointer_(reinterpret_cast<AccessType *>(ref.data())),
    layout_(ref.stride()[0] / Policy::kElementsPerAccess) {

    int quad_id = (lane_id / Detail::kLanesInQuad); 
    int lane_in_quad = (lane_id % Detail::kLanesInQuad);

    thread_offset_ = {
      quad_id, lane_in_quad * Policy::kElementsPerAccess
    };

    pointer_ += layout_({thread_offset_.row(), thread_offset_.column() / Policy::kElementsPerAccess});
  }

  /// Adds a pointer offset
  CUTLASS_HOST_DEVICE
  TileIteratorTensorOpV2 & add_pointer_offset(Index pointer_offset) {
    pointer_ += pointer_offset / Policy::kElementsPerAccess;
    return *this;
  }

  ///< advances in units of whole tiles along the logical coordinate space of the tensor
  CUTLASS_HOST_DEVICE
  TileIteratorTensorOpV2 & add_tile_offset(TensorCoord const &tile_offset) {

    MatrixCoord coord_offset(
      tile_offset.row() * Shape::kRow, 
      tile_offset.column() * Shape::kColumn
    );

    thread_offset_ += coord_offset;

    pointer_ += layout_({
      coord_offset.row(),
      coord_offset.column() / Policy::kElementsPerAccess
    });

    return *this;
  }

  ///< advances in units of whole tiles along the logical coordinate space of the tensor
  CUTLASS_HOST_DEVICE
  TileIteratorTensorOpV2 & operator+=(TensorCoord const &tile_offset) {
    add_tile_offset(tile_offset);
    return *this;
  }

  /// Store
  CUTLASS_HOST_DEVICE
  void store_with_pointer_offset(Fragment const &frag, Index pointer_offset) {

    AccessType const *frag_ptr = reinterpret_cast<AccessType const *>(&frag);

    CUTLASS_PRAGMA_UNROLL
    for (int n = 0; n < Policy::OperatorCount::kColumn; ++n) {
      pointer_[n * Detail::kLanesInQuad + pointer_offset / Policy::kElementsPerAccess] = frag_ptr[n];
    }
  }

  /// Store
  CUTLASS_HOST_DEVICE
  void store(Fragment const &frag) {
    store_with_pointer_offset(frag, 0);
  }

  /// Load
  CUTLASS_HOST_DEVICE
  void load_with_pointer_offset(Fragment &frag, Index pointer_offset) const {

    AccessType *frag_ptr = reinterpret_cast<AccessType *>(&frag);

    CUTLASS_PRAGMA_UNROLL
    for (int n = 0; n < Policy::OperatorCount::kColumn; ++n) {
      frag_ptr[n] = pointer_[n * Detail::kLanesInQuad + pointer_offset / Policy::kElementsPerAccess];
    }
  }

  /// Load
  CUTLASS_HOST_DEVICE
  void load(Fragment &frag) const {
    load_with_pointer_offset(frag, 0);
  }

  CUTLASS_HOST_DEVICE
  TileIteratorTensorOpV2 & operator++() {
    return add_tile_offset({1, 0});
  }
};

/////////////////////////////////////////////////////////////////////////////////////////////////

} // namespace warp
} // namespace epilogue
} // namespace cutlass

/////////////////////////////////////////////////////////////////////////////////////////////////