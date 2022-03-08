////////////////////////////////////////////////////////////////////////////////

namespace cutlass {
namespace epilogue {
namespace warp {

////////////////////////////////////////////////////////////////////////////////

/// 
template <
  typename WarpShape,         ///< shape of warp-level GEMM (concept: MatrixShape)
  typename OperatorShape,     ///< matrix multiply operation shape (concept: gemm::GemmShape)
  typename OperatorElementC,  ///< matrix multiply operation data type (concept: data type)
  typename OperatorFragmentC, ///< matrix multiply operation fragment (concept: Array)
  typename Layout             ///< target shared memory layout
>
class FragmentIteratorTensorOpV2;

////////////////////////////////////////////////////////////////////////////////

/// Partial specialization for row-major shared memory
template <
  typename WarpShape_,         ///< shape of the warp-level GEMM tile
  typename OperatorShape_,     ///< matrix multiply operation shape (concept: gemm::GemmShape)
  typename OperatorElementC_,  ///< matrix multiply operation data type (concept: data type)
  typename OperatorFragmentC_  ///< matrix multiply operation fragment (concept: Array)
>
class FragmentIteratorTensorOpV2<WarpShape_, OperatorShape_, OperatorElementC_, OperatorFragmentC_, layout::RowMajor> {
public:

  using WarpShape = WarpShape_;
  using OperatorShape = OperatorShape_;
  using OperatorElementC = OperatorElementC_;
  using OperatorFragmentC = OperatorFragmentC_;
  using Layout = layout::RowMajor;

  using Policy = TensorOpPolicy<WarpShape, OperatorShape, Layout>;

  /// This is the fragment size produced by one access of the iterator.
  using Fragment = Array<
    OperatorElementC, 
    Policy::OperatorCount::kColumn * Policy::kElementsPerAccess>;

  /// This is the complete warp-level accumulator tile.
  using AccumulatorTile = Array<
    OperatorElementC, 
    OperatorFragmentC::kElements * Policy::OperatorCount::kRow * Policy::OperatorCount::kColumn>;

  using OutputAccumulatorTile = AccumulatorTile;

  /// Number of times this iterator can be incremented
  static int const kIterations = Policy::kIterations;
  using TileIterations = typename Policy::TileIterations;
  static int const kIterationsPerTile = kIterations / TileIterations::kCount;

private:

  /// Internal access type
  using AccessType = Array<OperatorElementC, Policy::kElementsPerAccess>;

private:

  //
  // Data members
  //

  /// Accumulator tile
  AccessType const *accumulators_;

  /// Internal index
  int index_;

public:

  /// Constructs an iterator
  CUTLASS_HOST_DEVICE
  FragmentIteratorTensorOpV2(AccumulatorTile const &accum): 
    accumulators_(reinterpret_cast<AccessType const *>(&accum)), 
    index_(0) {
  }

  /// Increments
  CUTLASS_HOST_DEVICE
  FragmentIteratorTensorOpV2 &operator++() {
    ++index_;
    return *this;
  }

  /// Decrements
  CUTLASS_HOST_DEVICE
  FragmentIteratorTensorOpV2 &operator--() {
    --index_;
    return *this;
  }

  /// Loads a fragment from the referenced part of the accumulator tile
  CUTLASS_HOST_DEVICE
  void load(Fragment &frag, int index_offset = 0) const {

    int index = index_ + index_offset;

    AccessType *frag_ptr = reinterpret_cast<AccessType *>(&frag);

    CUTLASS_PRAGMA_UNROLL
    for (int n = 0; n < Policy::OperatorCount::kColumn; ++n) {

      int accumulator_access_offset = 
        index + n * Policy::kAccumulatorColumnStride / Policy::kElementsPerAccess;

      frag_ptr[n] = accumulators_[accumulator_access_offset];
    }
  }
};

} // namespace warp
} // namespace epilogue
} // namespace cutlass

////////////////////////////////////////////////////////////////////////////////