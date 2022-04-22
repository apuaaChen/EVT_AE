////////////////////////////////////////////////////////////////////////////////

namespace cutlass {
namespace transform {


////////////////////////////////////////////////////////////////////////////////

/// Policy defining a warp-raked arrangement in which a shape is partitioned into contiguous
/// elements.
///
/// This ThreadMap is used by tensor core kernels.
template <
  typename Shape_,
  int Threads,
  typename WarpThreadArrangement_,
  int ElementsPerAccess = 1
>
struct PitchLinearWarpRakedThreadMapV3 {

  /// Tensor coordinate
  using TensorCoord = layout::PitchLinearCoord;

  /// Tile shape
  using Shape = Shape_;

  /// Number of threads total
  static int const kThreads = Threads;

  /// Extract vector length from Layout
  static int const kElementsPerAccess = ElementsPerAccess;

  /// Shape of access by each thread
  using ThreadAccessShape = layout::PitchLinearShape<kElementsPerAccess, 1>;

  /// Internal details made public to facilitate introspection
  struct Detail {

    /// Fixed arrangement of threads within a warp (units of threads).
    using WarpThreadArrangement = WarpThreadArrangement_;

    /// Number of threads per warp
    static int const kWarpSize = WarpThreadArrangement::kCount;

    /// Number of participating warps
    static int const kWarpCount = kThreads / kWarpSize;

    static_assert(
      !(Shape::kContiguous % kElementsPerAccess),
      "Shape must be divisible by vector length.");

    /// Compute the 'shape' of the overall tile in units of vectors
    using ShapeInAccesses = layout::PitchLinearShape<
      Shape::kContiguous / kElementsPerAccess,
      Shape::kStrided
    >;

    static_assert(
      !(ShapeInAccesses::kContiguous % WarpThreadArrangement::kContiguous),
      "ShapeInAccesses must be divisible by WarpThreadArrangement.");

    static_assert(
      !(ShapeInAccesses::kStrided % WarpThreadArrangement::kStrided),
      "ShapeInAccesses must be divisible by WarpThreadArrangement.");

    // compute number of warp-level accesses total
    using WarpAccessIterations = layout::PitchLinearShape<
      ShapeInAccesses::kContiguous / WarpThreadArrangement::kContiguous,
      ShapeInAccesses::kStrided / WarpThreadArrangement::kStrided
    >;

    // Divide it into the number of warps, first partitioning the strided dimension then the
    // contiguous.
    static int const kWarpsStrided =
        (WarpAccessIterations::kStrided >= kWarpCount
             ? kWarpCount
             : WarpAccessIterations::kStrided);

    static int const kWarpsContiguous =
        (kWarpCount > WarpAccessIterations::kStrided
             ? kWarpCount / kWarpsStrided
             : 1);

    /// Arrangement of warps within a threadblock-scoped tile
    using WarpArrangement = layout::PitchLinearShape<
      kWarpsContiguous, kWarpsStrided
    >;
  };

  ///< Iterations along each dimension (concept: PitchLinearShape)
  using Iterations = layout::PitchLinearShape<
    Detail::WarpAccessIterations::kContiguous / Detail::kWarpsContiguous,
    Detail::WarpAccessIterations::kStrided / Detail::kWarpsStrided
  >;

  static_assert(Iterations::kCount,
    "Number of iterations must be non-zero");

  ///< Delta betweeen accesses (units of elements, concept: PitchLinearShape)
  using Delta = layout::PitchLinearShape<
    Detail::WarpThreadArrangement::kContiguous * kElementsPerAccess,
    Detail::WarpThreadArrangement::kStrided * Detail::kWarpsStrided
  >;

  /// Maps thread ID to a coordinate offset within the tensor's logical coordinate space
  CUTLASS_HOST_DEVICE
  static TensorCoord initial_offset(int thread_id) {

    int sub_warp_id = thread_id / Detail::WarpThreadArrangement::kContiguous;
    int lane_id = thread_id % Detail::WarpThreadArrangement::kContiguous;

    layout::PitchLinearCoord thread_offset_in_threadblock_tile_base{
      lane_id * kElementsPerAccess,
      sub_warp_id
    };

    return thread_offset_in_threadblock_tile_base;
  }
};


} // namespace transform
} // namespace cutlass

////////////////////////////////////////////////////////////////////////////////