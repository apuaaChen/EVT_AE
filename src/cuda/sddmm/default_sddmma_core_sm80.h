#include "pitch_linear_thread_map.h"
#include "regular_tile_access_iterator_tensor_op.h"
////////////////////////////////////////////////////////////////////////////////

namespace cutlass {
namespace gemm {
namespace threadblock {

////////////////////////////////////////////////////////////////////////////////

////////////////////////////////////////////////////////////////////////////////


/// Template defininng default matrix multiply operators inferred from threadblock tile size,
/// global memory data layout, and target math instruction.
template <
    /// Shape of threadblock-scoped matrix multiply operator
    typename Shape,
    /// Shape of warp-level matrix multiply operator
    typename WarpShape,
    /// Shape of one matrix production operation (concept: GemmShape)
    typename InstructionShape,
    /// Element data type of A operand
    typename ElementA,
    /// Layout of operand A
    typename LayoutA,
    /// Element data type of B operand
    typename ElementB,
    /// Layout of operand B
    typename LayoutB,
    /// Data type of accumulator
    typename ElementC,
    /// Layout of accumulator
    typename LayoutC,
    /// Indicates type of math operator (arch::OpClassSimt or arch::OpClassTensorOp)
    typename OperatorClass,
    /// Number of stages
    int Stages = 2,
    /// Operation performed by MMA
    typename Operator = typename platform::conditional<
        (platform::is_same<OperatorClass,
                           cutlass::arch::OpClassTensorOp>::value) &&
            (platform::is_same<ElementA, int8_t>::value ||
             platform::is_same<ElementA, int4b_t>::value ||
             platform::is_same<ElementA, uint8_t>::value ||
             platform::is_same<ElementA, uint4b_t>::value),
        cutlass::arch::OpMultiplyAddSaturate,
        cutlass::arch::OpMultiplyAdd>::type,
    /// Store the accumulators in row major or column major.  Row major is used
    /// when output layout is interleaved.
    bool AccumulatorsInRowMajor = false,
    /// Cache operation of operand A
    cutlass::arch::CacheOperation::Kind CacheOpA =
        cutlass::arch::CacheOperation::Global,
    /// Cache operation of operand B
    cutlass::arch::CacheOperation::Kind CacheOpB =
        cutlass::arch::CacheOperation::Global,
    /// per-element transformation for elements of A
    ComplexTransform TransformA = ComplexTransform::kNone,
    /// per-element transformation for elements of B
    ComplexTransform TransformB = ComplexTransform::kNone,
    bool IsComplex = false // (is_complex<ElementA>::value || is_complex<ElementB>::value)
>
struct DefaultSDDMmaCore;

/// Partial specialization:
///
///   A: row-major
///   B: column-major
///   Operator: tensor op class
///
/// This uses the default warp-level operator given tile sizes
template <
    /// Shape of threadblock-scoped matrix multiply operator (concept:
    /// GemmShape)
    typename Shape_,
    /// Shape of warp-level matrix multiply operator (concept: GemmShape)
    typename WarpShape_,
    /// Shape of one matrix production operation (concept: GemmShape)
    typename InstructionShape_,
    /// Data type of A operand
    typename ElementA_,
    /// Data type of B operand
    typename ElementB_,
    /// Data type of accumulator
    typename ElementC_,
    /// Layout of accumulator
    typename LayoutC_,
    /// Number of stages
    int Stages,
    /// Operation performed by MMA
    typename Operator_,
    /// Cache operation of operand A
    cutlass::arch::CacheOperation::Kind CacheOpA,
    /// Cache operation of operand B
    cutlass::arch::CacheOperation::Kind CacheOpB>
struct DefaultSDDMmaCore<Shape_, WarpShape_, InstructionShape_, ElementA_,
                      layout::RowMajor, ElementB_, layout::ColumnMajor,
                      ElementC_, LayoutC_, arch::OpClassTensorOp, Stages,
                      Operator_, false, CacheOpA, CacheOpB> {
  using Shape = Shape_;
  using WarpShape = WarpShape_;
  using InstructionShape = InstructionShape_;
  using ElementA = ElementA_;
  using LayoutA = layout::RowMajor;
  using ElementB = ElementB_;
  using LayoutB = layout::ColumnMajor;
  using ElementC = ElementC_;
  using LayoutC = LayoutC_;
  static int const kStages = Stages;
  static cutlass::arch::CacheOperation::Kind const kCacheOpA = CacheOpA;
  static cutlass::arch::CacheOperation::Kind const kCacheOpB = CacheOpB;

  /// Number of warps present
  using WarpCount = GemmShape<Shape::kM / WarpShape::kM,
                              Shape::kN / WarpShape::kN, 
                              Shape::kK / WarpShape::kK>;

  // Divisility requirements
  static_assert(
      !(Shape::kM % WarpShape::kM) && !(Shape::kN % WarpShape::kN),
      "Threadblock-scoped GEMM should be divisible by warp-scoped GEMM size.");

  /// Number of threads per warp
  static int const kWarpSize = warp::WarpSize<arch::OpClassTensorOp>::value;

  /// Number of threads total
  static int const kThreads = WarpCount::kCount * kWarpSize;

  /// Size of a threadblock-scoped access
  static int const kAccessSizeInBits = 128;

  /// Default Operator
  using Operator = Operator_;

  // Warp thread arrangement
  static int const kWarpThreadArrangementContiguousA =
      Shape::kK / (kAccessSizeInBits / sizeof_bits<ElementA>::value);

  static int const kWarpThreadArrangementStridedA =
      kWarpSize / kWarpThreadArrangementContiguousA;

  static int const kWarpThreadArrangementContiguousB =
      Shape::kK / (kAccessSizeInBits / sizeof_bits<ElementA>::value);

  static int const kWarpThreadArrangementStridedB =
      kWarpSize / kWarpThreadArrangementContiguousB;

  //
  // Shared memory layouts
  //

  using SmemLayoutA = layout::RowMajorTensorOpMultiplicandCrosswise<
      sizeof_bits<ElementA>::value, Shape::kK>;

  // Shared memory layout
  using SmemLayoutB = layout::ColumnMajorTensorOpMultiplicandCrosswise<
      sizeof_bits<ElementB>::value, Shape::kK>;

  //
  // Iterators to write to shared memory
  //

  /// ThreadMap of iterator A
  using IteratorThreadMapA = transform::PitchLinearWarpRakedThreadMap<
      layout::PitchLinearShape<Shape::kK, Shape::kM>, kThreads,
      layout::PitchLinearShape<kWarpThreadArrangementContiguousA,
                               kWarpThreadArrangementStridedA>,
      kAccessSizeInBits / sizeof_bits<ElementA>::value>;

  /// Shared memory iterator to A operand
  using SmemIteratorA = transform::threadblock::RegularTileAccessIterator<
      MatrixShape<Shape::kM, Shape::kK>, ElementA, SmemLayoutA, 0,
      IteratorThreadMapA>;

  /// ThreadMap of iterator B
  using IteratorThreadMapB = transform::PitchLinearWarpRakedThreadMapV3<
      layout::PitchLinearShape<Shape::kK, Shape::kN>, kThreads,
      layout::PitchLinearShape<kWarpThreadArrangementContiguousB,
                               kWarpThreadArrangementStridedB>,
      kAccessSizeInBits / sizeof_bits<ElementB>::value>;

  /// Shared memory iterator to B operand
  using SmemIteratorB = transform::threadblock::RegularTileAccessIteratorV3<
      MatrixShape<Shape::kK, Shape::kN>, ElementB, SmemLayoutB, 1,
      IteratorThreadMapB>;

  //
  // Warp-level matrix multiply operator
  //

  // Define the warp-level tensor op
  using MmaTensorOp = typename cutlass::gemm::warp::DefaultMmaTensorOp<
      WarpShape, InstructionShape, ElementA, SmemLayoutA, ElementB, SmemLayoutB,
      ElementC, LayoutC, Operator, WarpCount::kK>::Type;

  /// Policy used to define MmaPipelined
  using MmaPolicy = MmaPolicy<MmaTensorOp, MatrixShape<0, 0>,
                                        MatrixShape<0, 0>, WarpCount::kK>;
};

////////////////////////////////////////////////////////////////////////////////

}  // namespace threadblock
}  // namespace gemm
}  // namespace cutlass