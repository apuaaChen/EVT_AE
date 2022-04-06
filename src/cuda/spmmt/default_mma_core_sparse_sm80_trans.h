#include "default_mma_sparse_tensor_op_trans.h"
////////////////////////////////////////////////////////////////////////////////

namespace cutlass {
namespace gemm {
namespace threadblock {

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
    int Stages,
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
    bool AccumulatorsInRowMajor = false
    /// Cache operation of operand A
    , cutlass::arch::CacheOperation::Kind CacheOpA =
        cutlass::arch::CacheOperation::Global,
    /// Cache operation of operand B
    cutlass::arch::CacheOperation::Kind CacheOpB =
        cutlass::arch::CacheOperation::Global
>
struct DefaultSparseMmaCoreTrans;


////////////////////////////////////////////////////////////////////////////////

/// Partial specialization:
///
///   A: row-major
///   B: column-major
///   Operator: tensor op class
///
/// This uses the default warp-level operator given tile sizes
template <
    /// ThreadblockShape: Shape of threadblock-scoped matrix multiply operator (concept:
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
struct DefaultSparseMmaCoreTrans<Shape_, WarpShape_, InstructionShape_, ElementA_,
                      layout::RowMajor, ElementB_, layout::ColumnMajor,
                      ElementC_, LayoutC_, arch::OpClassTensorOp, Stages,
                      Operator_, false, CacheOpA, CacheOpB> {
  using Base = cutlass::gemm::threadblock::DefaultSparseMmaCore<Shape_, WarpShape_, InstructionShape_, ElementA_,
                      layout::RowMajor, ElementB_, layout::ColumnMajor,
                      ElementC_, LayoutC_, arch::OpClassTensorOp, Stages,
                      Operator_, false, CacheOpA, CacheOpB>;
  using Shape = typename Base::Shape;
  using WarpShape = typename Base::WarpShape;
  using InstructionShape = typename Base::InstructionShape;
  using ElementA = typename Base::ElementA;
  using LayoutA = typename Base::LayoutA;
  using ElementB = typename Base::ElementB;
  using LayoutB = typename Base::LayoutB;
  using ElementC = typename Base::ElementC;
  using LayoutC = typename Base::LayoutC;
  static int const kStages = Base::Stages;
  static cutlass::arch::CacheOperation::Kind const kCacheOpA = CacheOpA;
  static cutlass::arch::CacheOperation::Kind const kCacheOpB = CacheOpB;

  static int const kSparse = 2;
  using WarpCount = typename Base::WarpCount;
  static int const kWarpSize = Base::kWarpSize; 
  static int const kThreads = Base::kThreads;
  static int const kAccessSizeInBits = Base::kAccessSizeInBits;
  using Operator = typename Base::Operator;
  static int const kCrosswiseB = Base::kCrosswiseB;
  static int const kWarpThreadArrangementContiguousB = Base::kWarpThreadArrangementContiguousB;
  static int const kWarpThreadArrangementStridedB = Base::kWarpThreadArrangementStridedB;

  // Warp thread arrangement
  // Shape::kM = 128, kSparse = 2, (kAccessSizeInBits / sizeof_bits<ElementA>::value) = 8 for bf16/f16
  static int const kWarpThreadArrangementContiguousA =
      Shape::kM / kSparse / (kAccessSizeInBits / sizeof_bits<ElementA>::value);
  
  // kWarpSize = 32, kWarpThreadArrangementContiguousA = 8 for bf16/f16
  static int const kWarpThreadArrangementStridedA =
      kWarpSize / kWarpThreadArrangementContiguousA;

  //
  // Shared memory layouts
  //

  using SmemLayoutA = layout::RowMajorTensorOpMultiplicandCrosswise<
      sizeof_bits<ElementA>::value, Shape::kM / kSparse>;

  // Shared memory layout
  using SmemLayoutB = typename Base::SmemLayoutB;

  //
  // Iterators to write to shared memory
  //

  /// ThreadMap of iterator A
  using IteratorThreadMapA = transform::PitchLinearWarpRakedThreadMap<
      layout::PitchLinearShape<Shape::kM / kSparse, Shape::kK>, kThreads,
      layout::PitchLinearShape<kWarpThreadArrangementContiguousA,
                               kWarpThreadArrangementStridedA>,
      kAccessSizeInBits / sizeof_bits<ElementA>::value>;

  /// Shared memory iterator to A operand
  using SmemIteratorA = transform::threadblock::RegularTileAccessIterator<
      MatrixShape<Shape::kK, Shape::kM / kSparse>, ElementA, SmemLayoutA, 0,
      IteratorThreadMapA>;

  /// ThreadMap of iterator B
  using IteratorThreadMapB = typename Base::IteratorThreadMapB;

  /// Shared memory iterator to B operand
  using SmemIteratorB = typename Base::SmemIteratorB;

  //
  // Warp-level matrix multiply operator
  //

  // Define the warp-level tensor op
  using MmaTensorOp = typename cutlass::gemm::warp::DefaultSparseMmaTensorOpTrans<
      WarpShape, InstructionShape, ElementA, SmemLayoutA, ElementB, SmemLayoutB,
      ElementC, LayoutC, Operator, WarpCount::kK>::Type;

  /// Cache operation of operand E
  static cutlass::arch::CacheOperation::Kind const kCacheOpE =
      cutlass::arch::CacheOperation::Global;

  static int const kInterleavedE = MmaTensorOp::kInterleaved;
  static int const kMetaSizeInBits = MmaTensorOp::kMetaSizeInBits;
  static int const kMaxID2 = MmaTensorOp::kMaxID2;
  static int const kElementsPerElementE = MmaTensorOp::kElementsPerElementE;

  using ElementE = typename MmaTensorOp::ElementE;
  using GmemLayoutE = cutlass::layout::ColumnMajorInterleaved<kInterleavedE>;

  // Shared memory layout.  Interleaved layout is mapped to PitchLinear layout.
  using SmemLayoutE = typename MmaTensorOp::LayoutE;

  /// ThreadMap of iterator E
  static int const kElementsPerAccessE = Base::kElementsPerAccessE;
  static int const kThreadsE = Base::kThreadsE;

  using IteratorThreadMapE = transform::PitchLinearStripminedThreadMap<
      layout::PitchLinearShape<Shape::kK * kInterleavedE,
                               Shape::kM / kSparse / kElementsPerElementE /
                                   kInterleavedE>,
      kThreadsE, kElementsPerAccessE>;


  /// Shared memory iterator to E operand
  using SmemIteratorE = transform::threadblock::RegularTileAccessIterator<
      MatrixShape<Shape::kK * kInterleavedE,
                  Shape::kM / kSparse / kElementsPerElementE / kInterleavedE>,
      ElementE, SmemLayoutE, 0, IteratorThreadMapE>;

  /// Policy used to define MmaPipelined
  using MmaPolicy =
      SparseMmaPolicy<MmaTensorOp, MatrixShape<0, 0>, MatrixShape<0, 0>,
                      MatrixShape<0, 0>, WarpCount::kK>;
};

////////////////////////////////////////////////////////////////////////////////

}  // namespace threadblock
}  // namespace gemm
}  // namespace cutlass