#include "mma_multistage.h"
#include "predicated_tile_access_iterator.h"
#include "pitch_linear_thread_map.h"
#include "regular_tile_access_iterator_tensor_op.h"
////////////////////////////////////////////////////////////////////////////////

namespace cutlass {
namespace gemm {
namespace threadblock {

////////////////////////////////////////////////////////////////////////////////

template <
    /// Element type for A matrix operand
    typename ElementA_,
    /// Layout type for A matrix operand
    typename LayoutA_,
    /// Access granularity of A matrix in units of elements
    int kAlignmentA,
    /// Element type for B matrix operand
    typename ElementB_,
    /// Layout type for B matrix operand
    typename LayoutB_,
    /// Access granularity of B matrix in units of elements
    int kAlignmentB,
    /// Element type for internal accumulation
    typename ElementAccumulator_,
    /// Layout type for C and D matrix operands
    typename LayoutC_,
    /// Operator class tag
    typename OperatorClass_,
    /// Tag indicating architecture to tune for
    typename ArchTag_,
    /// Threadblock-level tile size (concept: GemmShape)
    typename ThreadblockShape_,
    /// Warp-level tile size (concept: GemmShape)
    typename WarpShape_,
    /// Instruction-level tile size (concept: GemmShape)
    typename InstructionShape_,
    /// Number of stages used in the pipelined mainloop
    int Stages,
    /// Operation perfomed by GEMM
    typename Operator,
    /// Store the accumulators in row major or column major.  Row major is used
    /// when output layout is interleaved.
    bool AccumulatorsInRowMajor = false,
    /// Use zfill or predicate for out-of-bound cp.async
    SharedMemoryClearOption SharedMemoryClear = SharedMemoryClearOption::kNone
    >
struct DefaultMmaV2;
////////////////////////////////////////////////////////////////////////////////

/// Specialization for row-major output (OperatorClass TensorOp)
template <
    /// Element type for A matrix operand
    typename ElementA,
    /// Layout type for A matrix operand
    typename LayoutA,
    /// Access granularity of A matrix in units of elements
    int kAlignmentA,
    /// Element type for B matrix operand
    typename ElementB,
    /// Layout type for B matrix operand
    typename LayoutB,
    /// Access granularity of B matrix in units of elements
    int kAlignmentB,
    /// Element type for internal accumulation
    typename ElementAccumulator,
    /// Layout type for C and D matrix operand
    typename LayoutC,
    /// Tag indicating architecture to tune for
    typename ArchTag,
    /// Threadblock-level tile size (concept: GemmShape)
    typename ThreadblockShape,
    /// Warp-level tile size (concept: GemmShape)
    typename WarpShape,
    /// Instruction-level tile size (concept: GemmShape)
    typename InstructionShape,
    /// Number of stages used in the multistage mainloop
    int Stages,
    /// Operation perfomed by GEMM
    typename Operator,
    /// Use zfill or predicate for out-of-bound cp.async
    SharedMemoryClearOption SharedMemoryClear
    >
struct DefaultMmaV2<ElementA, LayoutA, kAlignmentA, ElementB, LayoutB,
                  kAlignmentB, ElementAccumulator, LayoutC,
                  arch::OpClassTensorOp, ArchTag, ThreadblockShape, WarpShape,
                  InstructionShape, Stages, Operator, false, SharedMemoryClear> {

  static_assert(platform::is_same<LayoutC, layout::RowMajor>::value
             || platform::is_same<LayoutC, layout::AffineRankN<2>>::value,
             "simt epilogue must be row major");

  static cutlass::arch::CacheOperation::Kind const CacheOpA =
      ((sizeof_bits<ElementA>::value * kAlignmentA) == 128)
          ? cutlass::arch::CacheOperation::Global
          : cutlass::arch::CacheOperation::Always;

  static cutlass::arch::CacheOperation::Kind const CacheOpB =
      ((sizeof_bits<ElementB>::value * kAlignmentB) == 128)
          ? cutlass::arch::CacheOperation::Global
          : cutlass::arch::CacheOperation::Always;

  // Define the MmaCore components
  using MmaCore = typename cutlass::gemm::threadblock::DefaultMmaCore<
      ThreadblockShape, WarpShape, InstructionShape, ElementA, LayoutA,
      ElementB, LayoutB, ElementAccumulator, LayoutC, arch::OpClassTensorOp,
      Stages, Operator, false, CacheOpA, CacheOpB>;

  // Define iterators over tiles from the A operand
  using ThreadMapA = typename MmaCore::IteratorThreadMapA;
  using AccessTypeA = cutlass::Array<ElementA, kAlignmentA>;
  using IteratorA =
      cutlass::transform::threadblock::PredicatedTileAccessIterator<
          cutlass::MatrixShape<ThreadblockShape::kM, ThreadblockShape::kK>,
          ElementA, LayoutA, 1, ThreadMapA, AccessTypeA>;

  // Define iterators over tiles from the B operand
  // using ThreadMapB = typename MmaCore::IteratorThreadMapB;
  using ThreadMapB = cutlass::transform::PitchLinearWarpRakedThreadMapV2<
      cutlass::layout::PitchLinearShape<ThreadblockShape::kN, ThreadblockShape::kK>, 128,
      layout::PitchLinearShape<8, 4>, 8>;

  using AccessTypeB = cutlass::Array<ElementB, kAlignmentB>;
  using IteratorB =
      cutlass::transform::threadblock::PredicatedTileAccessIteratorV2<
          cutlass::MatrixShape<ThreadblockShape::kK, ThreadblockShape::kN>,
          ElementB, LayoutB, 0, ThreadMapB, AccessTypeB>;
  using SmemIteratorB = cutlass::transform::threadblock::RegularTileAccessIteratorV2<
        cutlass::MatrixShape<ThreadblockShape::kK, ThreadblockShape::kN>,
        ElementB, layout::RowMajorTensorOpMultiplicandCongruous<sizeof_bits<ElementB>::value, int(128 / sizeof(ElementB))>,
        0, ThreadMapB>;

  // Define the threadblock-scoped multistage matrix multiply
  using ThreadblockMma = cutlass::gemm::threadblock::MmaMultistageV2<
      typename MmaCore::Shape, IteratorA, typename MmaCore::SmemIteratorA,
      MmaCore::kCacheOpA, IteratorB, SmemIteratorB,
      MmaCore::kCacheOpB, ElementAccumulator, LayoutC,
      typename MmaCore::MmaPolicy, Stages, SharedMemoryClear>;
};

} // namespace threadblock
} // namespace gemm
} // namespace cutlass 

////////////////////////////////////////////////////////////////////////////////