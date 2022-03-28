#include "helper.h"
#include "mma_sm80.h"
/////////////////////////////////////////////////////////////////////////////////////////////////

namespace cutlass {
namespace gemm {
namespace warp {

/////////////////////////////////////////////////////////////////////////////////////////////////

/// Structure to compute the matrix product targeting CUDA cores and SIMT math instructions.
template <
  /// Size of the Gemm problem - concept: gemm::GemmShape<>
  typename Shape_,
  /// Data type of A elements
  typename ElementA_,
  /// Layout of A matrix (concept: MatrixLayout)
  typename LayoutA_,
  /// Data type of B elements
  typename ElementB_,
  /// Layout of B matrix (concept: MatrixLayout)
  typename LayoutB_,
  /// Element type of C matrix
  typename ElementC_,
  /// Layout of C matrix (concept: MatrixLayout)
  typename LayoutC_,
  /// Policy describing warp-level MmaTensorOp (concept: MmaTensorOp policy)
  typename Policy_,
  /// Number of partitions along K dimension
  int PartitionsK_ = 1,
  /// Store the accumulators in row major or column major.  Row major is used
  /// when output layout is interleaved.
  bool AccumulatorsInRowMajor = false,
  /// Used for partial specialization
  typename Enable = bool
>
class MmaTensorOpV2 {
public:
  /// Shape of warp-level matrix operation (concept: GemmShape)
  using Shape = Shape_;

  /// Data type of multiplicand A
  using ElementA = ElementA_;

  /// Layout of multiplicand A
  using LayoutA = LayoutA_;

  /// Data type of multiplicand B
  using ElementB = ElementB_;

  /// Layout of multiplicand B
  using LayoutB = LayoutB_;

  /// Data type of accumulator matrix C
  using ElementC = ElementC_;

  /// Layout of accumulator matrix C
  using LayoutC = LayoutC_;

  /// Shape of the warp in units of thread (concept: MmaLanePolicySimt)
  using Policy = Policy_;

  /// Underlying matrix multiply operator (concept: arch::Mma)
  using ArchMmaOperator = typename Policy::Operator;

  /// Indicates math operator 
  using MathOperator = typename ArchMmaOperator::Operator;

  /// Architecture tag from underlying instruction
  using ArchTag = typename ArchMmaOperator::ArchTag;

  /// Indicates class of matrix operator
  using OperatorClass = arch::OpClassTensorOp;

  /// Shape of underlying instruction
  using InstructionShape = typename ArchMmaOperator::Shape;

  /// Complex transform on A operand
  static ComplexTransform const kTransformA = ComplexTransform::kNone;

  /// Complex transform on B operand
  static ComplexTransform const kTransformB = ComplexTransform::kNone;

  /// Number of threads participating in warp-level matrix product
  static int const kThreadCount = 32;

  /// Number of partitions along K dimension
  static int const kPartitionsK = PartitionsK_;

public:

  /// Iterates over the A operand in memory
  using IteratorA = MmaTensorOpMultiplicandTileIterator<
     MatrixShape<Shape::kM, Shape::kK>, Operand::kA, ElementA, LayoutA,
     MatrixShape<ArchMmaOperator::Shape::kM, ArchMmaOperator::Shape::kK>,
     Policy::OpDelta::kRow, kThreadCount, kPartitionsK>;

  /// Storage for A tile
  using FragmentA = typename IteratorA::Fragment;

  /// Storage for transformed A tile
  using TransformedFragmentA =
      Array<typename ArchMmaOperator::ElementA, FragmentA::kElements>;

  /// Iterates over the B operand in memory
  using IteratorB = MmaTensorOpMultiplicandTileIterator<
      MatrixShape<Shape::kK, Shape::kN>, Operand::kB, ElementB, LayoutB,
      MatrixShape<ArchMmaOperator::Shape::kK, ArchMmaOperator::Shape::kN>,
      Policy::OpDelta::kRow, kThreadCount, kPartitionsK>;

  /// Storage for B tile
  using FragmentB = typename IteratorB::Fragment;

  /// Storage for transformed B tile
  using TransformedFragmentB =
      Array<typename ArchMmaOperator::ElementB, FragmentB::kElements>;

  /// Iterates over the C operand in memory
  using IteratorC = MmaTensorOpAccumulatorTileIterator<
     MatrixShape<Shape::kM, Shape::kN>, ElementC, LayoutC,
     typename ArchMmaOperator::Shape, typename Policy::OpDelta>;

  /// Storage for C tile
  using FragmentC = typename IteratorC::Fragment;

  /// Number of mma operations performed
  using MmaIterations = MatrixShape<
    (Shape::kM + ArchMmaOperator::Shape::kM - 1) / ArchMmaOperator::Shape::kM,
    (Shape::kN + ArchMmaOperator::Shape::kN - 1) / ArchMmaOperator::Shape::kN / 2
  >;

public:

  /// Underlying matrix multiply operator (concept: arch::Mma)
  using ArchMmaOperatorV2 = typename cutlass::arch::MmaV2<cutlass::gemm::GemmShape<16, 16, 16>, 32, typename IteratorA::Element,
                         cutlass::layout::RowMajor, typename IteratorB::Element, 
                         cutlass::layout::ColumnMajor, float, 
                         cutlass::layout::RowMajor, cutlass::arch::OpMultiplyAdd>;
  ArchMmaOperatorV2 mma;

public:

  //
  // Methods
  //

  /// Ctor
  CUTLASS_DEVICE
  MmaTensorOpV2() {}

  /// Performs a warp-level matrix multiply-accumulate operation
  CUTLASS_DEVICE
  void operator()(
    FragmentC &D, 
    TransformedFragmentA const &A, 
    TransformedFragmentB const &B, 
    FragmentC const &C
  ) const {

    using MmaOperandA = typename ArchMmaOperator::FragmentA;
    using MmaOperandB = typename ArchMmaOperator::FragmentB;
    using MmaOperandC = typename ArchMmaOperator::FragmentC;

    D = C;

    MmaOperandA const *ptr_A = reinterpret_cast<MmaOperandA const *>(&A);
    MmaOperandB const *ptr_B = reinterpret_cast<MmaOperandB const *>(&B);
    MmaOperandC *ptr_D = reinterpret_cast<MmaOperandC *>(&D);

    #if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 800)
      // Serpentine visitation order maximizing reuse of Ra
      CUTLASS_PRAGMA_UNROLL
      for (int m = 0; m < MmaIterations::kRow; ++m) {

        CUTLASS_PRAGMA_UNROLL
        for (int n = 0; n < MmaIterations::kColumn; ++n) {

          int n_serpentine = ((m % 2) ? (MmaIterations::kColumn - 1 - n) : n) * 2;

          if (AccumulatorsInRowMajor) {  // matrix B is reordered
            mma(
              ptr_D[n_serpentine + m * MmaIterations::kColumn * 2],
              ptr_D[n_serpentine + m * MmaIterations::kColumn * 2 + 1],
              ptr_A[m],
              ptr_B[n_serpentine],
              ptr_D[n_serpentine + m * MmaIterations::kColumn * 2],
              ptr_D[n_serpentine + m * MmaIterations::kColumn * 2 + 1]);
          } else {
            mma(ptr_D[m + n_serpentine * MmaIterations::kRow],
                ptr_D[m + (n_serpentine + 1) * MmaIterations::kRow],
                ptr_A[m],
                ptr_B[n_serpentine],
                ptr_D[m + n_serpentine * MmaIterations::kRow],
                ptr_D[m + (n_serpentine + 1) * MmaIterations::kRow]);
          }
        }
      }
    #else
      assert(0);
    #endif
  }

  /// Transform the mma operands to the required types
  CUTLASS_DEVICE
  void transform(TransformedFragmentA &dst_A, TransformedFragmentB &dst_B,
                 FragmentA const &A, FragmentB const &B) const {

    //
    // Define conversions from source type to instruction type
    //
    FloatRoundStyle const kRoundA =
        PreferredRoundingMode<typename ArchMmaOperator::ElementA,
                              ElementA>::kRound;
    FloatRoundStyle const kRoundB =
        PreferredRoundingMode<typename ArchMmaOperator::ElementB,
                              ElementB>::kRound;
    #if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ < 800)
      detail::ConvertAndPack<typename ArchMmaOperator::ElementA, ElementA,
                            FragmentA::kElements, kRoundA>
          convert_A;
      NumericArrayConverter<typename ArchMmaOperator::ElementB, ElementB,
                            FragmentB::kElements / 2, kRoundB>
          convert_B;
      Array<ElementB, FragmentB::kElements / 2> const *ptr_B =
          reinterpret_cast<Array<ElementB, FragmentB::kElements / 2> const *>(&B);
      Array<typename ArchMmaOperator::ElementB, FragmentB::kElements / 2> *
          ptr_dst_B = reinterpret_cast<Array<typename ArchMmaOperator::ElementB,
                                             FragmentB::kElements / 2> *>(&dst_B);
  
      dst_A = convert_A(A);
  
      ptr_dst_B[0] = convert_B(ptr_B[0]);
      ptr_dst_B[1] = convert_B(ptr_B[1]);

    #elif defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 800)
      detail::ConvertAndPack<typename ArchMmaOperator::ElementA, ElementA,
                            FragmentA::kElements / 2, kRoundA>
          convert_A;
      NumericArrayConverter<typename ArchMmaOperator::ElementB, ElementB,
                            FragmentB::kElements, kRoundB>
          convert_B;
      Array<ElementA, FragmentA::kElements / 2> const *ptr_A =
          reinterpret_cast<Array<ElementA, FragmentA::kElements / 2> const *>(&A);
      Array<typename ArchMmaOperator::ElementA, FragmentA::kElements / 2> *
          ptr_dst_A = reinterpret_cast<Array<typename ArchMmaOperator::ElementA,
                                             FragmentA::kElements / 2> *>(&dst_A);
  
      dst_B = convert_B(B);
  
      ptr_dst_A[0] = convert_A(ptr_A[0]);
      ptr_dst_A[1] = convert_A(ptr_A[1]);
    #else
      assert(0);
    #endif
  }
};

/////////////////////////////////////////////////////////////////////////////////////////////////

} // namespace warp
} // namespace gemm
} // namespace cutlass

/////////////////////////////////////////////////////////////////////////////////////////////////