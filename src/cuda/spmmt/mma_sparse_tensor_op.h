#include "mma_sparse_sm80.h"
#include "helper.h"
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
class SparseMmaTensorOpV2 {
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

  /// Equivalant base dense mma
  using Base = MmaTensorOp<Shape, ElementA, LayoutA, ElementB, LayoutB,
                           ElementC, LayoutC, Policy, PartitionsK_,
                           AccumulatorsInRowMajor, Enable>;

  /// Underlying matrix multiply operator (concept: arch::Mma)
  // using ArchMmaOperator = typename Base::ArchMmaOperator;
  using ArchMmaOperator = typename cutlass::arch::SparseMmaV2<gemm::GemmShape<16, 8, 32>, 32, ElementA, layout::RowMajor,
           ElementB, layout::ColumnMajor, float, layout::RowMajor, cutlass::arch::OpMultiplyAdd, true>;

  /// Indicates math operator 
  using MathOperator = typename ArchMmaOperator::Operator;
  
  /// Architecture tag from underlying instruction
  using ArchTag = typename Base::ArchTag;

  /// Indicates class of matrix operator
  using OperatorClass = typename Base::OperatorClass;

  /// Shape of underlying instruction
  using InstructionShape = typename Base::InstructionShape;

  /// Complex transform on A operand
  static ComplexTransform const kTransformA = Base::kTransformA;

  /// Complex transform on B operand
  static ComplexTransform const kTransformB = Base::kTransformB;

  /// Number of threads participating in warp-level matrix product
  static int const kThreadCount = 32;

  /// Number of partitions along K dimension
  static int const kPartitionsK = PartitionsK_;

  /// Sparsity in Operand A
  static int const kSparse = Policy::Operator::kSparse;

  /// Meta data size in bits 
  static int const kMetaSizeInBits = Policy::Operator::kMetaSizeInBits;

  /// Max ID2
  static int const kMaxID2 = Policy::Operator::kMaxID2;

  /// Data type of meta E that is moved at the same time
  using ElementE =
      typename cutlass::platform::conditional<kMaxID2 == 1, uint32_t,
                                              uint16_t>::type;

  /// Number of ElementA that is associated with one ElementE
  static int const kElementsPerElementE =
      128 / cutlass::sizeof_bits<ElementA>::value;

  /// Meta data is essentially interleaved but mapped to ColumnMajor internally
  static int const kInterleaved = 2;

  /// Layout of meta E 
  using LayoutE = cutlass::layout::ColumnMajor;

 public:

  /// Iterates over the A operand in memory
 using IteratorA = MmaTensorOpMultiplicandTileIterator<
     MatrixShape<Shape::kM, Shape::kK / kSparse>, Operand::kA, ElementA,
     LayoutA,
     MatrixShape<Policy::Operator::Shape::kM,
                 Policy::Operator::Shape::kK / kSparse>,
     Policy::OpDelta::kRow, kThreadCount, kPartitionsK>;

 /// Storage for A tile
 using FragmentA = typename IteratorA::Fragment;

 /// Storage for transformed A tile
 using TransformedFragmentA =
     Array<typename Policy::Operator::ElementA, FragmentA::kElements>;

 /// Iterates over the B operand in memory
 using IteratorB = typename Base::IteratorB;

 /// Storage for B tile
 using FragmentB = typename Base::FragmentB;

 /// Storage for transformed B tile
 using TransformedFragmentB = typename Base::TransformedFragmentB;

 /// Iterates over the C operand in memory
 using IteratorC = typename Base::IteratorC;

 /// Storage for C tile
 using FragmentC = typename Base::FragmentC;

 /// Iterates over the E operand in memory
 using IteratorE = SparseMmaTensorOpMetaTileIterator<
     MatrixShape<Shape::kM * kInterleaved,
                 Shape::kK / kSparse / kElementsPerElementE / kInterleaved>,
     ElementE, LayoutE,
     MatrixShape<Policy::Operator::Shape::kM,
                 Policy::Operator::Shape::kK / kSparse / kElementsPerElementE /
                     kInterleaved>,
     Policy::OpDelta::kRow, kThreadCount, kPartitionsK>;

 /// Storage for E tile
 using FragmentE = typename IteratorE::Fragment;

 /// Number of mma operations performed
 using MmaIterations = typename Base::MmaIterations;

public:

  /// Underlying matrix multiply operator (concept: arch::Mma)
  ArchMmaOperator mma;

public:

  //
  // Methods
  //

  /// Ctor
  CUTLASS_DEVICE
  SparseMmaTensorOpV2() {}

  /// Performs a warp-level matrix multiply-accumulate operation
  CUTLASS_DEVICE
  void operator()(
    FragmentC &D, 
    TransformedFragmentA const &A, 
    TransformedFragmentB const &B, 
    FragmentC const &C,
    FragmentE const &E,
    ElementB (&I)[12]
  ) const {

    using MmaOperandA = typename Policy::Operator::FragmentA;
    using MmaOperandB = typename Policy::Operator::FragmentB;
    using MmaOperandC = typename Policy::Operator::FragmentC;
    using MmaOperandE = typename Policy::Operator::FragmentE;

    #if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 800)

    D = C;

    MmaOperandA const *ptr_A = reinterpret_cast<MmaOperandA const *>(&A);
    MmaOperandB const *ptr_B = reinterpret_cast<MmaOperandB const *>(&B);
    MmaOperandB const *ptr_I = reinterpret_cast<MmaOperandB const *>(&I);
    MmaOperandC *ptr_D = reinterpret_cast<MmaOperandC *>(&D);
    MmaOperandE const *ptr_E = reinterpret_cast<MmaOperandE const *>(&E);

      CUTLASS_PRAGMA_UNROLL
      for (int m = 0; m < MmaIterations::kRow; ++m) {

        int id2 = m % kMaxID2;

        // Convert sparse A to dense
        ElementA dense_A[16];
        mma.todense(ptr_A[m], dense_A, ptr_I[0], ptr_E[(m / kMaxID2)], id2);

        CUTLASS_PRAGMA_UNROLL
        for (int n = 0; n < MmaIterations::kColumn; ++n) {

          int n_serpentine = ((m % 2) ? (MmaIterations::kColumn - 1 - n) : n);

          if (AccumulatorsInRowMajor) {  // matrix B is reordered
            mma(
              ptr_D[n_serpentine + m * MmaIterations::kColumn],
              dense_A,
              ptr_B[n_serpentine],
              ptr_D[n_serpentine + m * MmaIterations::kColumn]);
          } else {
            mma(ptr_D[m + n_serpentine * MmaIterations::kRow],
                dense_A,
                ptr_B[n_serpentine],
                ptr_D[m + n_serpentine * MmaIterations::kRow]);
            // mma(ptr_D[m + n_serpentine * MmaIterations::kRow],
            //     ptr_A[m],
            //     ptr_I[0],
            //     ptr_D[m + n_serpentine * MmaIterations::kRow],
            //     ptr_E[(m / kMaxID2)],
            //     id2);
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

    #if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 800)
    //
    // Define conversions from source type to instruction type
    //
    FloatRoundStyle const kRoundA =
        PreferredRoundingMode<typename ArchMmaOperator::ElementA,
                              ElementA>::kRound;
    FloatRoundStyle const kRoundB =
        PreferredRoundingMode<typename ArchMmaOperator::ElementB,
                              ElementB>::kRound;
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