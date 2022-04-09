#include "mma_sparse_tensor_op_trans.h"
#include "mma_sparse_sm80.h"
namespace cutlass {
namespace gemm {
namespace warp {

/////////////////////////////////////////////////////////////////////////////////////////////////

template <
    /// Size of the Gemm problem - concept: gemm::GemmShape<>
    typename WarpShape_,
    /// Shape of one matrix production operation (concept: GemmShape)
    typename InstructionShape_,
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
    /// Operator describing the tensor operation
    typename Operator_ = arch::OpMultiplyAdd,
    /// Number of partitions along K dimension
    int PartitionsK = 1,
    /// Store the accumulators in row major or column major.  Row major is used
    /// when output layout is interleaved.
    bool AccumulatorsInRowMajor = false
>
struct DefaultSparseMmaTensorOpTrans;

/////////////////////////////////////////////////////////////////////////////////////////////////

/// Partial specialization for m-by-n-by-kgroup
template <
    /// Shape of one matrix production operation (concept: GemmShape)
    typename WarpShape_,
    /// Shape of one matrix production operation (concept: GemmShape)
    typename InstructionShape_,
    /// Data type of A elements
    typename ElementA,
    /// Layout of A matrix (concept: MatrixLayout)
    typename LayoutA,
    /// Data type of B elements
    typename ElementB,
    /// Layout of B matrix (concept: MatrixLayout)
    typename LayoutB,
    /// Element type of C matrix
    typename ElementC,
    /// Layout of C matrix (concept: MatrixLayout)
    typename LayoutC,
    /// Operator describing the tensor operation
    typename Operator_,
    /// Number of partitions along K dimension
    int PartitionsK,
    /// Store the accumulators in row major or column major.  Row major is used
    /// when output layout is interleaved.
    bool AccumulatorsInRowMajor>
struct DefaultSparseMmaTensorOpTrans {
  using Policy = cutlass::gemm::warp::MmaTensorOpPolicy<
      cutlass::arch::SparseMmaV2<InstructionShape_, 32, ElementA,
                               cutlass::layout::RowMajor, ElementB,
                               cutlass::layout::ColumnMajor, ElementC,
                               cutlass::layout::RowMajor, Operator_, true>,
      cutlass::MatrixShape<1, 1> >;

  // Define the warp-level tensor op
  using Type = cutlass::gemm::warp::SparseMmaTensorOpTrans<
      WarpShape_, ElementA, LayoutA, ElementB, LayoutB, ElementC, LayoutC,
      Policy, PartitionsK, AccumulatorsInRowMajor>;
};

/////////////////////////////////////////////////////////////////////////////////////////////////

} // namespace warp
} // namespace gemm
} // namespace cutlass