////////////////////////////////////////////////////////////////////////////////

namespace cutlass {
namespace gemm {
namespace threadblock {

////////////////////////////////////////////////////////////////////////////////

/// Structure to compute the matrix product targeting CUDA cores and SIMT math
/// instructions.
template <
    /// Size of the Gemm problem - concept: gemm::GemmShape<>
    typename Shape_,
    /// Policy describing tuning details (concept: MmaPolicy)
    typename Policy_,
    /// Number of stages,
    int Stages,
    /// Used for partial specialization
    typename Enable = bool>
class SparseMmaBaseTrans : public SparseMmaBase<Shape_, Policy_, Stages> {
 public:
  using Base = SparseMmaBase<Shape_, Policy_, Stages>;

  /// Shared storage object needed by threadblock-scoped GEMM
  class SharedStorage : public Base::SharedStorage {
   public:
    //
    // Type definitions
    //

    /// Shape of the A matrix operand in shared memory
    // ShapeA is [kK * Stages, kM/2]
    using ShapeA = MatrixShape<Base::Shape::kK * Base::kStages + Base::Policy::SmemPaddingA::kRow,
                               Base::Shape::kM / Base::kSparse +
                                   Base::Policy::SmemPaddingA::kColumn>;

    /// Shape of the E matrix operand in shared memory
    using ShapeE =
        MatrixShape<Base::Shape::kK * 2 * Base::kStages + Base::Policy::SmemPaddingE::kRow,
                    Base::Shape::kM / Base::kSparse / Base::kElementsPerElementE / 2 +
                        Base::Policy::SmemPaddingE::kColumn>;

   public:
    //
    // Data members
    //

    /// Buffer for A operand
    AlignedBuffer<typename Base::Operator::ElementA, ShapeA::kCount> operand_A;

    /// Buffer for E operand
    AlignedBuffer<typename Base::Operator::ElementE, ShapeE::kCount> operand_E;

   public:

    //
    // Methods
    //

    /// Returns a layout object for the A matrix
    CUTLASS_DEVICE
    static typename Base::Operator::LayoutA LayoutA() {
      return Base::Operator::LayoutA::packed({ShapeA::kRow, ShapeA::kColumn});
    }

    /// Returns a layout object for the E matrix
    CUTLASS_HOST_DEVICE
    static typename Base::Operator::LayoutE LayoutE() {
      return Base::Operator::LayoutE::packed({ShapeE::kRow, ShapeE::kColumn});
    }

    /// Returns a TensorRef to the A operand
    CUTLASS_HOST_DEVICE
    typename Base::TensorRefA operand_A_ref() {
      return typename Base::TensorRefA{operand_A.data(), LayoutA()};
    }

    /// Returns a TensorRef to the E operand
    CUTLASS_HOST_DEVICE
    typename Base::TensorRefE operand_E_ref() {
      return typename Base::TensorRefE{operand_E.data(), LayoutE()};
    }
  };

 protected:

  //
  // Data members
  //

  /// Iterator to load a warp-scoped tile of A operand from shared memory
  typename Base::Operator::IteratorA warp_tile_iterator_A_;

  /// Iterator to load a warp-scoped tile of B operand from shared memory
  typename Base::Operator::IteratorB warp_tile_iterator_B_;

  /// Iterator to load a warp-scoped tile of E operand from shared memory
  typename Base::Operator::IteratorE warp_tile_iterator_E_;


public:

  /// Construct from tensor references
  CUTLASS_DEVICE
  SparseMmaBaseTrans(
      ///< Shared storage needed for internal use by threadblock-scoped GEMM
      SharedStorage &shared_storage,
      ///< ID within the threadblock
      int thread_idx,
      ///< ID of warp
      int warp_idx,
      ///< ID of each thread within a warp
      int lane_idx
    ):
      Base(shared_storage, thread_idx, warp_idx, lane_idx),
      warp_tile_iterator_A_(shared_storage.operand_A_ref(), lane_idx),
      warp_tile_iterator_B_(shared_storage.operand_B_ref(), lane_idx),
      warp_tile_iterator_E_(shared_storage.operand_E_ref(), lane_idx) {}

};

/////////////////////////////////////////////////////////////////////////////////////////////////

}  // namespace threadblock
}  // namespace gemm
}  // namespace cutlass

/////////////////////////////////////////////////////////////////////////////////////////////////