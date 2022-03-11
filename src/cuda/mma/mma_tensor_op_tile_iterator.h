////////////////////////////////////////////////////////////////////////////////

namespace cutlass {
namespace gemm {
namespace warp {

////////////////////////////////////////////////////////////////////////////////

template <
    /// Size of the matrix to load (concept: MatrixShape)
    typename Shape_,
    /// Operand identity
    Operand Operand,
    /// Data type of A elements
    typename Element_,
    /// Layout of operand
    typename Layout_,
    /// Shape of one matrix production operation (concept: GemmShape)
    typename InstructionShape_,
    /// Delta between *MMA operations (in units of *MMA operations, concept:
    /// MatrixShape)
    int OpDelta_,
    /// Number of threads participating in one matrix operation
    int Threads,
    /// Number of partitions along K dimension
    int PartitionsK_ = 1>
class MmaTensorOpMultiplicandTileIteratorV2;

////////////////////////////////////////////////////////////////////////////////
/// This tile iterator is specialized for 32-thread TensorOps. It uses LDSM to load from shared
/// memory and therefore must be initialized with a TensorRef to shared memory. 
///
/// Satisfies:
///   ReadableRandomAccessContiguousTileIteratorConcept
///
template <
    /// Size of the matrix to load (concept: PitchLinearShape)
    typename Shape_,
    /// Identifies A or B multiplicand
    Operand Operand_,
    /// Data type of elements
    typename Element_,
    /// Shape of one matrix product operation (concept: PitchLinearShape)
    typename InstructionShape_,
    /// Interval between adjacent *MMA instructions (in units of MMA
    /// instructions)
    int OpDelta_,
    /// Number of partitions along K dimension
    int PartitionsK_>
class MmaTensorOpMultiplicandTileIteratorV2<
    Shape_, Operand_, Element_,
    cutlass::layout::TensorOpMultiplicandCongruous<sizeof_bits<Element_>::value,
                                                   64>,
    InstructionShape_, OpDelta_, 32, PartitionsK_> {
 public:

  /// Shape of tile to load (concept: PitchLinearShape)
  using Shape = Shape_;

  /// Operand tag
  static Operand const kOperand = Operand_;

  static_assert(kOperand == Operand::kA || kOperand== Operand::kB,
    "MmaTensorOpMultiplicandIterator may only be instantiated for A or B operands to warp-level Mma.");

  /// Element type
  using Element = Element_;

  /// Layout of source tile
  using Layout = cutlass::layout::TensorOpMultiplicandCongruous<
      sizeof_bits<Element_>::value, 64>;

  /// Shape of one matrix product operation (concept: GemmShape)
  using InstructionShape = InstructionShape_;

  /// Delta between *MMA operations (in units of *MMA operations, concept: MatrixShape)
  static int const kOpDelta = OpDelta_;

  /// Number of participating threads
  static int const kThreads = 32;

  /// Number of partitions along K dimension
  static int const kPartitionsK = PartitionsK_;

  /// TensorRef type for loading element from a tensor
  using TensorRef = TensorRef<Element, Layout>;

  /// Index type
  using Index = typename TensorRef::Index;

  /// Long Index type
  using LongIndex = typename TensorRef::LongIndex;

  /// Long Index type
  using StrideIndex = typename TensorRef::Layout::Stride::Index;

  /// Coordinate for an element in the tensor
  using TensorCoord = typename TensorRef::TensorCoord;

  /// Internal structure of iterator - made public to enable introspection
  struct Policy {
    static_assert(
        !(Shape::kContiguous % InstructionShape::kContiguous),
        "Shape of warp-level Mma must be divisible by operator shape.");

    // Determine number of elements along outer dimension per individual LDSM op
    static int const kLdsmOpOuter = Layout::kElementsPerAccess;
    static int const kLdsmOpInner = 8;

    static_assert(!(Shape::kContiguous % kLdsmOpOuter),
      "Shape of warp-level mma must be divisible by LDSM's fundamental tile size.");

    static_assert(!(Shape::kStrided % kLdsmOpInner), 
      "Shape of warp-level mma must be divisible by LDSM's fundamental tile size.");

    /// Shape of one individual LDSM instruction
    static int const LdsmShapeStrided =
        InstructionShape::kStrided / kLdsmOpInner;
    static int const LdsmShapeContiguous = 4 / LdsmShapeStrided;
    using LdsmShape =
        layout::PitchLinearShape<LdsmShapeContiguous, LdsmShapeStrided>;

    /// Number and arrangement of LDSM instructions
    using LdsmIterations = layout::PitchLinearShape<
        Shape::kContiguous / Layout::kElementsPerAccess / LdsmShapeContiguous,
        1>;

    /// Number of groups for each tile
    static int const kGroupsPerTile =
        Shape::kStrided / InstructionShape::kStrided;
  };

private:

  /// Not working on this feature at the moment.
  static_assert(kOpDelta == 1,
    "Alternative arrangements not supported at present.");

  /// Number of internal pointers needed to reference shared memory
  static int const kPointerCount =
      Layout::TileShape::kContiguous / Policy::LdsmShape::kContiguous;

  /// Pointer type used for accesses
  using AccessType = Array<Element, Layout::kElementsPerAccess>;

  /// Internal counter used to jump to next K partition
  int k_group_idx_;

public:

  //
  // Derived quantities
  //

  /// Fragment object holding a thread's part of a tile
 using Fragment =
     Array<Element, Shape::kContiguous * InstructionShape::kStrided / kThreads>;

private:

  /// Layout object storing stride values
  StrideIndex stride_;

  /// Shared memory base pointers - not advanced
  AccessType const *pointer_[kPointerCount];

  /// Byte offset incremented as iterator advances
  Index byte_offset_;

public:
  
  /// Default ctor constructs null iterator
  CUTLASS_HOST_DEVICE
  MmaTensorOpMultiplicandTileIteratorV2(): stride_(0), byte_offset_(0) { }

  /// Constructor from TensorRef
  CUTLASS_DEVICE
  MmaTensorOpMultiplicandTileIteratorV2(
    TensorRef const &ref, 
    int lane_id
  ):
    stride_(ref.stride(0) / Layout::kElementsPerAccess), byte_offset_(0),
    k_group_idx_(0) {
      
    int quad_pair = (lane_id >> 3);
    int quad_quad = (lane_id >> 4);
    int lane_in_quad = (lane_id & 3);
    int lane_in_quad_pair = (lane_id & 7);
    int lane_in_quad_quad = (lane_id & 15);

    CUTLASS_PRAGMA_UNROLL
    for (int i = 0; i < kPointerCount; ++i) {
      int partition_contiguous_idx = -1;
      int access_contiguous_idx = -1;
      int access_strided_idx = -1;

      if (Policy::LdsmShape::kContiguous == 4) {
        // Matrix multiply 1688 A/B
        // Q0 Q1 Q2 Q3 (Q stands for 1 8x128bit block).
        // Four blocks are next to each other in the contiguous dimension.
        partition_contiguous_idx = ((lane_in_quad_pair >> 2) ^ i);
        access_contiguous_idx = (quad_pair ^ lane_in_quad);
        access_strided_idx = lane_in_quad_pair;
      }
      else if (Policy::LdsmShape::kContiguous == 2 &&
                 kOperand == Operand::kA) {
        // Matrix multiply 16816 A
        // Q0 Q2
        // Q1 Q3
        partition_contiguous_idx = ((lane_in_quad_pair >> 2) ^ (i >> 1));
        access_contiguous_idx =
            (((quad_pair & 1) + ((i & 1) << 1)) ^ lane_in_quad);
        access_strided_idx = lane_in_quad_pair + (lane_id >> 4 << 3);
      } else if (Policy::LdsmShape::kContiguous == 2 &&
                 kOperand == Operand::kB) {
        // Matrix multiply 16816 B
        // Q0 Q1
        // Q2 Q3
        partition_contiguous_idx = ((lane_in_quad_pair >> 2) ^ (i >> 1));
        access_contiguous_idx = ((quad_quad + ((i & 1) << 1)) ^ lane_in_quad);
        access_strided_idx = lane_in_quad_quad;
      } else if (Policy::LdsmShape::kContiguous == 1) {
        // Matrix multiply 16832.SP B
        // Q0
        // Q1
        // Q2
        // Q3
        partition_contiguous_idx = ((lane_in_quad_pair >> 2) ^ (i >> 2)); 
        access_contiguous_idx = ((i & 3) ^ lane_in_quad); 
        access_strided_idx = lane_id; 
      }

      int access_contiguous =
          partition_contiguous_idx * Layout::PartitionShape::kContiguous +
          access_contiguous_idx;

      int access_strided = access_strided_idx;

      pointer_[i] = reinterpret_cast<AccessType const *>(ref.data()) +
                    access_contiguous + access_strided * stride_;
    }
  }

  /// Adds a pointer offset to internal pointer(s) to advance through memory
  CUTLASS_DEVICE
  MmaTensorOpMultiplicandTileIteratorV2 &add_pointer_offset(LongIndex offset) {

    byte_offset_ += offset * sizeof(Element);

    return *this;
  }

  /// Advances an iterator along logical dimensions of matrix in units of whole tiles
  CUTLASS_HOST_DEVICE
  MmaTensorOpMultiplicandTileIteratorV2 &add_tile_offset(TensorCoord const &tile_offset) {

    int contiguous_offset = tile_offset.contiguous();
    if (Shape::kContiguous ==
        Layout::PartitionShape::kContiguous * Layout::kElementsPerAccess) {
      if (tile_offset.contiguous() % 2) {
        CUTLASS_PRAGMA_UNROLL
        for (int i = 0; i < kPointerCount / 2; ++i) {
          AccessType const *tmp_pointer = pointer_[i];
          pointer_[i] = pointer_[i + kPointerCount / 2];
          pointer_[i + kPointerCount / 2] = tmp_pointer;
        }
      }
      contiguous_offset = (tile_offset.contiguous() >> 1) << 1;
    }

    int offset = (tile_offset.strided() * InstructionShape::kStrided) *
                     stride_ * Layout::kElementsPerAccess +
                 contiguous_offset * Shape::kContiguous;

    add_pointer_offset(offset);

    return *this;
  }

  /// Advances the iterator along the advance dimension
  CUTLASS_DEVICE
  MmaTensorOpMultiplicandTileIteratorV2 & operator++() {

    add_tile_offset({0, 1});

    if (kPartitionsK > 1) {
      ++k_group_idx_;
      // Jump to next stage
      if (k_group_idx_ == Policy::kGroupsPerTile) {
        k_group_idx_ = 0;
        add_tile_offset(
            {0, ((kPartitionsK - 1) * Policy::kGroupsPerTile)});
      }
    }

    return *this;
  }

  /// Advances the iterator along the opposite of the advance dimension
  CUTLASS_HOST_DEVICE
  MmaTensorOpMultiplicandTileIteratorV2 & operator--() {
    byte_offset_ -= stride_ * InstructionShape::kStrided * sizeof(Element) *
                    Layout::kElementsPerAccess;

    return *this;
  }

  ///< advances in units of whole tiles along the logical coordinate space of the tensor
  CUTLASS_DEVICE
  MmaTensorOpMultiplicandTileIteratorV2 & operator+=(TensorCoord const &tile_offset) {
    add_tile_offset(tile_offset);
    return *this;
  }

  ///< advances in units of whole tiles along the logical coordinate space of the tensor
  CUTLASS_DEVICE
  MmaTensorOpMultiplicandTileIteratorV2 & operator-=(TensorCoord const &tile_offset) {
    add_tile_offset(-tile_offset);
    return *this;
  }

  /// Loads a fragment from memory at the location pointed to by the iterator.
  CUTLASS_HOST_DEVICE
  void load(Fragment &frag) const {

    load_with_byte_offset(frag, 0);
  }

  /// Loads a fragment from memory with additional logical offset
  CUTLASS_DEVICE
  void load_with_byte_offset(
      /// fragment to load from the tensor
      Fragment &frag,
      /// loads a tile with a linear offset in units of bytes
      Index byte_offset) const {

    Array<unsigned, Policy::LdsmShape::kCount> *fetch_ptr = 
      reinterpret_cast<Array<unsigned, Policy::LdsmShape::kCount> *>(&frag);

    CUTLASS_PRAGMA_UNROLL
    for (int s = 0; s < Policy::LdsmIterations::kStrided; ++s) {

      CUTLASS_PRAGMA_UNROLL
      for (int c = 0; c < Policy::LdsmIterations::kContiguous; ++c) {

        int access_idx = c + s * Policy::LdsmIterations::kContiguous;

        AccessType const *source_ptr =
            pointer_[c % kPointerCount] +
            Layout::TileShape::kContiguous * (c / kPointerCount) +
            Policy::LdsmShape::kStrided * s * stride_;

        char const *source_byte_ptr = reinterpret_cast<char const *>(source_ptr) + byte_offset + byte_offset_;

        cutlass::arch::ldsm<layout::ColumnMajor, Policy::LdsmShape::kCount>(
          fetch_ptr[access_idx],
          source_byte_ptr
        );
      }
    }
  }

  /// Loads a fragment from memory with additional logical offset
  CUTLASS_DEVICE
  void load_with_pointer_offset(
      /// fragment to load from the tensor
      Fragment &frag,
      /// loads a tile with a linear offset
      Index pointer_offset) const {
    load_with_byte_offset(frag, pointer_offset * sizeof(Element));
  }

  /// Loads a fragment from memory with logical offset in units of whole tiles.
  CUTLASS_DEVICE
  void load(
      /// fragment to load from the tensor
      Fragment &frag,
      /// loads a tile with a logical offset in units of whole tiles
      TensorCoord const &tile_offset) const {
    load_with_byte_offset(frag, tile_offset, 0);
  }

  /// Loads a fragment from memory with logical offset in units of whole tiles.
  CUTLASS_DEVICE
  void load(
      /// fragment to load from the tensor
      Fragment &frag,
      /// loads a tile with a logical offset in units of whole tiles
      TensorCoord const &tile_offset,
      /// loads a tile with a logical offset AND a pointer offset
      Index pointer_offset) const {
    load_with_byte_offset(frag, tile_offset, pointer_offset * sizeof(Element));
  }

  /// Loads a fragment from memory with logical offset in units of whole tiles.
  CUTLASS_DEVICE
  void load_with_byte_offset(
      /// fragment to load from the tensor
      Fragment &frag,
      /// loads a tile with a logical offset in units of whole tiles
      TensorCoord const &tile_offset,
      /// loads a tile with a logical offset AND a pointer offset
      Index byte_offset) const {
    Index pointer_offset = 
      tile_offset.contiguous() * Shape::kContiguous / Layout::kElementsPerAccess + 
      tile_offset.strided() * InstructionShape::kStrided * stride_;

    byte_offset += sizeof(AccessType) * pointer_offset;

    load_with_byte_offset(frag, byte_offset);
  }

  /// Notify the iterator which k-group it is currently pointing to.
  ///
  /// This does not advance the iterator. Rather, it overrides its internal
  /// tracking with constant-valued k-group index to enable the compiler to
  /// fold constants and achieve more efficient code.
  ///
  /// This is used by some nontrivial permuted layouts.
  CUTLASS_DEVICE
  void set_kgroup_index(int k_group) {
    // no op
  }
};

////////////////////////////////////////////////////////////////////////////////

////////////////////////////////////////////////////////////////////////////////
/// This tile iterator is specialized for 32-thread TensorOps. It uses LDSM to load from shared
/// memory and therefore must be initialized with a TensorRef to shared memory. 
///
/// Satisfies:
///   ReadableRandomAccessContiguousTileIteratorConcept
///
template <
    /// Size of the matrix to load (concept: MatrixShape)
    typename Shape_,
    /// Identifies A or B multiplicand
    Operand Operand_,
    /// Data type of elements
    typename Element_,
    /// Shape of one matrix product operation (concept: MatrixShape)
    typename InstructionShape_,
    /// Interval between adjacent *MMA instructions (in units of MMA
    /// instructions)
    int OpDelta_,
    /// Number of partitions along K dimension
    int PartitionsK_>
class MmaTensorOpMultiplicandTileIteratorV2<
    Shape_, Operand_, Element_,
    cutlass::layout::RowMajorTensorOpMultiplicandCongruous<
        sizeof_bits<Element_>::value, int(128 / sizeof(Element_))>,
    InstructionShape_, OpDelta_, 32, PartitionsK_> {
 public:

  /// Shape of tile to load (concept: PitchLinearShape)
  using Shape = Shape_;

  /// Operand tag
  static Operand const kOperand = Operand_;

  static_assert(kOperand == Operand::kB,
                "MmaTensorOpMultiplicandIterator for RowMajor Congruous may "
                "only be instantiated for B operand to warp-level Mma.");

  /// Element type
  using Element = Element_;

  /// Layout of source tile
  using Layout = cutlass::layout::RowMajorTensorOpMultiplicandCongruous<
      sizeof_bits<Element_>::value, int(128 / sizeof(Element_))>;

  /// Shape of one matrix product operation (concept: MatrixShape)
  using InstructionShape = InstructionShape_;

  /// Delta between *MMA operations (in units of *MMA operations, concept: MatrixShape)
  static int const kOpDelta = OpDelta_;

  /// Number of participating threads
  static int const kThreads = 32;

  /// TensorRef type for loading element from a tensor
  using TensorRef = TensorRef<Element, Layout>;

  /// Index type
  using Index = typename TensorRef::Index;

  /// Long Index type
  using LongIndex = typename TensorRef::LongIndex;

  /// Coordinate for an element in the tensor
  using TensorCoord = typename TensorRef::TensorCoord;

  /// Underlying tile iterator implementation
  using Base = MmaTensorOpMultiplicandTileIteratorV2<
      layout::PitchLinearShape<Shape::kColumn, Shape::kRow>, kOperand, Element,
      layout::TensorOpMultiplicandCongruous<sizeof_bits<Element_>::value,
                                            int(128 / sizeof(Element_))>,
      layout::PitchLinearShape<InstructionShape::kColumn,
                               InstructionShape::kRow>,
      kOpDelta, kThreads, PartitionsK_>;

 public:

  //
  // Derived quantities
  //

  /// Fragment object holding a thread's part of a tile
  using Fragment = typename Base::Fragment;

private:

  /// Underlying tile iterator
  Base iterator_;

public:
  
  /// Default ctor constructs null iterator
  CUTLASS_HOST_DEVICE
  MmaTensorOpMultiplicandTileIteratorV2() { }

  /// Constructor from TensorRef
  CUTLASS_HOST_DEVICE
  MmaTensorOpMultiplicandTileIteratorV2(
    TensorRef const &ref, 
    int lane_id
  ): iterator_({ref.data(), ref.stride()}, lane_id) {
  }

  /// Adds a pointer offset to internal pointer(s) to advance through memory
  CUTLASS_HOST_DEVICE
  MmaTensorOpMultiplicandTileIteratorV2 &add_pointer_offset(LongIndex offset) {

    iterator_.add_pointer_offset(offset);

    return *this;
  }

  /// Advances an iterator along logical dimensions of matrix in units of whole tiles
  CUTLASS_HOST_DEVICE
  MmaTensorOpMultiplicandTileIteratorV2 &add_tile_offset(TensorCoord const &tile_offset) {

    iterator_.add_tile_offset({tile_offset.column(), tile_offset.row()});

    return *this;
  }

  /// Advances the iterator along the advance dimension
  CUTLASS_HOST_DEVICE
  MmaTensorOpMultiplicandTileIteratorV2 & operator++() {

    ++iterator_;

    return *this;
  }

  /// Advances the iterator along the advance dimension
  CUTLASS_HOST_DEVICE
  MmaTensorOpMultiplicandTileIteratorV2 & operator--() {

    --iterator_;

    return *this;
  }

  ///< advances in units of whole tiles along the logical coordinate space of the tensor
  CUTLASS_DEVICE
  MmaTensorOpMultiplicandTileIteratorV2 & operator+=(TensorCoord const &tile_offset) {
    add_tile_offset(PitchLinearCoord(tile_offset.column(), tile_offset.row()));
    return *this;
  }

  ///< advances in units of whole tiles along the logical coordinate space of the tensor
  CUTLASS_DEVICE
  MmaTensorOpMultiplicandTileIteratorV2 & operator-=(TensorCoord const &tile_offset) {
    add_tile_offset(-PitchLinearCoord(tile_offset.column(), tile_offset.row()));
    return *this;
  }

  /// Loads a fragment from memory at the location pointed to by the iterator.
  CUTLASS_HOST_DEVICE
  void load(Fragment &frag) const {

    iterator_.load(frag);
  }

  /// Loads a fragment from memory with additional logical offset
  CUTLASS_DEVICE
  void load_with_pointer_offset(
      /// fragment to load from the tensor
      Fragment &frag,
      /// loads a tile with a linear offset
      Index pointer_offset) const {
    iterator_.load_with_pointer_offset(frag, pointer_offset);
  }

  /// Loads a fragment from memory with additional logical offset
  CUTLASS_DEVICE
  void load_with_byte_offset(
      /// fragment to load from the tensor
      Fragment &frag,
      /// loads a tile with a linear offset
      Index byte_offset) const {
    iterator_.load_with_byte_offset(frag, byte_offset);
  }

  /// Loads a fragment from memory with logical offset in units of whole tiles.
  CUTLASS_DEVICE
  void load(
      /// fragment to load from the tensor
      Fragment &frag,
      /// loads a tile with a logical offset in units of whole tiles
      TensorCoord const &tile_offset) const {
    // TODO
  }

  /// Loads a fragment from memory with logical offset in units of whole tiles.
  CUTLASS_DEVICE
  void load(
      /// fragment to load from the tensor
      Fragment &frag,
      /// loads a tile with a logical offset in units of whole tiles
      TensorCoord const &tile_offset,
      /// loads a tile with a logical offset AND a pointer offset
      Index pointer_offset) const {
    // TODO
  }

  /// Loads a fragment from memory with logical offset in units of whole tiles.
  CUTLASS_DEVICE
  void load_with_byte_offset(
      /// fragment to load from the tensor
      Fragment &frag,
      /// loads a tile with a logical offset in units of whole tiles
      TensorCoord const &tile_offset,
      /// loads a tile with a logical offset AND a pointer offset
      Index byte_offset) const {
    iterator_.load_with_byte_offset(
      frag,
      {tile_offset.strided(), tile_offset.contiguous()},
      byte_offset);
  }

  /// Notify the iterator which k-group it is currently pointing to.
  ///
  /// This does not advance the iterator. Rather, it overrides its internal
  /// tracking with constant-valued k-group index to enable the compiler to
  /// fold constants and achieve more efficient code.
  ///
  /// This is used by some nontrivial permuted layouts.
  CUTLASS_DEVICE
  void set_kgroup_index(int k_group) {
    iterator_.set_kgroup_index(k_group); 
  }
};


////////////////////////////////////////////////////////////////////////////////

} // namespace warp
} // namespace gemm
} // namespace cutlass

////////////////////////////////////////////////////////////////////////////////