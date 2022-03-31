#include "helper.h"
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
class MmaTensorOpMultiplicandTileIteratorTrans;


////////////////////////////////////////////////////////////////////////////////

/// This tile iterator is specialized for 32-thread TensorOps. It uses LDSM to
/// load from shared memory and therefore must be initialized with a TensorRef
/// to shared memory.
///
/// Satisfies:
///   ReadableRandomAccessContiguousTileIteratorConcept
///
template <
    /// Size of the matrix to load (concept: PitchLinearShape)
    typename Shape_,  // <32, 64>
    /// Identifies A or B multiplicand
    Operand Operand_,
    /// Data type of elements
    typename Element_,
    /// Shape of one matrix product operation (concept: PitchLinearShape)
    typename InstructionShape_,  // <16, 16>
    /// Interval between adjacent *MMA instructions (in units of MMA
    /// instructions)
    int OpDelta_,
    /// Element number when the layout crosses (in units of elements)
    int Crosswise,
    /// Number of partitions along K dimension
    int PartitionsK_>
class MmaTensorOpMultiplicandTileIteratorTrans<
    Shape_, Operand_, Element_,
    cutlass::layout::TensorOpMultiplicandCrosswise<sizeof_bits<Element_>::value,
                                                   Crosswise>,
    InstructionShape_, OpDelta_, 32, PartitionsK_> {
 public:
  /// Shape of tile to load (concept: PitchLinearShape)
  using Shape = Shape_;

  /// Operand tag
  static Operand const kOperand = Operand_;

  static_assert(kOperand == Operand::kA || kOperand == Operand::kB,
                "MmaTensorOpMultiplicandIterator may only be instantiated for "
                "A or B operands to warp-level Mma.");

  /// Element type
  using Element = Element_;

  /// Element number when the layout crosses
  static int const kCrosswise = Crosswise;

  /// Layout of source tile
  using Layout = cutlass::layout::TensorOpMultiplicandCrosswise<
      sizeof_bits<Element_>::value, kCrosswise>;

  /// Shape of one matrix product operation (concept: GemmShape)
  using InstructionShape = InstructionShape_;

  /// Delta between *MMA operations (in units of *MMA operations, concept:
  /// MatrixShape)
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
                  "Shape of warp-level mma must be divisible by LDSM's "
                  "fundamental tile size.");

    static_assert(!(Shape::kStrided % kLdsmOpInner),
                  "Shape of warp-level mma must be divisible by LDSM's "
                  "fundamental tile size.");

    /// Shape of one individual LDSM instruction
    static int const LdsmShapeContiguous =
        InstructionShape::kContiguous / kLdsmOpOuter;
    static int const LdsmShapeStrided =
        ((4 / LdsmShapeContiguous * kLdsmOpInner) > Shape::kStrided)
            ? (Shape::kStrided / kLdsmOpInner)
            : (4 / LdsmShapeContiguous);
    using LdsmShape =
        layout::PitchLinearShape<LdsmShapeContiguous, LdsmShapeStrided>;

    /// Number and arrangement of LDSM instructions
    // using LdsmIterations =
    //     layout::PitchLinearShape<1, Shape::kStrided / kLdsmOpInner /
    //                                     LdsmShape::kStrided>;
    // TODO: formalize this part of code
    using LdsmIterations =
        layout::PitchLinearShape<2, 2>;

    ///
    static int const kGroupsPerTile = Layout::TileShape::kContiguous /
                                      Layout::kFactor / LdsmShape::kContiguous;
  };

 private:
  /// Not working on this feature at the moment.
  static_assert(kOpDelta == 1,
                "Alternative arrangements not supported at present.");

  /// Pointer type used for accesses
  using AccessType = Array<Element, Layout::kElementsPerAccess>;

 public:
  //
  // Derived quantities
  //

  /// Fragment object holding a thread's part of a tile
  using Fragment = Array<Element, Shape::kStrided *
                                      InstructionShape::kContiguous / kThreads>;

 private:

  /// Total number of sections.  The memory is divided into stages.  One stage
  /// can store one tile.  Stage is divided into sections.  Interleaved layout
  /// can have multiple sections in a stage.  The rest layout only has one section
  /// in a stage.
  int sections_;

  /// Layout object storing stride values
  StrideIndex stride_;

  /// Shared memory base pointers - not advanced
  AccessType const *pointer_;

  /// Byte offset incremented as iterator advances
  Index byte_offset_;

  /// Internal counter used to determine when to increment byte offset and when
  /// to XOR it
  int k_group_idx_;

 public:
  /// Default ctor constructs null iterator
  CUTLASS_HOST_DEVICE
  MmaTensorOpMultiplicandTileIteratorTrans()
      : pointer_(nullptr),
        sections_(0),
        stride_(0),
        byte_offset_(0),
        k_group_idx_(0) {}

  /// Constructor from TensorRef
  CUTLASS_DEVICE
  MmaTensorOpMultiplicandTileIteratorTrans(TensorRef const &ref, int lane_id)
      : pointer_(reinterpret_cast<AccessType const *>(ref.data())),
        sections_(ref.stride(0) / kCrosswise),
        // stride_ = kCrosswise x sections_ x kFactor
        stride_(ref.stride(0) * Layout::kFactor / Layout::kElementsPerAccess),
        byte_offset_(0),
        k_group_idx_(0) {
    // Warp level iterator at most use double buffer to hide latency.  If there
    // are more than 2 sections, every stage should have more than 1 section.

    // Turing silicon requires all 32 threads in a warp provide valid addresses
    // even for LDSM.1 and LDSM.2
#if (defined(__CUDA_ARCH__) && (__CUDA_ARCH__ == 750))
    lane_id = lane_id % (Policy::LdsmShape::kCount * Policy::kLdsmOpInner);
#endif

    int quad_quad = (lane_id >> 4);
    int quad_pair = (lane_id >> 3);
    int lane_in_pair = (lane_id & 1);
    int lane_in_quad = (lane_id & 3);
    int lane_in_quad_pair = (lane_id & 7);
    int lane_in_quad_quad = (lane_id & 15);

    int partition_contiguous_idx = -1;
    int access_contiguous_idx = -1;
    int access_strided_idx = -1;

    if (Layout::kFactor == 4) {
      // Super Integer matrix multiply Interleaved-32

      int factor_in_partition =
          (Layout::PartitionShape::kContiguous * Layout::kFactor /
           Layout::TileShape::kContiguous);

      if (Policy::LdsmShape::kStrided == Policy::LdsmShape::kCount) {
        // Integer matrix multiply 8816  A/B
        partition_contiguous_idx = lane_in_quad / factor_in_partition;
        access_contiguous_idx = ((lane_in_pair * factor_in_partition) ^
                                 (lane_in_quad_quad / Layout::kFactor));
        access_strided_idx = lane_id / Layout::kFactor;
      }
      else if (Policy::LdsmShape::kStrided ==
                     (Policy::LdsmShape::kCount / 2) &&
                 kOperand == Operand::kA) {
        // Integer matrix multiply 16832 A
        partition_contiguous_idx = lane_in_quad / factor_in_partition;
        access_strided_idx = lane_in_quad_quad / Layout::kFactor;
        access_contiguous_idx =
            ((lane_in_pair * factor_in_partition + quad_quad) ^
             access_strided_idx);
      }
      else if (Policy::LdsmShape::kStrided ==
                     (Policy::LdsmShape::kCount / 2) &&
                 kOperand == Operand::kB) {
        // Integer matrix multiply 16832 B
        partition_contiguous_idx = lane_in_quad / factor_in_partition;
        access_strided_idx = lane_in_quad_pair / Layout::kFactor + quad_quad * 2;
        access_contiguous_idx =
            ((lane_in_pair * factor_in_partition + ((lane_id & 8) >> 3)) ^
             access_strided_idx);
      }
    } else if (Layout::kFactor == 2) {
      // Super Matrix multiply kBlock = 32
      if (Policy::LdsmShape::kStrided == Policy::LdsmShape::kCount) {
        // Matrix multiply 1688 A/B
        // (Q stands for 1 8x128bit block).
        // Q0
        // Q1
        // Q2
        // Q3
        // Four blocks are next to each other in the strided dimension.
        partition_contiguous_idx = (lane_id % Layout::kFactor);
        access_contiguous_idx = (lane_in_quad_pair / Layout::kFactor);
        access_strided_idx = lane_id / Layout::kFactor;
      }
      else if (Policy::LdsmShape::kStrided ==
                     (Policy::LdsmShape::kCount / 2) &&
                 kOperand == Operand::kA) {
        // Matrix multiply 16816|1688.TF32 A
        // Q0 Q2
        // Q1 Q3
        partition_contiguous_idx = (lane_id % Layout::kFactor);
        access_contiguous_idx =
            (quad_quad ^ (lane_in_quad_pair / Layout::kFactor));
        access_strided_idx = (lane_in_quad_quad / Layout::kFactor);
      } else if (Policy::LdsmShape::kStrided ==
                     (Policy::LdsmShape::kCount / 2) &&
                 kOperand == Operand::kB) {
        // Matrix multiply 16816|1688.TF32 B
        // Q0 Q1
        // Q2 Q3
        partition_contiguous_idx = (lane_id % Layout::kFactor);
        access_contiguous_idx =
            ((quad_pair & 1) ^ (lane_in_quad_pair / Layout::kFactor));
        access_strided_idx =
            (lane_in_quad_pair + (lane_id >> 4 << 3)) / Layout::kFactor;
      } 
      else if (Policy::LdsmShape::kContiguous == Policy::LdsmShape::kCount) {
        // Matrix multiply 16832.SP B
        // Q0 Q1 Q2 Q3
        partition_contiguous_idx = (lane_id % Layout::kFactor);
        access_contiguous_idx =
            (quad_pair ^ (lane_in_quad_pair / Layout::kFactor));
        access_strided_idx = lane_in_quad_pair / Layout::kFactor;
      }
    } else if (Layout::kFactor == 1) {
      // Super Matrix multiply kBlock = 64
      if (Policy::LdsmShape::kStrided == Policy::LdsmShape::kCount) {
        // Q0
        // Q1
        // Q2
        // Q3
        partition_contiguous_idx = (lane_in_quad_pair >> 2);
        access_contiguous_idx = lane_in_quad;
        access_strided_idx = lane_id;
      }
      else if (Policy::LdsmShape::kStrided ==
                     (Policy::LdsmShape::kCount / 2) &&
                 kOperand == Operand::kA) {
        // Matrix multiply 16816|1688.TF32 A
        // Q0 Q2
        // Q1 Q3
        partition_contiguous_idx = (lane_in_quad_pair >> 2);
        access_contiguous_idx = (quad_quad ^ lane_in_quad);
        access_strided_idx = lane_in_quad_quad;
      } else if (Policy::LdsmShape::kStrided ==
                     (Policy::LdsmShape::kCount / 2) &&
                 kOperand == Operand::kB) {
        // Matrix multiply 16816|1688.TF32 B
        // Q0 Q1
        // Q2 Q3
        partition_contiguous_idx = (lane_in_quad_pair >> 2);
        access_contiguous_idx = ((quad_pair & 1) ^ lane_in_quad);
        access_strided_idx = lane_in_quad_pair + (lane_id >> 4 << 3);
      } 
      else if (Policy::LdsmShape::kContiguous == Policy::LdsmShape::kCount) {
        // Matrix multiply 16832.SP B
        // Q0 Q1 Q2 Q3
        partition_contiguous_idx = (lane_in_quad_pair >> 2);
        access_contiguous_idx = (quad_pair ^ lane_in_quad);
        access_strided_idx = lane_in_quad_pair;
      }
    }

    int access_contiguous =
        partition_contiguous_idx * Layout::PartitionShape::kContiguous +
        access_contiguous_idx;

    int access_strided = access_strided_idx;

    byte_offset_ = (access_contiguous + access_strided * stride_) *
                   sizeof_bits<Element>::value * Layout::kElementsPerAccess / 8;
  }

  /// Adds a pointer offset to internal pointer(s) to advance through memory
  CUTLASS_DEVICE
  MmaTensorOpMultiplicandTileIteratorTrans &add_pointer_offset(LongIndex offset) {
    byte_offset_ += offset * sizeof_bits<Element>::value / 8;

    return *this;
  }

  /// Advances an iterator along logical dimensions of matrix in units of whole
  /// tiles
  CUTLASS_DEVICE
  MmaTensorOpMultiplicandTileIteratorTrans &add_tile_offset(
      TensorCoord const &tile_offset) {
    int whole_tiles = tile_offset.contiguous() / Policy::kGroupsPerTile;
    int k_groups_delta = tile_offset.contiguous() % Policy::kGroupsPerTile;

    byte_offset_ ^= k_groups_delta * sizeof_bits<Element>::value *
                    Layout::kElementsPerAccess *
                    Policy::LdsmShape::kContiguous / 8;
    pointer_ +=
        tile_offset.strided() * stride_ * Shape::kStrided / Layout::kFactor +
        whole_tiles * stride_ / sections_;
    return *this;
  }

  /// Advances an iterator along logical dimensions of matrix in units of whole
  /// tiles
  CUTLASS_DEVICE
  MmaTensorOpMultiplicandTileIteratorTrans &add_tile_offset_negative(
      TensorCoord const &tile_offset) {

    int whole_tiles = tile_offset.contiguous() / Policy::kGroupsPerTile;
    int k_groups_delta = tile_offset.contiguous() % Policy::kGroupsPerTile;
    if (k_groups_delta < 0) {
        whole_tiles -= 1;
        k_groups_delta += Policy::kGroupsPerTile;
    }

    if ((Policy::kGroupsPerTile / kPartitionsK) >= 2) {
      byte_offset_ ^= (k_groups_delta & 1) * Policy::LdsmShape::kContiguous *
                        sizeof_bits<Element>::value *
                        Layout::kElementsPerAccess / 8;
    }
    if ((Policy::kGroupsPerTile / kPartitionsK) >= 4) {
      byte_offset_ ^= ((k_groups_delta + (k_group_idx_ & 1)) & 2) * 
                        Policy::LdsmShape::kContiguous *
                        sizeof_bits<Element>::value *
                        Layout::kElementsPerAccess / 8;
    }
    if ((Policy::kGroupsPerTile / kPartitionsK) == 8) {
      byte_offset_ ^= ((k_groups_delta + (k_group_idx_ & 3)) & 4) * 
                        Policy::LdsmShape::kContiguous *
                        sizeof_bits<Element>::value *
                        Layout::kElementsPerAccess / 8;
    }

    k_group_idx_ += k_groups_delta;
    whole_tiles += k_group_idx_ / (Policy::kGroupsPerTile / kPartitionsK);
    k_group_idx_ = k_group_idx_ % (Policy::kGroupsPerTile / kPartitionsK);

    pointer_ +=
        tile_offset.strided() * stride_ * Shape::kStrided / Layout::kFactor +
        whole_tiles * stride_ / sections_;
    return *this;
  }

  /// Advances the iterator along the advance dimension
  CUTLASS_DEVICE
  MmaTensorOpMultiplicandTileIteratorTrans &operator++() {

    // Integer matrix multiply 16832 Interleaved-32
    //   NONE
    // Integer matrix multiply 16816 Interleaved-32 || Integer matrix multiply 16816 kblock=32

    // Integer matrix multiply 8816  Interleaved-32
    //   ^1 ^1
    // Matrix multiply 1684.TF32 kblock=16 || Integer matrix multiply 16816 kblock=64
    // Matrix multiply 1688 kblock=32 || Integer matrix multiply 8816 kblock=64
    //   ^1 ^3 ^1 ^3
    // Matrix multiply 1688 kblock=64
    //   ^1 ^3 ^1 ^7 ^1 ^3 ^1 ^7

    // Matrix multiply 16816 kblock=32 | 1688.TF32 kblock=16 || Integer matrix multiply 16832 kblock=64
    //   ^2 ^2
    // Matrix multiply 16816 kblock=64 | 1688.TF32 kblock=32 || Integer matrix multiply 16832 kblock=128
    //   ^2 ^6 ^2 ^6

    // if ((Policy::kGroupsPerTile / kPartitionsK) > 1) {
    //   int mask = ((Policy::kGroupsPerTile / kPartitionsK) == 8)
    //                  ? 3
    //                  : (((Policy::kGroupsPerTile / kPartitionsK) == 4) ? 1 : 0);

    //   if (((k_group_idx_ & mask) % 2) == 0)
    //     byte_offset_ ^= 1 * Policy::LdsmShape::kContiguous *
    //                     sizeof_bits<Element>::value *
    //                     Layout::kElementsPerAccess / 8;
    //   else if ((k_group_idx_ & mask) == 1)
    //     byte_offset_ ^= 3 * Policy::LdsmShape::kContiguous *
    //                     sizeof_bits<Element>::value *
    //                     Layout::kElementsPerAccess / 8;
    //   else if ((k_group_idx_ & mask) == 3)
    //     byte_offset_ ^= 7 * Policy::LdsmShape::kContiguous *
    //                     sizeof_bits<Element>::value *
    //                     Layout::kElementsPerAccess / 8;
    // }

    // k_group_idx_++;

    // if (k_group_idx_ == (Policy::kGroupsPerTile / kPartitionsK)) {
    //   k_group_idx_ = 0;
    //   add_tile_offset({Policy::kGroupsPerTile, 0});
    // }

 
    pointer_ += Policy::kLdsmOpInner / Layout::kFactor *
                Policy::LdsmShape::kStrided * stride_ * 2;

    return *this;
  }

  /// Advances the iterator along the advance dimension
  CUTLASS_HOST_DEVICE
  MmaTensorOpMultiplicandTileIteratorTrans &operator--() { assert(0); }

  ///< advances in units of whole tiles along the logical coordinate space of
  ///< the tensor
  CUTLASS_DEVICE
  MmaTensorOpMultiplicandTileIteratorTrans &operator+=(
      TensorCoord const &tile_offset) {
    add_tile_offset(tile_offset);
    return *this;
  }

  ///< advances in units of whole tiles along the logical coordinate space of
  ///< the tensor
  CUTLASS_DEVICE
  MmaTensorOpMultiplicandTileIteratorTrans &operator-=(
      TensorCoord const &tile_offset) {
    add_tile_offset(-tile_offset);
    return *this;
  }

  /// Loads a fragment from memory at the location pointed to by the iterator.
  CUTLASS_HOST_DEVICE
  void load(Fragment &frag) const { load_with_byte_offset(frag, 0); }

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
      for (int c = 0; c < Policy::LdsmIterations::kContiguous; ++c){
        int access_idx = c * Policy::LdsmIterations::kStrided + s;

        AccessType const *source_ptr = 
            pointer_ + Policy::kLdsmOpInner / Layout::kFactor *
                Policy::LdsmShape::kStrided * s * stride_;
        Index byte_offset_t = (byte_offset + byte_offset_) ^ (c * Policy::LdsmShape::kContiguous *
                                                              sizeof_bits<Element>::value *
                                                              Layout::kElementsPerAccess / 8);
        char const *source_byte_ptr =
            reinterpret_cast<char const *>(source_ptr) + byte_offset_t;
        cutlass::arch::ldsm<layout::RowMajor, Policy::LdsmShape::kCount>(
            fetch_ptr[access_idx], source_byte_ptr);
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
    Index pointer_offset = tile_offset.contiguous() *
                               InstructionShape::kContiguous /
                               Layout::kElementsPerAccess +
                           tile_offset.strided() * Shape::kStrided * stride_;

    byte_offset += sizeof_bits<AccessType>::value * pointer_offset / 8;

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
    k_group_idx_ = k_group % (Policy::kGroupsPerTile / kPartitionsK);
  }
};

////////////////////////////////////////////////////////////////////////////////////////////

/// This tile iterator is specialized for 32-thread TensorOps. It uses LDSM to
/// load from shared memory and therefore must be initialized with a TensorRef
/// to shared memory.
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
    /// Element number when the layout crosses (in units of elements)
    int Crosswise,
    /// Number of partitions along K dimension
    int PartitionsK_>
class MmaTensorOpMultiplicandTileIteratorTrans<
    Shape_, Operand_, Element_,
    cutlass::layout::RowMajorTensorOpMultiplicandCrosswise<
        sizeof_bits<Element_>::value, Crosswise>,
    InstructionShape_, OpDelta_, 32, PartitionsK_> {
 public:
  /// Shape of tile to load (concept: PitchLinearShape)
  using Shape = Shape_;

  /// Operand tag
  static Operand const kOperand = Operand_;

  static_assert(kOperand == Operand::kA,
                "MmaTensorOpMultiplicandIterator for RowMajor Crosswise may "
                "only be instantiated for A operand to warp-level Mma.");

  /// Element type
  using Element = Element_;

  /// Element number when the layout crosses
  static int const kCrosswise = Crosswise;

  /// Layout of source tile
  using Layout = cutlass::layout::RowMajorTensorOpMultiplicandCrosswise<
      sizeof_bits<Element_>::value, kCrosswise>;

  /// Shape of one matrix product operation (concept: MatrixShape)
  using InstructionShape = InstructionShape_;

  /// Delta between *MMA operations (in units of *MMA operations, concept:
  /// MatrixShape)
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
  using Base = MmaTensorOpMultiplicandTileIteratorTrans<
      // <32, 64>
      layout::PitchLinearShape<Shape::kColumn, Shape::kRow>, kOperand, Element,
      layout::TensorOpMultiplicandCrosswise<sizeof_bits<Element_>::value,
                                            kCrosswise>,
      // <16, 16>
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
  MmaTensorOpMultiplicandTileIteratorTrans() {}

  /// Constructor from TensorRef
  CUTLASS_HOST_DEVICE
  MmaTensorOpMultiplicandTileIteratorTrans(TensorRef const &ref, int lane_id)
      : iterator_({ref.data(), ref.stride()}, lane_id) {}

  /// Adds a pointer offset to internal pointer(s) to advance through memory
  CUTLASS_HOST_DEVICE
  MmaTensorOpMultiplicandTileIteratorTrans &add_pointer_offset(LongIndex offset) {
    iterator_.add_pointer_offset(offset);

    return *this;
  }

  /// Advances an iterator along logical dimensions of matrix in units of whole
  /// tiles
  CUTLASS_HOST_DEVICE
  MmaTensorOpMultiplicandTileIteratorTrans &add_tile_offset(
      TensorCoord const &tile_offset) {
    iterator_.add_tile_offset({tile_offset.column(), tile_offset.row()});

    return *this;
  }

  /// Advances an iterator along logical dimensions of matrix in units of whole
  /// tiles
  CUTLASS_DEVICE
  MmaTensorOpMultiplicandTileIteratorTrans &add_tile_offset_negative(
      TensorCoord const &tile_offset) {
    iterator_.add_tile_offset_negative({tile_offset.column(), tile_offset.row()});

    return *this;
  }

  /// Advances the iterator along the advance dimension
  CUTLASS_HOST_DEVICE
  MmaTensorOpMultiplicandTileIteratorTrans &operator++() {
    ++iterator_;

    return *this;
  }

  /// Advances the iterator along the advance dimension
  CUTLASS_HOST_DEVICE
  MmaTensorOpMultiplicandTileIteratorTrans &operator--() {
    --iterator_;

    return *this;
  }

  ///< advances in units of whole tiles along the logical coordinate space of
  ///< the tensor
  CUTLASS_DEVICE
  MmaTensorOpMultiplicandTileIteratorTrans &operator+=(
      TensorCoord const &tile_offset) {
    add_tile_offset(PitchLinearCoord(tile_offset.column(), tile_offset.row()));
    return *this;
  }

  ///< advances in units of whole tiles along the logical coordinate space of
  ///< the tensor
  CUTLASS_DEVICE
  MmaTensorOpMultiplicandTileIteratorTrans &operator-=(
      TensorCoord const &tile_offset) {
    add_tile_offset(-PitchLinearCoord(tile_offset.column(), tile_offset.row()));
    return *this;
  }

  /// Loads a fragment from memory at the location pointed to by the iterator.
  CUTLASS_HOST_DEVICE
  void load(Fragment &frag) const { iterator_.load(frag); }

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
    assert(0);
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
    assert(0);
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
        frag, {tile_offset.strided(), tile_offset.contiguous()}, byte_offset);
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