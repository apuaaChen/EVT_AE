#include "cuda_bf16.h"
/////////////////////////////////////////////////////////////////////////////////////////////////

namespace cutlass {
namespace arch {

/////////////////////////////////////////////////////////////////////////////////////////////////

/// Matrix multiply-add operation
template <
  /// Size of the matrix product (concept: GemmShape)
  typename Shape_,
  /// Number of threads participating
  int kThreads_,
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
  /// Inner product operator
  typename Operator
>
struct MmaV2;

////////////////////////////////////////////////////////////////////////////////

/// Matrix multiply-add operation: F32 = bf16 * bf16 + F32
template <>
struct MmaV2<
  gemm::GemmShape<16, 16, 16>,
  32,
  bfloat16_t,
  layout::RowMajor,
  bfloat16_t,
  layout::ColumnMajor,
  float,
  layout::RowMajor,
  OpMultiplyAdd> {

  using Shape = gemm::GemmShape<16, 16, 16>;

  using ElementA = bfloat16_t;
  using LayoutA = layout::RowMajor;
  using FragmentA = Array<bfloat16_t, 8>;

  using ElementB = bfloat16_t;
  using LayoutB = layout::ColumnMajor;
  using FragmentB = Array<bfloat16_t, 4>;

  using ElementC = float;
  using LayoutC = layout::RowMajor;
  using FragmentC = Array<float, 4>;

  using Operator = OpMultiplyAdd;
  using ArchTag = arch::Sm80;

  /// Computes multiply-add
  CUTLASS_HOST_DEVICE
  void operator()(
    FragmentC &d1,
    FragmentC &d2,
    FragmentA const &a,
    FragmentB const &b,
    FragmentC const &c1,
    FragmentC const &c2
  ) const {

#if defined(CUTLASS_ARCH_MMA_SM80_ENABLED)

    uint32_t const *A = reinterpret_cast<uint32_t const *>(&a);
    uint32_t const *B = reinterpret_cast<uint32_t const *>(&b);
    float const *C1 = reinterpret_cast<float const *>(&c1);
    float *D1 = reinterpret_cast<float *>(&d1);
    float const *C2 = reinterpret_cast<float const *>(&c2);
    float *D2 = reinterpret_cast<float *>(&d2);

    // Explicitly transpose the matrix B with warp shuffle
    __nv_bfloat16 tmp_share;
    uint32_t tmp_share_b;
    __nv_bfloat16 const *B16 = reinterpret_cast<const __nv_bfloat16 *>(&b);
    uint32_t BT[4];
    __nv_bfloat16 *BT16 = reinterpret_cast<__nv_bfloat16 *>(&BT);
    int lane_idx = threadIdx.x % 32;
    int quad_idx = lane_idx / 4;

    int target_lane = 0;
    if (lane_idx == 8 || lane_idx == 12 || lane_idx == 17 || lane_idx == 21 || lane_idx == 26 || lane_idx == 30) target_lane = lane_idx - 7;
    else if (lane_idx == 1 || lane_idx == 5 || lane_idx == 10 || lane_idx == 14 || lane_idx == 19 || lane_idx == 23) target_lane = lane_idx + 7;
    else if (lane_idx == 16 || lane_idx == 20 || lane_idx == 25 || lane_idx == 29) target_lane = lane_idx - 14;
    else if (lane_idx == 2 || lane_idx == 6 || lane_idx == 11 || lane_idx == 15) target_lane = lane_idx + 14;
    else if (lane_idx == 24 || lane_idx == 28) target_lane = lane_idx - 21;
    else if (lane_idx == 3 || lane_idx == 7) target_lane = lane_idx + 21;
    else target_lane = lane_idx;

    #pragma unroll
    for (int i = 0; i < 4; i++){
        if (quad_idx % 2 == 0){
            tmp_share = B16[1 + i * 2];
        } else {
            tmp_share = B16[0 + i * 2];
        }
        tmp_share = __shfl_xor_sync(0xffffffff, tmp_share, 4);
        if (quad_idx % 2 == 0){
            BT16[0 + i * 2] = B16[0 + i * 2];
            BT16[1 + i * 2] = tmp_share;
        } else {
            BT16[0 + i * 2] = tmp_share;
            BT16[1 + i * 2] = B16[1 + i * 2];
        }
        tmp_share_b = __shfl_sync(0xffffffff, BT[i], target_lane);
        BT[i] = tmp_share_b;
    }

    asm volatile(
        "mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 "
        "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%10,%11,%12,%13};\n"
        : "=f"(D1[0]), "=f"(D1[1]), "=f"(D1[2]), "=f"(D1[3])
        : "r"(A[0]), "r"(A[1]), "r"(A[2]), "r"(A[3]), "r"(BT[0]), "r"(BT[1]),
          "f"(C1[0]), "f"(C1[1]), "f"(C1[2]), "f"(C1[3]));
    
    asm volatile(
        "mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 "
        "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%10,%11,%12,%13};\n"
        : "=f"(D2[0]), "=f"(D2[1]), "=f"(D2[2]), "=f"(D2[3])
        : "r"(A[0]), "r"(A[1]), "r"(A[2]), "r"(A[3]), "r"(BT[2]), "r"(BT[3]),
          "f"(C2[0]), "f"(C2[1]), "f"(C2[2]), "f"(C2[3]));

#else

    CUTLASS_UNUSED(d1);
    CUTLASS_UNUSED(d2);
    CUTLASS_UNUSED(a);
    CUTLASS_UNUSED(b);
    CUTLASS_UNUSED(c1);
    CUTLASS_UNUSED(c2);
    CUTLASS_NOT_IMPLEMENTED();

#endif
  }
};

} // namespace arch
} // namespace cutlass