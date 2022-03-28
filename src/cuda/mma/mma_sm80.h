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

    asm volatile(
        "mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 "
        "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%10,%11,%12,%13};\n"
        : "=f"(D1[0]), "=f"(D1[1]), "=f"(D1[2]), "=f"(D1[3])
        : "r"(A[0]), "r"(A[1]), "r"(A[2]), "r"(A[3]), "r"(B[0]), "r"(B[1]),
          "f"(C1[0]), "f"(C1[1]), "f"(C1[2]), "f"(C1[3]));
    
    asm volatile(
        "mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 "
        "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%10,%11,%12,%13};\n"
        : "=f"(D2[0]), "=f"(D2[1]), "=f"(D2[2]), "=f"(D2[3])
        : "r"(A[0]), "r"(A[1]), "r"(A[2]), "r"(A[3]), "r"(B[2]), "r"(B[3]),
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