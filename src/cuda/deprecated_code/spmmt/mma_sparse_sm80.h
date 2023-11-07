#pragma once
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
  typename Operator,
  // Dense
  bool Dense,
  /// Specifies meta data format
  SPFormatType::Kind SPFormat = SPFormatType::Thread
>
struct SparseMmaV2;


template <>
struct SparseMmaV2<gemm::GemmShape<16, 8, 32>, 32, bfloat16_t, layout::RowMajor,
           bfloat16_t, layout::ColumnMajor, float, layout::RowMajor,
           OpMultiplyAdd, false, SPFormatType::Thread> {
  using Shape = gemm::GemmShape<16, 8, 32>;

  using ElementA = bfloat16_t;
  using LayoutA = layout::RowMajor;
  using FragmentA = Array<bfloat16_t, 8>;

  using ElementB = bfloat16_t;
  using LayoutB = layout::ColumnMajor;
  using FragmentB = Array<bfloat16_t, 8>;

  using ElementC = float;
  using LayoutC = layout::RowMajor;
  using FragmentC = Array<float, 4>;

  using FragmentE = uint32_t;

  using Operator = OpMultiplyAdd;
  using ArchTag = arch::Sm80;

  static int const kSparse = 2;

  static int const kMetaSizeInBits = 2;

  static int const kMaxID2 = 2;

  CUTLASS_HOST_DEVICE
  void operator()(FragmentC &d, FragmentA const &a, FragmentB const &b,
                  FragmentC const &c, uint32_t const &E, int const id2) const {

    uint32_t const *A = reinterpret_cast<uint32_t const *>(&a);
    uint32_t const *B = reinterpret_cast<uint32_t const *>(&b);
    float const *C = reinterpret_cast<float const *>(&c);
    float *D = reinterpret_cast<float *>(&d);

    if (id2 == 0) {
    asm volatile(
        "mma.sp.sync.aligned.m16n8k32.row.col.f32.bf16.bf16.f32 "
        "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9,%10,%11}, {%12,%13,%14,%15}, %16, 0x0;\n"
        : "=f"(D[0]), "=f"(D[1]), "=f"(D[2]), "=f"(D[3])
        : "r"(A[0]), "r"(A[1]), "r"(A[2]), "r"(A[3]), "r"(B[0]), "r"(B[1]), "r"(B[2]), "r"(B[3]), 
          "f"(C[0]), "f"(C[1]), "f"(C[2]), "f"(C[3]), "r"(E));
    } else if (id2 == 1) {
    asm volatile(
        "mma.sp.sync.aligned.m16n8k32.row.col.f32.bf16.bf16.f32 "
        "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9,%10,%11}, {%12,%13,%14,%15}, %16, 0x1;\n"
        : "=f"(D[0]), "=f"(D[1]), "=f"(D[2]), "=f"(D[3])
        : "r"(A[0]), "r"(A[1]), "r"(A[2]), "r"(A[3]), "r"(B[0]), "r"(B[1]), "r"(B[2]), "r"(B[3]), 
          "f"(C[0]), "f"(C[1]), "f"(C[2]), "f"(C[3]), "r"(E));
    } else {
    assert(0);
    }
  }
};


template <>
struct SparseMmaV2<gemm::GemmShape<16, 8, 32>, 32, bfloat16_t, layout::RowMajor,
           bfloat16_t, layout::ColumnMajor, float, layout::RowMajor,
           OpMultiplyAdd, true, SPFormatType::Thread> {
  using Shape = gemm::GemmShape<16, 8, 32>;

  using ElementA = bfloat16_t;
  using LayoutA = layout::RowMajor;
  using FragmentA = Array<bfloat16_t, 8>;

  using FragmentAT = Array<bfloat16_t, 16>;

  using ElementB = bfloat16_t;
  using LayoutB = layout::ColumnMajor;
  using FragmentB = Array<bfloat16_t, 8>;

  using ElementC = float;
  using LayoutC = layout::RowMajor;
  using FragmentC = Array<float, 4>;

  using FragmentE = uint32_t;

  using Operator = OpMultiplyAdd;
  using ArchTag = arch::Sm80;

  static int const kSparse = 2;

  static int const kMetaSizeInBits = 2;

  static int const kMaxID2 = 2;

  CUTLASS_DEVICE
  void transpose(FragmentAT &da, FragmentB const &i) const{
    // Transpose the matrix with wmma
    uint32_t *A = reinterpret_cast<uint32_t *>(&da);
    uint32_t const *I = reinterpret_cast<uint32_t const *>(&i);

    float O[16];
    float Z[4];

    #pragma unroll
    for (int j=0; j < 4; j++){
        Z[j] = 0.0f;
    }

    asm volatile(
          "mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 "
          "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%10,%11,%12,%13};\n"
          : "=f"(O[0]), "=f"(O[1]), "=f"(O[2]), "=f"(O[3])
          : "r"(I[0]), "r"(I[1]), "r"(I[2]), "r"(I[3]), "r"(A[0]), "r"(A[2]),
            "f"(Z[0]), "f"(Z[1]), "f"(Z[2]), "f"(Z[3]));
    
    asm volatile(
          "mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 "
          "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%10,%11,%12,%13};\n"
          : "=f"(O[4]), "=f"(O[5]), "=f"(O[6]), "=f"(O[7])
          : "r"(I[0]), "r"(I[1]), "r"(I[2]), "r"(I[3]), "r"(A[1]), "r"(A[3]),
            "f"(Z[0]), "f"(Z[1]), "f"(Z[2]), "f"(Z[3]));
    
    asm volatile(
          "mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 "
          "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%10,%11,%12,%13};\n"
          : "=f"(O[8]), "=f"(O[9]), "=f"(O[10]), "=f"(O[11])
          : "r"(I[0]), "r"(I[1]), "r"(I[2]), "r"(I[3]), "r"(A[4]), "r"(A[6]),
            "f"(Z[0]), "f"(Z[1]), "f"(Z[2]), "f"(Z[3]));
    
    asm volatile(
          "mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 "
          "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%10,%11,%12,%13};\n"
          : "=f"(O[12]), "=f"(O[13]), "=f"(O[14]), "=f"(O[15])
          : "r"(I[0]), "r"(I[1]), "r"(I[2]), "r"(I[3]), "r"(A[5]), "r"(A[7]),
            "f"(Z[0]), "f"(Z[1]), "f"(Z[2]), "f"(Z[3]));
    
    bfloat16_t *da_t = reinterpret_cast<bfloat16_t *>(&da);

    #pragma unroll
    for (int i=0; i < 16; i++){
      da_t[i] = bfloat16_t(O[i]);
    }
    
  }

  CUTLASS_HOST_DEVICE
  void todense(FragmentA const&a, FragmentAT &da, FragmentB const &i, uint32_t const &E, int const id2) const {
    uint32_t const *A = reinterpret_cast<uint32_t const *>(&a);
    uint32_t const *I = reinterpret_cast<uint32_t const *>(&i);

    float dense[16];
    float zeros[4];

    #pragma unroll
    for (int j=0; j < 4; j++){
        zeros[j] = 0.0f;
    }

    float *DO = reinterpret_cast<float *>(dense);
    float *Z = reinterpret_cast<float *>(zeros);

    if (id2 == 0) {
    asm volatile(
        "mma.sp.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 "
        "{%0,%1,%2,%3}, {%4,%5}, {%6,%7}, {%8,%9,%10,%11}, %12, 0x0;\n"
        : "=f"(DO[0]), "=f"(DO[1]), "=f"(DO[2]), "=f"(DO[3])
        : "r"(A[0]), "r"(A[1]), "r"(I[0]), "r"(I[1]), 
          "f"(Z[0]), "f"(Z[1]), "f"(Z[2]), "f"(Z[3]), "r"(E));
    asm volatile(
        "mma.sp.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 "
        "{%0,%1,%2,%3}, {%4,%5}, {%6,%7}, {%8,%9,%10,%11}, %12, 0x0;\n"
        : "=f"(DO[4]), "=f"(DO[5]), "=f"(DO[6]), "=f"(DO[7])
        : "r"(A[0]), "r"(A[1]), "r"(I[1]), "r"(I[0]), 
          "f"(Z[0]), "f"(Z[1]), "f"(Z[2]), "f"(Z[3]), "r"(E));
    asm volatile(
        "mma.sp.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 "
        "{%0,%1,%2,%3}, {%4,%5}, {%6,%7}, {%8,%9,%10,%11}, %12, 0x1;\n"
        : "=f"(DO[8]), "=f"(DO[9]), "=f"(DO[10]), "=f"(DO[11])
        : "r"(A[2]), "r"(A[3]), "r"(I[0]), "r"(I[1]), 
          "f"(Z[0]), "f"(Z[1]), "f"(Z[2]), "f"(Z[3]), "r"(E));
    asm volatile(
        "mma.sp.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 "
        "{%0,%1,%2,%3}, {%4,%5}, {%6,%7}, {%8,%9,%10,%11}, %12, 0x1;\n"
        : "=f"(DO[12]), "=f"(DO[13]), "=f"(DO[14]), "=f"(DO[15])
        : "r"(A[2]), "r"(A[3]), "r"(I[1]), "r"(I[0]), 
          "f"(Z[0]), "f"(Z[1]), "f"(Z[2]), "f"(Z[3]), "r"(E));
    } else if (id2 == 1) {
    asm volatile(
        "mma.sp.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 "
        "{%0,%1,%2,%3}, {%4,%5}, {%6,%7}, {%8,%9,%10,%11}, %12, 0x2;\n"
        : "=f"(DO[0]), "=f"(DO[1]), "=f"(DO[2]), "=f"(DO[3])
        : "r"(A[0]), "r"(A[1]), "r"(I[0]), "r"(I[1]), 
          "f"(Z[0]), "f"(Z[1]), "f"(Z[2]), "f"(Z[3]), "r"(E));
    asm volatile(
        "mma.sp.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 "
        "{%0,%1,%2,%3}, {%4,%5}, {%6,%7}, {%8,%9,%10,%11}, %12, 0x2;\n"
        : "=f"(DO[4]), "=f"(DO[5]), "=f"(DO[6]), "=f"(DO[7])
        : "r"(A[0]), "r"(A[1]), "r"(I[1]), "r"(I[0]), 
          "f"(Z[0]), "f"(Z[1]), "f"(Z[2]), "f"(Z[3]), "r"(E));
    asm volatile(
        "mma.sp.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 "
        "{%0,%1,%2,%3}, {%4,%5}, {%6,%7}, {%8,%9,%10,%11}, %12, 0x3;\n"
        : "=f"(DO[8]), "=f"(DO[9]), "=f"(DO[10]), "=f"(DO[11])
        : "r"(A[2]), "r"(A[3]), "r"(I[0]), "r"(I[1]), 
          "f"(Z[0]), "f"(Z[1]), "f"(Z[2]), "f"(Z[3]), "r"(E));
    asm volatile(
        "mma.sp.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 "
        "{%0,%1,%2,%3}, {%4,%5}, {%6,%7}, {%8,%9,%10,%11}, %12, 0x3;\n"
        : "=f"(DO[12]), "=f"(DO[13]), "=f"(DO[14]), "=f"(DO[15])
        : "r"(A[2]), "r"(A[3]), "r"(I[1]), "r"(I[0]), 
          "f"(Z[0]), "f"(Z[1]), "f"(Z[2]), "f"(Z[3]), "r"(E));
    } else {
    assert(0);
    }

    bfloat16_t *da_t = reinterpret_cast<bfloat16_t *>(&da);
    #pragma unroll
    for (int i=0; i < 16; i++){
      da_t[i] = bfloat16_t(dense[i]);
    }
  }

  CUTLASS_HOST_DEVICE
  void operator()(FragmentC &d1, FragmentC &d2, FragmentAT const &a, FragmentB const &b,
                  FragmentC const &c1, FragmentC const &c2, int const id2) const {

    uint32_t const *A = reinterpret_cast<uint32_t const *>(&a);
    uint32_t const *B = reinterpret_cast<uint32_t const *>(&b);
    float const *C1 = reinterpret_cast<float const *>(&c1);
    float const *C2 = reinterpret_cast<float const *>(&c2);
    float *D1 = reinterpret_cast<float *>(&d1);
    float *D2 = reinterpret_cast<float *>(&d2);

    if (id2 == 0){
      // Multiply with B
      asm volatile(
          "mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 "
          "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%10,%11,%12,%13};\n"
          : "=f"(D1[0]), "=f"(D1[1]), "=f"(D1[2]), "=f"(D1[3])
          : "r"(A[0]), "r"(A[1]), "r"(A[2]), "r"(A[3]), "r"(B[0]), "r"(B[1]),
            "f"(C1[0]), "f"(C1[1]), "f"(C1[2]), "f"(C1[3]));

      // Multiply with B
      asm volatile(
          "mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 "
          "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%10,%11,%12,%13};\n"
          : "=f"(D2[0]), "=f"(D2[1]), "=f"(D2[2]), "=f"(D2[3])
          : "r"(A[4]), "r"(A[5]), "r"(A[6]), "r"(A[7]), "r"(B[0]), "r"(B[1]),
            "f"(C2[0]), "f"(C2[1]), "f"(C2[2]), "f"(C2[3]));
      } else {
        // Multiply with B
        asm volatile(
            "mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 "
            "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%10,%11,%12,%13};\n"
            : "=f"(D1[0]), "=f"(D1[1]), "=f"(D1[2]), "=f"(D1[3])
            : "r"(A[0]), "r"(A[1]), "r"(A[2]), "r"(A[3]), "r"(B[2]), "r"(B[3]),
              "f"(C1[0]), "f"(C1[1]), "f"(C1[2]), "f"(C1[3]));

        // Multiply with B
        asm volatile(
            "mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 "
            "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%10,%11,%12,%13};\n"
            : "=f"(D2[0]), "=f"(D2[1]), "=f"(D2[2]), "=f"(D2[3])
            : "r"(A[4]), "r"(A[5]), "r"(A[6]), "r"(A[7]), "r"(B[2]), "r"(B[3]),
              "f"(C2[0]), "f"(C2[1]), "f"(C2[2]), "f"(C2[3]));
      }
  }
};


template <>
struct SparseMmaV2<gemm::GemmShape<16, 8, 32>, 32, half_t, layout::RowMajor,
           half_t, layout::ColumnMajor, float, layout::RowMajor,
           OpMultiplyAdd, true, SPFormatType::Thread> {
  using Shape = gemm::GemmShape<16, 8, 32>;

  using ElementA = half_t;
  using LayoutA = layout::RowMajor;
  using FragmentA = Array<half_t, 8>;
  using FragmentAT = Array<half_t, 16>;

  using ElementB = half_t;
  using LayoutB = layout::ColumnMajor;
  using FragmentB = Array<half_t, 8>;

  using ElementC = float;
  using LayoutC = layout::RowMajor;
  using FragmentC = Array<float, 4>;

  using FragmentE = uint32_t;

  using Operator = OpMultiplyAdd;
  using ArchTag = arch::Sm80;

  static int const kSparse = 2;

  static int const kMetaSizeInBits = 2;

  static int const kMaxID2 = 2;


  CUTLASS_DEVICE
  void transpose(FragmentAT &da, FragmentB const &i) const{
    // Transpose the matrix with wmma
    uint32_t *A = reinterpret_cast<uint32_t *>(&da);
    uint32_t const *I = reinterpret_cast<uint32_t const *>(&i);

    uint32_t tmp = A[1];

    asm volatile(
        "mma.sync.aligned.m16n8k16.row.col.f16.f16.f16.f16 "
        "{%0,%1}, {%2,%3,%4,%5}, {%6,%7}, {%8,%9};\n"
        : "=r"(A[0]), "=r"(A[1])
        : "r"(I[0]), "r"(I[1]), "r"(I[2]), "r"(I[3]), "r"(A[0]), "r"(A[2]),
          "r"(I[4]), "r"(I[5])
    );

    asm volatile(
        "mma.sync.aligned.m16n8k16.row.col.f16.f16.f16.f16 "
        "{%0,%1}, {%2,%3,%4,%5}, {%6,%7}, {%8,%9};\n"
        : "=r"(A[2]), "=r"(A[3])
        : "r"(I[0]), "r"(I[1]), "r"(I[2]), "r"(I[3]), "r"(tmp), "r"(A[3]),
          "r"(I[4]), "r"(I[5])
    );

    tmp = A[5];

    asm volatile(
        "mma.sync.aligned.m16n8k16.row.col.f16.f16.f16.f16 "
        "{%0,%1}, {%2,%3,%4,%5}, {%6,%7}, {%8,%9};\n"
        : "=r"(A[4]), "=r"(A[5])
        : "r"(I[0]), "r"(I[1]), "r"(I[2]), "r"(I[3]), "r"(A[4]), "r"(A[6]),
          "r"(I[4]), "r"(I[5])
    );

    asm volatile(
        "mma.sync.aligned.m16n8k16.row.col.f16.f16.f16.f16 "
        "{%0,%1}, {%2,%3,%4,%5}, {%6,%7}, {%8,%9};\n"
        : "=r"(A[6]), "=r"(A[7])
        : "r"(I[0]), "r"(I[1]), "r"(I[2]), "r"(I[3]), "r"(tmp), "r"(A[7]),
          "r"(I[4]), "r"(I[5])
    );
  
  }

  CUTLASS_HOST_DEVICE
  void todense(FragmentA const&a, FragmentAT &da, FragmentB const &i, uint32_t const &E, int const id2) const {
    uint32_t const *A = reinterpret_cast<uint32_t const *>(&a);
    uint32_t *D = reinterpret_cast<uint32_t *>(&da);
    uint32_t const *I = reinterpret_cast<uint32_t const *>(&i);

    if (id2 == 0) {
    asm volatile(
        "mma.sp.sync.aligned.m16n8k16.row.col.f16.f16.f16.f16 "
        "{%0,%1}, {%2,%3}, {%4,%5}, {%6,%7}, %8, 0x0;\n"
        : "=r"(D[0]), "=r"(D[1])
        : "r"(A[0]), "r"(A[1]), "r"(I[0]), "r"(I[1]), 
          "r"(I[4]), "r"(I[5]), "r"(E));
    asm volatile(
        "mma.sp.sync.aligned.m16n8k16.row.col.f16.f16.f16.f16 "
        "{%0,%1}, {%2,%3}, {%4,%5}, {%6,%7}, %8, 0x0;\n"
        : "=r"(D[2]), "=r"(D[3])
        : "r"(A[0]), "r"(A[1]), "r"(I[2]), "r"(I[3]), 
          "r"(I[4]), "r"(I[5]), "r"(E));
    asm volatile(
        "mma.sp.sync.aligned.m16n8k16.row.col.f16.f16.f16.f16 "
        "{%0,%1}, {%2,%3}, {%4,%5}, {%6,%7}, %8, 0x1;\n"
        : "=r"(D[4]), "=r"(D[5])
        : "r"(A[2]), "r"(A[3]), "r"(I[0]), "r"(I[1]), 
          "r"(I[4]), "r"(I[5]), "r"(E));
    asm volatile(
        "mma.sp.sync.aligned.m16n8k16.row.col.f16.f16.f16.f16 "
        "{%0,%1}, {%2,%3}, {%4,%5}, {%6,%7}, %8, 0x1;\n"
        : "=r"(D[6]), "=r"(D[7])
        : "r"(A[2]), "r"(A[3]), "r"(I[2]), "r"(I[3]), 
          "r"(I[4]), "r"(I[5]), "r"(E));
    } else if (id2 == 1) {
    asm volatile(
        "mma.sp.sync.aligned.m16n8k16.row.col.f16.f16.f16.f16 "
        "{%0,%1}, {%2,%3}, {%4,%5}, {%6,%7}, %8, 0x2;\n"
        : "=r"(D[0]), "=r"(D[1])
        : "r"(A[0]), "r"(A[1]), "r"(I[0]), "r"(I[1]), 
          "r"(I[4]), "r"(I[5]), "r"(E));
    asm volatile(
        "mma.sp.sync.aligned.m16n8k16.row.col.f16.f16.f16.f16 "
        "{%0,%1}, {%2,%3}, {%4,%5}, {%6,%7}, %8, 0x2;\n"
        : "=r"(D[2]), "=r"(D[3])
        : "r"(A[0]), "r"(A[1]), "r"(I[2]), "r"(I[3]), 
          "r"(I[4]), "r"(I[5]), "r"(E));
    asm volatile(
        "mma.sp.sync.aligned.m16n8k16.row.col.f16.f16.f16.f16 "
        "{%0,%1}, {%2,%3}, {%4,%5}, {%6,%7}, %8, 0x3;\n"
        : "=r"(D[4]), "=r"(D[5])
        : "r"(A[2]), "r"(A[3]), "r"(I[0]), "r"(I[1]), 
          "r"(I[4]), "r"(I[5]), "r"(E));
    asm volatile(
        "mma.sp.sync.aligned.m16n8k16.row.col.f16.f16.f16.f16 "
        "{%0,%1}, {%2,%3}, {%4,%5}, {%6,%7}, %8, 0x3;\n"
        : "=r"(D[6]), "=r"(D[7])
        : "r"(A[2]), "r"(A[3]), "r"(I[2]), "r"(I[3]), 
          "r"(I[4]), "r"(I[5]), "r"(E));
    } else {
    assert(0);
    }
  }
  CUTLASS_HOST_DEVICE
  void operator()(FragmentC &d1, FragmentC &d2, FragmentAT const &a, FragmentB const &b,
                  FragmentC const &c1, FragmentC const &c2, int const id2) const {

    uint32_t const *A = reinterpret_cast<uint32_t const *>(&a);
    uint32_t const *B = reinterpret_cast<uint32_t const *>(&b);
    float const *C1 = reinterpret_cast<float const *>(&c1);
    float const *C2 = reinterpret_cast<float const *>(&c2);
    float *D1 = reinterpret_cast<float *>(&d1);
    float *D2 = reinterpret_cast<float *>(&d2);

    if (id2 == 0){
      // Multiply with B
      asm volatile(
          "mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 "
          "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%10,%11,%12,%13};\n"
          : "=f"(D1[0]), "=f"(D1[1]), "=f"(D1[2]), "=f"(D1[3])
          : "r"(A[0]), "r"(A[1]), "r"(A[2]), "r"(A[3]), "r"(B[0]), "r"(B[1]),
            "f"(C1[0]), "f"(C1[1]), "f"(C1[2]), "f"(C1[3]));

      // Multiply with B
      asm volatile(
          "mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 "
          "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%10,%11,%12,%13};\n"
          : "=f"(D2[0]), "=f"(D2[1]), "=f"(D2[2]), "=f"(D2[3])
          : "r"(A[4]), "r"(A[5]), "r"(A[6]), "r"(A[7]), "r"(B[0]), "r"(B[1]),
            "f"(C2[0]), "f"(C2[1]), "f"(C2[2]), "f"(C2[3]));
    } else {
      asm volatile(
          "mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 "
          "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%10,%11,%12,%13};\n"
          : "=f"(D1[0]), "=f"(D1[1]), "=f"(D1[2]), "=f"(D1[3])
          : "r"(A[0]), "r"(A[1]), "r"(A[2]), "r"(A[3]), "r"(B[2]), "r"(B[3]),
            "f"(C1[0]), "f"(C1[1]), "f"(C1[2]), "f"(C1[3]));

      // Multiply with B
      asm volatile(
          "mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 "
          "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%10,%11,%12,%13};\n"
          : "=f"(D2[0]), "=f"(D2[1]), "=f"(D2[2]), "=f"(D2[3])
          : "r"(A[4]), "r"(A[5]), "r"(A[6]), "r"(A[7]), "r"(B[2]), "r"(B[3]),
            "f"(C2[0]), "f"(C2[1]), "f"(C2[2]), "f"(C2[3]));
    }
  }
};

/////////////////////////////////////////////////////////////////////////////////////////////////

} // namespace arch
} // namespace cutlass

/////////////////////////////////////////////////////////////////////////////////////////////////