// Auto-generated file. DO NOT MODIFY!
using Element = cutlass::half_t;
using LayoutB = cutlass::layout::ColumnMajor;
static const bool Trans = false;

using ThreadblockShape = cutlass::gemm::GemmShape<128, 128, 64>;
using WarpShape = cutlass::gemm::GemmShape<64, 64, 64>;
using InstructionShape = cutlass::gemm::GemmShape<16, 8, 32>;
static const int NumStages = 3;
