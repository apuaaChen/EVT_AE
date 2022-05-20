#include <cuda.h>
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <vector>

#include "spmm_template.h"


// Define the Tile Size in different levels

using ThreadblockShape = cutlass::gemm::GemmShape<128, 128, 64>;
using WarpShape = cutlass::gemm::GemmShape<64, 64, 64>;
using InstructionShape = cutlass::gemm::GemmShape<16, 8, 32>;

// Pipeline stages in GEMM
constexpr int NumStages = 3;



torch::Tensor spmmv2_bf16_nnn_cuda(
    torch::Tensor tensor_a,
    torch::Tensor tensor_b,
    torch::Tensor tensor_e,
    const float alpha)
{
    using Config = SpMMConfigure<cutlass::bfloat16_t, cutlass::layout::RowMajor, false, ThreadblockShape, WarpShape, InstructionShape, NumStages>;
    return spmm_cuda<Config>(tensor_a, tensor_b, tensor_e, alpha);
}


torch::Tensor spmmv2_bf16_ntn_cuda(
    torch::Tensor tensor_a,
    torch::Tensor tensor_b,
    torch::Tensor tensor_e,
    const float alpha)
{
    using Config = SpMMConfigure<cutlass::bfloat16_t, cutlass::layout::ColumnMajor, false, ThreadblockShape, WarpShape, InstructionShape, NumStages>;
    return spmm_cuda<Config>(tensor_a, tensor_b, tensor_e, alpha);
}


torch::Tensor spmmv2_bf16_ntt_cuda(
    torch::Tensor tensor_a,
    torch::Tensor tensor_b,
    torch::Tensor tensor_e,
    const float alpha)
{
    using Config = SpMMConfigure<cutlass::bfloat16_t, cutlass::layout::ColumnMajor, true, ThreadblockShape, WarpShape, InstructionShape, NumStages>;
    return spmm_cuda<Config>(tensor_a, tensor_b, tensor_e, alpha);
}


torch::Tensor spmmv2_f16_nnn_cuda(
    torch::Tensor tensor_a,
    torch::Tensor tensor_b,
    torch::Tensor tensor_e,
    const float alpha)
{
    using Config = SpMMConfigure<cutlass::half_t, cutlass::layout::RowMajor, false, ThreadblockShape, WarpShape, InstructionShape, NumStages>;
    return spmm_cuda<Config>(tensor_a, tensor_b, tensor_e, alpha);
}


torch::Tensor spmmv2_f16_ntn_cuda(
    torch::Tensor tensor_a,
    torch::Tensor tensor_b,
    torch::Tensor tensor_e,
    const float alpha)
{
    using Config = SpMMConfigure<cutlass::half_t, cutlass::layout::ColumnMajor, false, ThreadblockShape, WarpShape, InstructionShape, NumStages>;
    return spmm_cuda<Config>(tensor_a, tensor_b, tensor_e, alpha);
}


torch::Tensor spmmv2_f16_ntt_cuda(
    torch::Tensor tensor_a,
    torch::Tensor tensor_b,
    torch::Tensor tensor_e,
    const float alpha)
{
    using Config = SpMMConfigure<cutlass::half_t, cutlass::layout::ColumnMajor, true, ThreadblockShape, WarpShape, InstructionShape, NumStages>;
    return spmm_cuda<Config>(tensor_a, tensor_b, tensor_e, alpha);
}