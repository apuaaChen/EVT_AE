#include <cuda.h>
#include <torch/extension.h>
#include <cuda_runtime.h>

#include "../src/cuda/spmm_template.h"

#include "./scratch_space/matmul_config.h"

torch::Tensor spmm_f16_ntn_cuda(
    torch::Tensor tensor_a,
    torch::Tensor tensor_b,
    torch::Tensor tensor_e,
    const float alpha)
{
    using Config = SpMMConfigure<Element, LayoutB, Trans, ThreadblockShape, WarpShape, InstructionShape, NumStages>;
    return spmm_cuda<Config>(tensor_a, tensor_b, tensor_e, alpha);
}