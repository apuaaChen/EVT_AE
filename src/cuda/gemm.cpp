#include <torch/extension.h>
#include <vector>


torch::Tensor gemm_bf16_nnn_cuda(
    torch::Tensor tensor_a_,
    torch::Tensor tensor_b_);

torch::Tensor gemm_bf16_nnn(
    torch::Tensor tensor_a_,
    torch::Tensor tensor_b_)
{
    return gemm_bf16_nnn_cuda(tensor_a_, tensor_b_);
}


PYBIND11_MODULE(TORCH_EXTENSION_NAME, m){
    m.def("gemm_bf16_nnn", &gemm_bf16_nnn, "Cutlass GEMM bf16 kernel nnn");
}