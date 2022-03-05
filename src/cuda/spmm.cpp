#include <torch/extension.h>
#include <vector>


torch::Tensor spmmv2_bf16_cuda(
    torch::Tensor tensor_a_,
    torch::Tensor tensor_b_,
    torch::Tensor tensor_e_);

torch::Tensor spmmv2_bf16(
    torch::Tensor tensor_a_,
    torch::Tensor tensor_b_,
    torch::Tensor tensor_e_)
{
    return spmmv2_bf16_cuda(tensor_a_, tensor_b_, tensor_e_);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m){
    m.def("spmmv2_bf16", &spmmv2_bf16, "Cutlass SpMM bf16 kernel");
}