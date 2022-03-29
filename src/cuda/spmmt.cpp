#include <torch/extension.h>
#include <vector>

torch::Tensor spmmt_bf16_ntn_cuda(
    torch::Tensor tensor_a_,
    torch::Tensor tensor_b_,
    torch::Tensor tensor_e_);

torch::Tensor spmmt_bf16_ntn(
    torch::Tensor tensor_a_,
    torch::Tensor tensor_b_,
    torch::Tensor tensor_e_)
{
    return spmmt_bf16_ntn_cuda(tensor_a_, tensor_b_, tensor_e_);
}

torch::Tensor spmmt_f16_ntn_cuda(
    torch::Tensor tensor_a_,
    torch::Tensor tensor_b_,
    torch::Tensor tensor_e_);

torch::Tensor spmmt_f16_ntn(
    torch::Tensor tensor_a_,
    torch::Tensor tensor_b_,
    torch::Tensor tensor_e_)
{
    return spmmt_f16_ntn_cuda(tensor_a_, tensor_b_, tensor_e_);
}

torch::Tensor spmmt_bf16_nnn_cuda(
    torch::Tensor tensor_a_,
    torch::Tensor tensor_b_);

torch::Tensor spmmt_bf16_nnn(
    torch::Tensor tensor_a_,
    torch::Tensor tensor_b_)
{
    return spmmt_bf16_nnn_cuda(tensor_a_, tensor_b_);
}


PYBIND11_MODULE(TORCH_EXTENSION_NAME, m){
    m.def("spmmt_bf16_ntn", &spmmt_bf16_ntn, "Cutlass SpMM bf16 kernel ntn");
    m.def("spmmt_f16_ntn", &spmmt_f16_ntn, "Cutlass SpMM fp16 kernel ntn");
    m.def("spmmt_bf16_nnn", &spmmt_bf16_nnn, "Cutlass SpMM bf16 kernel nnn");
}