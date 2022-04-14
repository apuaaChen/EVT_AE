#include <torch/extension.h>
#include <vector>


torch::Tensor spmmv2_bf16_nnn_cuda(
    torch::Tensor tensor_a_,
    torch::Tensor tensor_b_,
    torch::Tensor tensor_e_);

torch::Tensor spmmv2_bf16_nnn(
    torch::Tensor tensor_a_,
    torch::Tensor tensor_b_,
    torch::Tensor tensor_e_)
{
    return spmmv2_bf16_nnn_cuda(tensor_a_, tensor_b_, tensor_e_);
}


torch::Tensor spmmv2_bf16_ntn_cuda(
    torch::Tensor tensor_a_,
    torch::Tensor tensor_b_,
    torch::Tensor tensor_e_);

torch::Tensor spmmv2_bf16_ntn(
    torch::Tensor tensor_a_,
    torch::Tensor tensor_b_,
    torch::Tensor tensor_e_)
{
    return spmmv2_bf16_ntn_cuda(tensor_a_, tensor_b_, tensor_e_);
}


torch::Tensor spmmv2_bf16_ntt_cuda(
    torch::Tensor tensor_a_,
    torch::Tensor tensor_b_,
    torch::Tensor tensor_e_);

torch::Tensor spmmv2_bf16_ntt(
    torch::Tensor tensor_a_,
    torch::Tensor tensor_b_,
    torch::Tensor tensor_e_)
{
    return spmmv2_bf16_ntt_cuda(tensor_a_, tensor_b_, tensor_e_);
}


torch::Tensor spmmv2_f16_nnn_cuda(
    torch::Tensor tensor_a_,
    torch::Tensor tensor_b_,
    torch::Tensor tensor_e_);

torch::Tensor spmmv2_f16_nnn(
    torch::Tensor tensor_a_,
    torch::Tensor tensor_b_,
    torch::Tensor tensor_e_)
{
    return spmmv2_f16_nnn_cuda(tensor_a_, tensor_b_, tensor_e_);
}


torch::Tensor spmmv2_f16_ntn_cuda(
    torch::Tensor tensor_a_,
    torch::Tensor tensor_b_,
    torch::Tensor tensor_e_);

torch::Tensor spmmv2_f16_ntn(
    torch::Tensor tensor_a_,
    torch::Tensor tensor_b_,
    torch::Tensor tensor_e_)
{
    return spmmv2_f16_ntn_cuda(tensor_a_, tensor_b_, tensor_e_);
}


torch::Tensor spmmv2_f16_ntt_cuda(
    torch::Tensor tensor_a_,
    torch::Tensor tensor_b_,
    torch::Tensor tensor_e_);

torch::Tensor spmmv2_f16_ntt(
    torch::Tensor tensor_a_,
    torch::Tensor tensor_b_,
    torch::Tensor tensor_e_)
{
    return spmmv2_f16_ntt_cuda(tensor_a_, tensor_b_, tensor_e_);
}




PYBIND11_MODULE(TORCH_EXTENSION_NAME, m){
    m.def("spmmv2_bf16_nnn", &spmmv2_bf16_nnn, "Cutlass SpMM bf16 kernel nnn");
    m.def("spmmv2_bf16_ntn", &spmmv2_bf16_ntn, "Cutlass SpMM bf16 kernel ntn");
    m.def("spmmv2_bf16_ntt", &spmmv2_bf16_ntt, "Cutlass SpMM bf16 kernel ntt");
    m.def("spmmv2_f16_nnn", &spmmv2_f16_nnn, "Cutlass SpMM f16 kernel nnn");
    m.def("spmmv2_f16_ntn", &spmmv2_f16_ntn, "Cutlass SpMM f16 kernel ntn");
    m.def("spmmv2_f16_ntt", &spmmv2_f16_ntt, "Cutlass SpMM f16 kernel ntt");
}