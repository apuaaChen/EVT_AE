#include <torch/extension.h>
#include <vector>

std::vector<torch::Tensor> spmmt_reduce_bf16_ntn_cuda(
    torch::Tensor tensor_a_,
    torch::Tensor tensor_b_,
    torch::Tensor tensor_e_,
    const float alpha);

std::vector<torch::Tensor> spmmt_reduce_bf16_ntn(
    torch::Tensor tensor_a_,
    torch::Tensor tensor_b_,
    torch::Tensor tensor_e_,
    const float alpha)
{
    return spmmt_reduce_bf16_ntn_cuda(tensor_a_, tensor_b_, tensor_e_, alpha);
}

std::vector<torch::Tensor> spmmt_reduce_bf16_ntt_cuda(
    torch::Tensor tensor_a_,
    torch::Tensor tensor_b_,
    torch::Tensor tensor_e_,
    const float alpha);

std::vector<torch::Tensor> spmmt_reduce_bf16_ntt(
    torch::Tensor tensor_a_,
    torch::Tensor tensor_b_,
    torch::Tensor tensor_e_,
    const float alpha)
{
    return spmmt_reduce_bf16_ntt_cuda(tensor_a_, tensor_b_, tensor_e_, alpha);
}


std::vector<torch::Tensor> spmmt_reduce_f16_ntn_cuda(
    torch::Tensor tensor_a_,
    torch::Tensor tensor_b_,
    torch::Tensor tensor_e_,
    const float alpha);

std::vector<torch::Tensor> spmmt_reduce_f16_ntn(
    torch::Tensor tensor_a_,
    torch::Tensor tensor_b_,
    torch::Tensor tensor_e_,
    const float alpha)
{
    return spmmt_reduce_f16_ntn_cuda(tensor_a_, tensor_b_, tensor_e_, alpha);
}

std::vector<torch::Tensor> spmmt_reduce_f16_ntt_cuda(
    torch::Tensor tensor_a_,
    torch::Tensor tensor_b_,
    torch::Tensor tensor_e_,
    const float alpha);

std::vector<torch::Tensor> spmmt_reduce_f16_ntt(
    torch::Tensor tensor_a_,
    torch::Tensor tensor_b_,
    torch::Tensor tensor_e_,
    const float alpha)
{
    return spmmt_reduce_f16_ntt_cuda(tensor_a_, tensor_b_, tensor_e_, alpha);
}

std::vector<torch::Tensor> spmmt_reduce_bf16_nnn_cuda(
    torch::Tensor tensor_a_,
    torch::Tensor tensor_b_,
    torch::Tensor tensor_e_,
    const float alpha);

std::vector<torch::Tensor> spmmt_reduce_bf16_nnn(
    torch::Tensor tensor_a_,
    torch::Tensor tensor_b_,
    torch::Tensor tensor_e_,
    const float alpha)
{
    return spmmt_reduce_bf16_nnn_cuda(tensor_a_, tensor_b_, tensor_e_, alpha);
}

std::vector<torch::Tensor> spmmt_reduce_bf16_nnt_cuda(
    torch::Tensor tensor_a_,
    torch::Tensor tensor_b_,
    torch::Tensor tensor_e_,
    const float alpha);

std::vector<torch::Tensor> spmmt_reduce_bf16_nnt(
    torch::Tensor tensor_a_,
    torch::Tensor tensor_b_,
    torch::Tensor tensor_e_,
    const float alpha)
{
    return spmmt_reduce_bf16_nnt_cuda(tensor_a_, tensor_b_, tensor_e_, alpha);
}


std::vector<torch::Tensor> spmmt_reduce_f16_nnn_cuda(
    torch::Tensor tensor_a_,
    torch::Tensor tensor_b_,
    torch::Tensor tensor_e_,
    const float alpha);

std::vector<torch::Tensor> spmmt_reduce_f16_nnn(
    torch::Tensor tensor_a_,
    torch::Tensor tensor_b_,
    torch::Tensor tensor_e_,
    const float alpha)
{
    return spmmt_reduce_f16_nnn_cuda(tensor_a_, tensor_b_, tensor_e_, alpha);
}

std::vector<torch::Tensor> spmmt_reduce_f16_nnt_cuda(
    torch::Tensor tensor_a_,
    torch::Tensor tensor_b_,
    torch::Tensor tensor_e_,
    const float alpha);

std::vector<torch::Tensor> spmmt_reduce_f16_nnt(
    torch::Tensor tensor_a_,
    torch::Tensor tensor_b_,
    torch::Tensor tensor_e_,
    const float alpha)
{
    return spmmt_reduce_f16_nnt_cuda(tensor_a_, tensor_b_, tensor_e_, alpha);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m){
    m.def("spmmt_reduce_bf16_ntn", &spmmt_reduce_bf16_ntn, "Cutlass SpMM bf16 kernel ntn");
    m.def("spmmt_reduce_bf16_ntt", &spmmt_reduce_bf16_ntt, "Cutlass SpMM bf16 kernel ntt");
    m.def("spmmt_reduce_f16_ntn", &spmmt_reduce_f16_ntn, "Cutlass SpMM fp16 kernel ntn");
    m.def("spmmt_reduce_f16_ntt", &spmmt_reduce_f16_ntt, "Cutlass SpMM fp16 kernel ntt");
    m.def("spmmt_reduce_bf16_nnn", &spmmt_reduce_bf16_nnn, "Cutlass SpMM bf16 kernel ntn");
    m.def("spmmt_reduce_bf16_nnt", &spmmt_reduce_bf16_nnt, "Cutlass SpMM bf16 kernel ntt");
    m.def("spmmt_reduce_f16_nnn", &spmmt_reduce_f16_nnn, "Cutlass SpMM fp16 kernel ntn");
    m.def("spmmt_reduce_f16_nnt", &spmmt_reduce_f16_nnt, "Cutlass SpMM fp16 kernel ntt");
}