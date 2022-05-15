#include <torch/extension.h>
#include <vector>


std::vector<torch::Tensor> sddmm_target_bf16_ntn_cuda(
    torch::Tensor tensor_a_,
    torch::Tensor tensor_b_,
    torch::Tensor target_,
    torch::optional<torch::Tensor> mask_,
    const float alpha_);

std::vector<torch::Tensor> sddmm_target_bf16_ntn(
    torch::Tensor tensor_a_,
    torch::Tensor tensor_b_,
    torch::Tensor target_,
    torch::optional<torch::Tensor> mask_,
    const float alpha_)
{
    return sddmm_target_bf16_ntn_cuda(tensor_a_, tensor_b_, target_, mask_, alpha_);
}


std::vector<torch::Tensor> sddmm_target_f16_ntn_cuda(
    torch::Tensor tensor_a_,
    torch::Tensor tensor_b_,
    torch::Tensor target_,
    torch::optional<torch::Tensor> mask_,
    const float alpha_);

std::vector<torch::Tensor> sddmm_target_f16_ntn(
    torch::Tensor tensor_a_,
    torch::Tensor tensor_b_,
    torch::Tensor target_,
    torch::optional<torch::Tensor> mask_,
    const float alpha_)
{
    return sddmm_target_f16_ntn_cuda(tensor_a_, tensor_b_, target_, mask_, alpha_);
}


PYBIND11_MODULE(TORCH_EXTENSION_NAME, m){
    m.def("sddmm_target_bf16_ntn", &sddmm_target_bf16_ntn, "SDDMM bf16 kernel ntn");
    m.def("sddmm_target_f16_ntn", &sddmm_target_f16_ntn, "SDDMM f16 kernel ntn");
}