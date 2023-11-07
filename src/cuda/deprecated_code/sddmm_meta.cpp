#include <torch/extension.h>
#include <vector>


torch::Tensor sddmm_meta_bf16_ntn_cuda(
    torch::Tensor tensor_a_,
    torch::Tensor tensor_b_,
    torch::Tensor tensor_e_,
    const float alpha_);

torch::Tensor sddmm_meta_bf16_ntn(
    torch::Tensor tensor_a_,
    torch::Tensor tensor_b_,
    torch::Tensor tensor_e_,
    const float alpha_)
{
    return sddmm_meta_bf16_ntn_cuda(tensor_a_, tensor_b_, tensor_e_, alpha_);
}


torch::Tensor sddmm_meta_bf16_tnn_cuda(
    torch::Tensor tensor_a_,
    torch::Tensor tensor_b_,
    torch::Tensor tensor_e_,
    const float alpha_);

torch::Tensor sddmm_meta_bf16_tnn(
    torch::Tensor tensor_a_,
    torch::Tensor tensor_b_,
    torch::Tensor tensor_e_,
    const float alpha_)
{
    return sddmm_meta_bf16_tnn_cuda(tensor_a_, tensor_b_, tensor_e_, alpha_);
}


torch::Tensor sddmm_meta_f16_ntn_cuda(
    torch::Tensor tensor_a_,
    torch::Tensor tensor_b_,
    torch::Tensor tensor_e_,
    const float alpha_);

torch::Tensor sddmm_meta_f16_ntn(
    torch::Tensor tensor_a_,
    torch::Tensor tensor_b_,
    torch::Tensor tensor_e_,
    const float alpha_)
{
    return sddmm_meta_f16_ntn_cuda(tensor_a_, tensor_b_, tensor_e_, alpha_);
}


torch::Tensor sddmm_meta_f16_tnn_cuda(
    torch::Tensor tensor_a_,
    torch::Tensor tensor_b_,
    torch::Tensor tensor_e_,
    const float alpha_);

torch::Tensor sddmm_meta_f16_tnn(
    torch::Tensor tensor_a_,
    torch::Tensor tensor_b_,
    torch::Tensor tensor_e_,
    const float alpha_)
{
    return sddmm_meta_f16_tnn_cuda(tensor_a_, tensor_b_, tensor_e_, alpha_);
}



PYBIND11_MODULE(TORCH_EXTENSION_NAME, m){
    m.def("sddmm_meta_bf16_ntn", &sddmm_meta_bf16_ntn, "SDDMM bf16 kernel ntn");
    m.def("sddmm_meta_bf16_tnn", &sddmm_meta_bf16_tnn, "SDDMM bf16 kernel tnn");
    m.def("sddmm_meta_f16_ntn", &sddmm_meta_f16_ntn, "SDDMM f16 kernel ntn");
    m.def("sddmm_meta_f16_tnn", &sddmm_meta_f16_tnn, "SDDMM f16 kernel tnn");
}