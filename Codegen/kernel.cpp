#include <torch/extension.h>
#include <vector>


torch::Tensor spmm_f16_ntn_cuda(
    torch::Tensor tensor_a_,
    torch::Tensor tensor_b_,
    torch::Tensor tensor_e_,
    const float alpha);

torch::Tensor spmm_f16_ntn(
    torch::Tensor tensor_a_,
    torch::Tensor tensor_b_,
    torch::Tensor tensor_e_,
    const float alpha)
{
    return spmm_f16_ntn_cuda(tensor_a_, tensor_b_, tensor_e_, alpha);
}


PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("spmm_f16_ntn", &spmm_f16_ntn, "spmm_f16_ntn");
}