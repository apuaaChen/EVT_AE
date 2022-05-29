#include <torch/extension.h>
#include <vector>

torch::Tensor spmm_ntn_cuda(
    torch::Tensor tensor_a,
    torch::Tensor tensor_b,
    torch::Tensor tensor_e);

torch::Tensor spmm_ntn(
    torch::Tensor tensor_a,
    torch::Tensor tensor_b,
    torch::Tensor tensor_e)
{
    return spmm_ntn_cuda(tensor_a, tensor_b, tensor_e);
}



PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
 m.def("spmm_ntn", &spmm_ntn, "spmm_ntn");
}