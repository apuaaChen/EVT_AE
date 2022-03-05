#include <torch/extension.h>
#include <vector>


std::vector<torch::Tensor> batched_dense2sparse_gold_cuda(
    torch::Tensor dense_tensor);

std::vector<torch::Tensor> batched_dense2sparse_gold(
    torch::Tensor dense_tensor)
{
    return batched_dense2sparse_gold_cuda(dense_tensor);
}

std::vector<torch::Tensor> batched_dense2sparse_cuda(
    torch::Tensor dense_tensor);

std::vector<torch::Tensor> batched_dense2sparse(
    torch::Tensor dense_tensor)
{
    return batched_dense2sparse_cuda(dense_tensor);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m){
    m.def("bdense2sparse_gold", &batched_dense2sparse_gold, "Gold function. Convert dense matrix to sparse");
    m.def("bdense2sparse", &batched_dense2sparse, "Convert dense matrix to sparse");
}