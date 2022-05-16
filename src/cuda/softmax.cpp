#include <torch/extension.h>
#include <vector>


torch::Tensor softmax_cuda(
    torch::Tensor input_,
    const int64_t dim_,
    float bias_);

torch::Tensor softmax(
    torch::Tensor input_,
    const int64_t dim_,
    float bias_)
{
    return softmax_cuda(input_, dim_, bias_);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m){
    m.def("softmax", &softmax, "softmax kernel");
}