#include <torch/script.h>


using torch::autograd::variable_list;
using torch::autograd::AutogradContext;
using torch::autograd::Variable;
using torch::autograd::variable_list;

class SPMM_Trace_Fn : public torch::autograd::Function<SPMM_Trace_Fn> {
public:
    static torch::Tensor forward(
        AutogradContext* ctx,
        Variable tensor_a,
        Variable tensor_b,
        Variable tensor_e) {
        
        ctx->save_for_backward({tensor_a, tensor_b, tensor_e});

        int m, n, k;

        m = tensor_a.size(-2);
        n = tensor_b.size(-2);
        k = tensor_b.size(-1);

        int batch_size = tensor_b.numel() / n / k;
        auto options_val = torch::TensorOptions().dtype(tensor_a.dtype()).device(tensor_b.device());
        torch::Tensor output_matrix = torch::empty({batch_size, m, n}, options_val);

        return output_matrix;
    }

    static variable_list backward(
        AutogradContext* ctx,
        variable_list grad_output){
        
        auto saved = ctx->get_saved_variables();
        auto tensor_a = saved[0];
        auto tensor_b = saved[1];
        auto tensor_e = saved[2];


        auto grad_a = torch::randn_like(tensor_a);
        auto grad_b = torch::randn_like(tensor_b);

        return {grad_a, grad_b, Variable()};
    }
};


torch::Tensor spmm_trace(
    torch::Tensor tensor_a,
    torch::Tensor tensor_b,
    torch::Tensor tensor_e)
{
    // get problem size
    int m, n, k;

    m = tensor_a.size(-2);
    n = tensor_b.size(-2);
    k = tensor_b.size(-1);

    int batch_size = tensor_b.numel() / n / k;
    auto options_val = torch::TensorOptions().dtype(tensor_a.dtype()).device(tensor_b.device());
    torch::Tensor output_matrix = torch::empty({batch_size, m, n}, options_val);

    auto results = SPMM_Trace_Fn::apply(tensor_a, tensor_b, tensor_e);

    return results;
}

TORCH_LIBRARY(my_ops, m) {
    m.def("spmm_trace", spmm_trace);
}