#include <torch/script.h>
#include <torch/custom_class.h>
#include <ATen/cuda/CUDAGraph.h>
#include <c10/cuda/CUDACachingAllocator.h>
#include <ATen/cuda/Exceptions.h>
#include <ATen/cuda/CUDAGeneratorImpl.h>



using torch::autograd::variable_list;
using torch::autograd::AutogradContext;
using torch::autograd::Variable;
using torch::autograd::variable_list;

// Next step: Hake the cuda graph

struct myCUDAGraph : at::cuda::CUDAGraph {
    void capture_end_priority() {
        auto stream = at::cuda::getCurrentCUDAStream();

        TORCH_CHECK(stream == capture_stream_,
                    "Capture must end on the same stream it began on.");

        c10::cuda::CUDACachingAllocator::notifyCaptureEnd(capture_dev_, id_);

        AT_CUDA_CHECK(cudaStreamEndCapture(capture_stream_, &graph_));
        TORCH_CHECK(graph_ != NULL, "Invalid capture.");
        has_graph_ = true;

        // Trailing NULL, NULL, 0 arguments were recommended by Cuda driver people,
        // who prefer not to report error message through these arguments moving forward
        // (they prefer return value, or errors on api calls internal to the capture)
        // AT_CUDA_CHECK(cudaGraphInstantiate(&graph_exec_, graph_, NULL, NULL, 0));
        AT_CUDA_CHECK(cudaGraphInstantiateWithFlags(&graph_exec_, graph_, cudaGraphInstantiateFlagUseNodePriority));
        has_graph_exec_ = true;

        auto* gen = at::get_generator_or_default<at::CUDAGeneratorImpl>(
            c10::nullopt, at::cuda::detail::getDefaultCUDAGenerator());
        TORCH_CHECK(gen == capture_gen_,
                    "Default CUDA RNG generator on current device at capture end "
                    "is different from default generator on current device "
                    "when capture began");
        wholegraph_increment_ = gen->capture_epilogue();

        // Now that we've instantiated graph_ into graph_exec_,
        // we don't need graph_ anymore.
        AT_CUDA_CHECK(cudaGraphDestroy(graph_));
        has_graph_ = false;
    }
};

class SimpleClass : public myCUDAGraph, public torch::CustomClassHolder {
public:
    int64_t value_;
    myCUDAGraph graph_;
    SimpleClass(int64_t value) {
        value_ = value;
        graph_ = myCUDAGraph();
    }

    void capture_begin() {
        graph_.capture_begin();
    }

    void capture_end() {
        graph_.capture_end_priority();
    }

    void replay() {
        graph_.replay();
    }
};





class SPMM_Trace_Fn : public torch::autograd::Function<SPMM_Trace_Fn> {
public:
    static torch::Tensor forward(
        AutogradContext* ctx,
        Variable tensor_a,
        Variable tensor_b) {
        
        ctx->save_for_backward({tensor_a, tensor_b});

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


        auto grad_a = torch::randn_like(tensor_a);
        auto grad_b = torch::randn_like(tensor_b);

        return {grad_a, grad_b};
    }
};


torch::Tensor spmm_trace(
    torch::Tensor tensor_a,
    torch::Tensor tensor_b)
{
    // get problem size
    int m, n, k;

    m = tensor_a.size(-2);
    n = tensor_b.size(-2);
    k = tensor_b.size(-1);

    int batch_size = tensor_b.numel() / n / k;
    auto options_val = torch::TensorOptions().dtype(tensor_a.dtype()).device(tensor_b.device());
    torch::Tensor output_matrix = torch::empty({batch_size, m, n}, options_val);

    auto results = SPMM_Trace_Fn::apply(tensor_a, tensor_b);

    return results;
}



TORCH_LIBRARY(my_ops, m) {
    m.def("spmm_trace", spmm_trace);
    m.class_<SimpleClass>("SimpleClass")
        .def(torch::init<int64_t>())
        .def("capture_begin", [](c10::intrusive_ptr<SimpleClass> self){self->capture_begin();})
        .def("capture_end", [](c10::intrusive_ptr<SimpleClass> self){self->capture_end();})
        .def("replay", [](c10::intrusive_ptr<SimpleClass> self){self->replay();})
        .def_readwrite("value", &SimpleClass::value_);
}