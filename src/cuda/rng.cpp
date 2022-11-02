#include <torch/extension.h>
#include <ATen/cuda/CUDAGeneratorImpl.h>
// https://github.com/pytorch/pytorch/blob/d321be61c07bc1201c7fe10cd03d045277a326c1/aten/src/ATen/native/cuda/Dropout.cu


std::tuple<uint64_t, uint64_t> philox_state(int64_t counter_offset) {
    auto gen = at::get_generator_or_default<at::CUDAGeneratorImpl>(c10::nullopt, at::cuda::detail::getDefaultCUDAGenerator());

    at::PhiloxCudaState rng_engine_inputs;
    {
        // See Note [Acquire lock when using random generators]
        std::lock_guard<std::mutex> lock(gen->mutex_);
        rng_engine_inputs = gen->philox_cuda_state(counter_offset);
    }
    
    return std::make_tuple(rng_engine_inputs.seed_, rng_engine_inputs.offset_.val);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("philox_state", &philox_state, "philox_state");
}