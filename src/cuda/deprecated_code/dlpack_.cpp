#include <torch/extension.h>
#include <c10/cuda/CUDACachingAllocator.h>

void notifyCaptureBegin(int capture_dev_, unsigned long long id_) {
    c10::cuda::CUDACachingAllocator::notifyCaptureBegin(capture_dev_, id_, {id_, 0});
}



PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("notify_capture_begin", &notifyCaptureBegin);
}