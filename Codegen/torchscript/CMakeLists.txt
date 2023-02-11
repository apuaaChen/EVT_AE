cmake_minimum_required(VERSION 3.1 FATAL_ERROR)
project(spmm_trace)
set(CMAKE_PREFIX_PATH "/opt/conda/lib/python3.8/site-packages/torch/share/cmake/Torch;/opt/conda/lib/python3.8/site-packages/pybind11/share/cmake/pybind11")
find_package(Torch REQUIRED)
find_package(pybind11 REQUIRED)

# Define our library target
add_library(spmm_trace SHARED spmm.cpp)
# Enable C++14
target_compile_features(spmm_trace PRIVATE cxx_std_14)
# Link against LibTorch
target_compile_features(spmm_trace PRIVATE cxx_range_for)
target_link_libraries(spmm_trace PRIVATE "${TORCH_LIBRARIES}")
target_link_libraries(spmm_trace PRIVATE pybind11::module pybind11::lto pybind11::windows_extras)

