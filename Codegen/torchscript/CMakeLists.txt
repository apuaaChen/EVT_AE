cmake_minimum_required(VERSION 3.1 FATAL_ERROR)
project(spmm_trace)

find_package(Torch REQUIRED)

# Define our library target
add_library(spmm_trace SHARED spmm.cpp)
# Enable C++14
target_compile_features(spmm_trace PRIVATE cxx_std_14)
# Link against LibTorch
target_compile_features(spmm_trace PRIVATE cxx_range_for)
target_link_libraries(spmm_trace PRIVATE "${TORCH_LIBRARIES}")


