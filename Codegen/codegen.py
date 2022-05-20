import torch
import os


# A basic spmm config
config = {
    "dtype": torch.float16,
    "lhs": ((4, 4096, 1024), "row_sp"),
    "rhs": ((4, 2048, 1024),"col"),
    "epilogue": "row",
    # Design Space
    "threadblockTileShape": (128, 128, 64),
    "warpTileShape": (64, 64, 64),
    "InstructionShape": (16, 8, 32),
    "Stage": 3
}


# A configuration template for generating CUDA code
# "%s" are the parameters to be filled for a concrete CUDA implementation
# TODO: this is a subject to change

config_template = """// Auto-generated file. DO NOT MODIFY!
using Element = %s;
using LayoutB = %s;
static const bool Trans = %s;

using ThreadblockShape = cutlass::gemm::GemmShape<%s, %s, %s>;
using WarpShape = cutlass::gemm::GemmShape<%s, %s, %s>;
using InstructionShape = cutlass::gemm::GemmShape<%s, %s, %s>;
static const int NumStages = %s;
"""

def generate_code(template, parameter, file_name="config.h"):
    if not os.path.isdir("scratch_space"):
        os.mkdir("sratch_space")
    config_file_path = os.path.join("scratch_space", file_name)
    # clean previous configuration
    try:
        os.remove(config_file_path)
    except OSError:
        pass
    file = open(config_file_path, 'w+')
    parameters = tuple(parameter)

    file.write(template % parameters)

if __name__ == "__main__":
    # Generate configuration code
    generate_code(
        config_template,
        parameter=("cutlass::half_t", "cutlass::layout::ColumnMajor", "false", 128, 128, 64, 64, 64, 64, 16, 8, 32, 3),
        file_name="matmul_config.h"
    )