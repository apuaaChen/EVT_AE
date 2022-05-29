from utils import get_kernel_name, generate_code
import torch


def add_header(code):
    headers = ["torch/extension.h", "vector"]
    for h in headers:
        code += """#include <{header}>\n""".format(header=h)
    
    return code


def add_kernel(code, config):

    if config["out_format"] in ["row", "col"]:
        code += """
torch::Tensor {kernel_l}_cuda(
    torch::Tensor tensor_a,
    torch::Tensor tensor_b,
    torch::Tensor tensor_e);

torch::Tensor {kernel_l}(
    torch::Tensor tensor_a,
    torch::Tensor tensor_b,
    torch::Tensor tensor_e)
{{
    return {kernel_l}_cuda(tensor_a, tensor_b, tensor_e);
}}
""".format(kernel_l = get_kernel_name(config).lower())

    return code

def pybind_prologue(code):
    code += """\n\n\nPYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {\n"""

    return code

def register_pybind(code, config):
    code += """ m.def("{kernel_l}", &{kernel_l}, "{kernel_l}");\n""".format(kernel_l = get_kernel_name(config).lower())

    return code

def pybind_epilogue(code):
    code += """}"""

    return code

    

######################################################
# Test case
######################################################

code = ""

config = {
    "dtype": torch.float16,
    "lhs_format": "row_sp",
    "rhs_format": "col",
    "out_format": "row",
    "batched": True,
    "name": "spmm_ntn",
    "threadblockTileShape": (128, 128, 64),
    "warpTileShape": (64, 64, 64),
    "InstructionShape": (16, 8, 32),
    "Stage": 3
}

code = add_header(code)
code = add_kernel(code, config)
code = pybind_prologue(code)
code = register_pybind(code, config)
code = pybind_epilogue(code)


generate_code(code, None, file_name="kernel.cpp")