import torch
from torch.utils.cpp_extension import load
from sptrain.meta import bdense2sparse_gold
import os

# base_dir = "./scratch_space/"
# # WDIR = os.getcwd()

# kernels = load(name="spmm", sources=[
#     base_dir + "kernel.cpp",
#     base_dir + "spmm_ntn.cu"
# ], 
# extra_cflags={'-lineinfo'},
# extra_cuda_cflags={'-arch=sm_80', '--ptxas-options=-v', '-lineinfo', '-lcublass', '-use_fast_math'},
# extra_include_paths=['../thirdparty/cutlass/include', '../thirdparty/cutlass/tools/util/include', '../thirdparty/cutlass/examples/common', '../src/cuda'],
# verbose=True)



# L = 4
# batch_size = 4096
# feat_in = 1024
# feat_out = 2048
# half = torch.float16

# alpha = 0.5

# dense_matrix = torch.randn(size=(L, batch_size, feat_in), dtype=half, device="cuda")
# rhs_matrix = torch.randn(size=(L, feat_out, feat_in), dtype=half, device="cuda")
# nonzeros, uncompressed, metadata = bdense2sparse_gold(dense_matrix, False)
# output_matrix_ref = torch.bmm(uncompressed, rhs_matrix.transpose(1, 2))
# output_matrix = kernels.spmm_ntn(nonzeros, rhs_matrix, metadata)


# assert torch.allclose(output_matrix, output_matrix_ref, atol=0.5)


L = 4
batch_size = 4096
feat_in = 1024
feat_out = 2048
half = torch.float16


dense_matrix = torch.randn(size=(L, batch_size, feat_in), dtype=half, device="cuda")
rhs_matrix = torch.randn(size=(L, feat_out, feat_in), dtype=half, device="cuda")

nonzeros, uncompressed, metadata = bdense2sparse_gold(dense_matrix, False)

nonzeros.requires_grad_(True)
rhs_matrix.requires_grad_(True)

# @torch.jit.script
# def spmm(sp_uncompressed, rhs_matrix):
#     output_matrix = torch.bmm(sp_uncompressed.dense, rhs_matrix.transpose(1, 2))
#     return output_matrix

torch.ops.load_library("torchscript/build/libspmm_trace.so")
@torch.jit.script
def spmm(nonzeros, rhs_matrix, metadata):
    output_matrix = torch.ops.my_ops.spmm_trace(nonzeros, rhs_matrix.transpose(1, 2), metadata)
    return output_matrix

print(spmm.graph)

out = spmm(nonzeros, rhs_matrix, metadata)

print(out)

grad = torch.randn_like(out)
out.backward(grad)
