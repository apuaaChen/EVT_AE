import torch
from torch.utils.cpp_extension import load
from sptrain.meta import bdense2sparse_gold
import os

base_dir = "./"
WDIR = os.getcwd()

kernels = load(name="spmm", sources=[
    base_dir + "kernel.cpp",
    base_dir + "kernel.cu"
], 
extra_cflags={'-lineinfo'},
extra_cuda_cflags={'-arch=sm_80', '--ptxas-options=-v', '-lineinfo', '-lcublass', '-use_fast_math'},
extra_include_paths=['../thirdparty/cutlass/include', '../thirdparty/cutlass/tools/util/include', '../thirdparty/cutlass/examples/common'],
verbose=True)



L = 4
batch_size = 4096
feat_in = 1024
feat_out = 2048
half = torch.float16

alpha = 0.5

dense_matrix = torch.randn(size=(L, batch_size, feat_in), dtype=half, device="cuda")
rhs_matrix = torch.randn(size=(L, feat_out, feat_in), dtype=half, device="cuda")
nonzeros, uncompressed, metadata = bdense2sparse_gold(dense_matrix, False)
output_matrix_ref = torch.bmm(uncompressed, rhs_matrix.transpose(1, 2)) * alpha
output_matrix = kernels.spmm_f16_ntn(nonzeros, rhs_matrix, metadata, alpha)


assert torch.allclose(output_matrix, output_matrix_ref, atol=0.5)