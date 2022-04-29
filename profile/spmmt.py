import torch
import nvtx
import unittest
from sptrain.meta import bdense2sparse_gold
from sptrain.spmm import spmmv2_bf16_ntn
from sptrain.spmmt import spmmt_bf16_ntn, spmmt_f16_ntn, spmmt_f16_ntt, spmmt_bf16_ntt, spmmt_f16_nnn
import torch.nn.functional as F


# batch_size = 16384
# feat_in = 1024
# feat_out = 2048
# dtype = torch.float16


# dense_matrix = torch.randn(size=(feat_out, feat_in), dtype=dtype, device="cuda")
# lhs_matrix = torch.randn(size=(batch_size, feat_out), dtype=dtype, device="cuda")
# nonzeros, uncompressed, metadata = bdense2sparse_gold(dense_matrix, False)

# for i in range(10):
#     with nvtx.annotate("pytorch"):
#         output_matrix_ref = torch.matmul(lhs_matrix.to(torch.float16), uncompressed.to(torch.float16))
    
#     with nvtx.annotate("spmmt"):
#         output_matrix = spmmt_f16_ntt(nonzeros.to(torch.float16), lhs_matrix.to(torch.float16), metadata)

L = 2
batch_size = 16384
feat_in = 1024
feat_out = 2048
dtype = torch.float16


dense_matrix = torch.randn(size=(L, feat_out, feat_in), dtype=dtype, device="cuda")
lhs_matrix = torch.randn(size=(L, batch_size, feat_out), dtype=dtype, device="cuda")
lhs_matrix_t = torch.randn(size=(L, feat_out, batch_size), dtype=dtype, device="cuda")
nonzeros, uncompressed, metadata = bdense2sparse_gold(dense_matrix, False)

for i in range(10):
    with nvtx.annotate("pytorch"):
        output_matrix_ref = torch.matmul(lhs_matrix, uncompressed)
    
    with nvtx.annotate("spmmt"):
        output_matrix = spmmt_f16_ntn(nonzeros, lhs_matrix, metadata)
    
    with nvtx.annotate("spmmt_nnn"):
        output_matrix = spmmt_f16_nnn(nonzeros, lhs_matrix_t, metadata)


# dense_matrix = torch.randn(size=(batch_size, feat_in), dtype=dtype, device="cuda")
# rhs_matrix = torch.randn(size=(feat_out, feat_in), dtype=dtype, device="cuda")
# nonzeros, uncompressed, metadata = bdense2sparse_gold(dense_matrix.to(torch.bfloat16), False)

# nonzeros = nonzeros.to(dtype)
# uncompressed = uncompressed.to(dtype)

# for i in range(10):
#     with nvtx.annotate("pytorch"):
#         output_matrix_ref = torch.matmul(uncompressed, rhs_matrix.t())
#     # with nvtx.annotate("spmm"):
#     #     output_matrix = spmmv2_bf16_ntn(nonzeros, rhs_matrix, metadata)
    
#     with nvtx.annotate("spmmt"):
#         output_matrix = spmmt_f16_ntn(nonzeros, rhs_matrix, metadata)