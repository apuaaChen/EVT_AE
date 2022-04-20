import torch
import nvtx
import unittest
from sptrain.meta import bdense2sparse_gold, bdense2sparse
from sptrain.spmm import spmmv2_bf16_nnn, spmmv2_bf16_ntn, spmmv2_bf16_ntt
import torch.nn.functional as F


batch_size = 16384
feat_in = 1024
feat_out = 2048
dtype = torch.bfloat16


dense_matrix = torch.randn(size=(batch_size, feat_in), dtype=dtype, device="cuda")
rhs_matrix = torch.randn(size=(feat_out, feat_in), dtype=dtype, device="cuda")
nonzeros, uncompressed, metadata = bdense2sparse_gold(dense_matrix, False)

for i in range(10):
    with nvtx.annotate("pytorch"):
        output_matrix_ref = torch.matmul(uncompressed, rhs_matrix.t()).t()
    with nvtx.annotate("sptrain"):
        output_matrix = spmmv2_bf16_ntt(nonzeros, rhs_matrix, metadata)
