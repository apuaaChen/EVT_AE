from numpy import dtype
import torch
from sptrain.meta import bdense2sparse
import nvtx

batch_size = 8192
feat_in = 1024
dtype = torch.bfloat16

dense_matrix = torch.randn(size=(batch_size, feat_in), dtype=dtype, device="cuda")

for i in range(20):
    with nvtx.annotate("pruning"):
        nonzeros, metadata = bdense2sparse(dense_matrix)