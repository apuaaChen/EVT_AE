import torch
import nvtx
from sptrain.sddmm import sddmm_bf16_ntn, sddmm_f16_ntn
import dspattn
import torch.nn.functional as F


sequence_length = 8192
embedding = 64


dtype = torch.bfloat16

query = torch.randn(size=(sequence_length, embedding), dtype=dtype, device="cuda")
key = torch.randn(size=(sequence_length, embedding), dtype=dtype, device="cuda")
key_t = torch.randn(size=(embedding, sequence_length), dtype=dtype, device="cuda")


for i in range(10):
    with nvtx.annotate("pytorch"):
        dense_matrix_ref = torch.matmul(query, key_t)


for i in range(10):
    with nvtx.annotate("sptrain"):
        nonzeros, metadata = sddmm_bf16_ntn(query, key)

for i in range(10):
    with nvtx.annotate("dspattn"):
        nonzeros, metadata = dspattn.sddmm(query, key)