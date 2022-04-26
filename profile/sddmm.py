import torch
import nvtx
from sptrain.sddmm import sddmm_bf16_ntn, sddmm_f16_ntn
import dspattn
import torch.nn.functional as F

batch_size = 4
sequence_length = 8192
embedding = 64


dtype = torch.bfloat16

query = torch.randn(size=(batch_size, sequence_length, embedding), dtype=dtype, device="cuda")
key = torch.randn(size=(batch_size, sequence_length, embedding), dtype=dtype, device="cuda")
key_t = torch.randn(size=(batch_size, embedding, sequence_length), dtype=dtype, device="cuda")


for i in range(10):
    with nvtx.annotate("pytorch"):
        dense_matrix_ref = torch.bmm(query, key_t)


for i in range(10):
    with nvtx.annotate("sptrain"):
        nonzeros, metadata = sddmm_bf16_ntn(query, key)

for i in range(10):
    with nvtx.annotate("dspattn"):
        nonzeros, metadata = dspattn.sddmm(query, key)