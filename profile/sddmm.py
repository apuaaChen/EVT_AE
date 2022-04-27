import torch
import nvtx
from sptrain.sddmm import sddmm_bf16_ntn, sddmm_f16_ntn
import dspattn
import torch.nn.functional as F

batch_size = 32
head = 8
bs = int(batch_size / head)
sequence_length = 4096
embedding = 64


dtype = torch.bfloat16

query = torch.randn(size=(batch_size, sequence_length, embedding), dtype=dtype, device="cuda")
key = torch.randn(size=(batch_size, sequence_length, embedding), dtype=dtype, device="cuda")
key_t = torch.randn(size=(batch_size, embedding, sequence_length), dtype=dtype, device="cuda")

prob = torch.ones(size=(bs, 1, 1, sequence_length), dtype=dtype, device='cuda') * 0.15
mask = torch.bernoulli(prob) * -1e16


for i in range(10):
    with nvtx.annotate("pytorch"):
        dense_matrix_ref = torch.bmm(query, key_t)


for i in range(10):
    with nvtx.annotate("sptrain"):
        nonzeros, metadata = sddmm_bf16_ntn(query, key, None)

for i in range(10):
    with nvtx.annotate("sptrain_mask"):
        nonzeros, metadata = sddmm_bf16_ntn(query, key, mask)

for i in range(10):
    with nvtx.annotate("dspattn"):
        nonzeros, metadata = dspattn.sddmm(query, key)