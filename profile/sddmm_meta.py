import torch
import nvtx
from sptrain.meta import bdense2sparse_gold
from sptrain.sddmm_meta import sddmm_meta_bf16_ntn, sddmm_meta_f16_ntn
from sptrain.sddmm import sddmm_f16_ntn
import dspattn
import torch.nn.functional as F

batch_size = 32
head = 8
bs = int(batch_size / head)
sequence_length = 4096
embedding = 64


dtype = torch.float16

query = torch.randn(size=(batch_size, sequence_length, embedding), dtype=dtype, device="cuda")
key = torch.randn(size=(batch_size, sequence_length, embedding), dtype=dtype, device="cuda")
key_t = torch.randn(size=(batch_size, embedding, sequence_length), dtype=dtype, device="cuda")

dense_matrix = torch.bmm(query, key_t)
_, _, metadata = bdense2sparse_gold(dense_matrix, False)


for i in range(10):
    with nvtx.annotate("pytorch"):
        dense_matrix_ref = torch.bmm(query, key_t)


for i in range(10):
    with nvtx.annotate("sptrain"):
        nonzeros = sddmm_meta_f16_ntn(query, key, metadata, 1.)

for i in range(10):
    with nvtx.annotate("sptrain_sddmm"):
        nonzeros, metadata = sddmm_f16_ntn(query, key, None, 1.)
