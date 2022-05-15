import torch
import nvtx
from sptrain.sddmm import sddmm_bf16_ntn, sddmm_f16_ntn
from sptrain.sddmm_target import sddmm_target_bf16_ntn, sddmm_target_f16_ntn
import dspattn
import torch.nn.functional as F

batch_size = 32
head = 8
bs = int(batch_size / head)
sequence_length = 4096
embedding = 64


half = torch.float16

input = torch.randn(size=(28*128, 1024), dtype=half, device="cuda")
weight = torch.randn(size=(32320, 1024), dtype=half, device="cuda")
target = torch.randint(low=0, high=1, size=(28 * 128,1), dtype=torch.int64, device="cuda")
mask = None

nonzeros, metadata, target_sp = sddmm_target_f16_ntn(input, weight, target, mask, 1.)


for i in range(10):
    with nvtx.annotate("pytorch"):
        dense_matrix_ref = torch.matmul(input, weight.t())

for i in range(10):
    with nvtx.annotate("sptrain_target"):
        nonzeros, metadata, target_sp = sddmm_target_f16_ntn(input, weight, target, mask, 1.)

for i in range(10):
    with nvtx.annotate("sptrain"):
        nonzeros, metadata = sddmm_f16_ntn(input, weight, mask, 1.)