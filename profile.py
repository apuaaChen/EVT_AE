from re import I
import torch
from sparse_ops import SparseLinear
import nvtx

bs = 128
in_feat = 1024
out_feat = 1024

model = SparseLinear(in_features=1024, out_features=1024, N=2, M=4).to("cuda").to(torch.bfloat16)
x = torch.randn(size=(bs, in_feat), dtype=torch.bfloat16, device="cuda", requires_grad=True)
grad = torch.randn(size=(bs, out_feat), dtype=torch.bfloat16, device="cuda")

# forward pass
for i in range(10):
    with nvtx.annotate("Forward"):
        y = model(x)
    with nvtx.annotate("Backward"):
        y.backward(grad)

# print(x.grad)
# print(model.weight.grad)