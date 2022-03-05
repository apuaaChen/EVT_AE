from re import I
import torch
from sparse_ops import SparseLinear, SparseLinearV2
import nvtx
import torch.optim as optim

bs = 4096
in_feat = 1024
out_feat = 512

model = SparseLinear(in_features=in_feat, out_features=out_feat, N=2, M=4).to("cuda").to(torch.bfloat16)
modelv2 = SparseLinearV2(in_features=in_feat, out_features=out_feat, N=2, M=4).to("cuda").to(torch.bfloat16)

modelv2.weight = torch.nn.Parameter(model.weight.clone())
modelv2.bias = torch.nn.Parameter(model.bias.clone())

x = torch.randn(size=(bs, in_feat), dtype=torch.bfloat16, device="cuda", requires_grad=True)
x_ref = x.detach().clone().requires_grad_(True)
grad = torch.randn(size=(bs, out_feat), dtype=torch.bfloat16, device="cuda")

# Verify the results
y = model(x_ref)
y2 = modelv2(x)

y.backward(grad)
y2.backward(grad)

assert torch.allclose(y, y2, rtol=1e-5)
assert torch.allclose(x.grad, x_ref.grad, rtol=1e-5)
assert torch.allclose(model.weight.grad, modelv2.weight.grad, rtol=1e-5)
assert torch.allclose(model.bias.grad, modelv2.bias.grad, rtol=1e-5)


# forward pass
for i in range(10):
    with nvtx.annotate("Forward"):
        y = model(x)
    with nvtx.annotate("Backward"):
        y.backward(grad)

for i in range(10):
    with nvtx.annotate("Forward"):
        y2 = modelv2(x_ref)
    with nvtx.annotate("Backward"):
        y2.backward(grad)

# print(x.grad)
# print(model.weight.grad)

optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.9)