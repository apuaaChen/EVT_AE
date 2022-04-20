from re import I
import torch
from sparse_ops import SparseLinear, SparseLinearV2, SparseLinearV3
import nvtx
import torch.optim as optim

bs = 32768
in_feat = 1024
out_feat = 2048

model = SparseLinearV2(in_features=in_feat, out_features=out_feat, N=2, M=4, bias=False).to("cuda").to(torch.float16)
modelv2 = SparseLinearV3(in_features=in_feat, out_features=out_feat, N=2, M=4, bias=False).to("cuda").to(torch.float16)

modelv2.weight = torch.nn.Parameter(model.weight.clone())
if model.bias is not None:
    modelv2.bias = torch.nn.Parameter(model.bias.clone())

x = torch.randn(size=(bs, in_feat), dtype=torch.float16, device="cuda", requires_grad=True)
x_ref = x.detach().clone().requires_grad_(True)
grad = torch.randn(size=(bs, out_feat), dtype=torch.float16, device="cuda")

# Verify the results
y = model(x_ref)
y2 = modelv2(x)

y.backward(grad)
y2.backward(grad)

assert torch.allclose(y, y2, atol=0.5)
assert torch.allclose(x.grad, x_ref.grad, atol=0.05)
assert torch.allclose(model.weight.grad, modelv2.weight.grad, rtol=1e-5)
if model.bias is not None:
    assert torch.allclose(model.bias.grad, modelv2.bias.grad, rtol=1e-5)


# forward pass
for i in range(10):
    with nvtx.annotate("Dense"):
        with nvtx.annotate("Forward"):
            y = model(x)
        with nvtx.annotate("Backward"):
            y.backward(grad)

for i in range(10):
    with nvtx.annotate("Sparse"):
        with nvtx.annotate("Forward"):
            y2 = modelv2(x_ref)
        with nvtx.annotate("Backward"):
            y2.backward(grad)

# print(x.grad)
# print(model.weight.grad)

optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.9)