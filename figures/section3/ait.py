import torch
from aitemplate.frontend import nn, Tensor
from aitemplate.compiler import compile_model, ops
from aitemplate.compiler.ops.common.epilogue import FuncEnum
from aitemplate.testing import detect_target
import nvtx

b = 128
m = 1024
n = 1024
k = 256

# create graph
X = Tensor(shape=(m, b, k), dtype="float16", name="X", is_input=True)
W = Tensor(shape=(b, k, n), dtype="float16", name="W", is_input=True)
bias = Tensor(shape=(b, 1, n), dtype="float16", name="bias", is_input=True)

X_permute = ops.permute102()(X)
bmm_out = ops.bmm_rrr()(X_permute, W)
relu = ops.relu(bmm_out)
add = ops.elementwise(FuncEnum.ADD)(relu, bias)

relu_permute = ops.permute102()(add)
view = ops.reshape()(relu_permute, shape=[m * b, n])
sum = ops.reduce_sum(dim=[0])(view)

relu._attrs["is_output"]=True
relu._attrs["name"]="relu"
view._attrs["is_output"]=True
view._attrs["name"]="view"
sum._attrs["is_output"]=True
sum._attrs["name"]="sum"
target = detect_target()
module = compile_model([relu, view, sum], target, "./", test_name="test")

X_torch = torch.randn((m, b, k), dtype=torch.float16, device="cuda")
W_torch = torch.randn((b, k, n), dtype=torch.float16, device="cuda")
bias_torch = torch.randn((b, 1, n), dtype=torch.float16, device="cuda")

relu_torch = torch.zeros([b, m, n], dtype=torch.float16, device="cuda")
view_torch = torch.zeros([m * b, n], dtype=torch.float16, device="cuda")
sum_torch = torch.zeros([n], dtype=torch.float16, device="cuda")


for i in range(10):
    module.run_with_tensors([X_torch, W_torch, bias_torch], [relu_torch, view_torch, sum_torch])

for i in range(10):
    with nvtx.annotate("ait"):
        module.run_with_tensors([X_torch, W_torch, bias_torch], [relu_torch, view_torch, sum_torch])
