import torch
from torch.fx import symbolic_trace
from torch.fx.passes.shape_prop import ShapeProp
import sys
sys.path.append("/workspace/sparseTraining/Codegen/compiler")
from passes import pass_gemm_fusion, pass_print_graph
import nvtx

## Define model to trace
class SoftmaxForward(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
    
    def forward(self, input):
        softmax_output = torch.ops.aten._softmax(input, -1, False)
        drouput_output, mask = torch.ops.aten.native_dropout(softmax_output, 0.1, True)
        return drouput_output, mask, softmax_output

## module instances
module = SoftmaxForward()
module_reference = SoftmaxForward()

## run the compiler pass
symbolic_traced : torch.fx.GraphModule = symbolic_trace(module)
x = torch.randn((32, 16, 512, 512), dtype=torch.float16, device="cuda")

ShapeProp(symbolic_traced).propagate(x)

pass_print_graph(symbolic_traced, "./softmax.svg")

pass_gemm_fusion(symbolic_traced, symbolic_traced.graph)

symbolic_traced.recompile()

pass_print_graph(symbolic_traced, "./softmax_optimized.svg")
for i in range(40):
    out = symbolic_traced(x)
for i in range(40):
    ref = module_reference(x)


for i in range(40):
    with nvtx.annotate("optimized"):
        out = symbolic_traced(x)
for i in range(40):
    with nvtx.annotate("origin"):
        ref = module_reference(x)
