import torch
from torch.fx import symbolic_trace
from torch.fx.passes.shape_prop import ShapeProp
import sys
sys.path.append("/workspace/sparseTraining/Codegen/compiler")
from passes import pass_gemm_fusion, pass_print_graph
import nvtx


class WeightGradient(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
    
    def forward(self, gradient_input, input):
        gradient_input_trans = torch.ops.aten.t(gradient_input)
        return torch.ops.aten.mm(gradient_input_trans, input)

## module instances
module = WeightGradient()
module_reference = WeightGradient()

## run the compiler pass
symbolic_traced : torch.fx.GraphModule = symbolic_trace(module)
gradient_input = torch.randn((16384, 4096), dtype=torch.float16, device="cuda")
input = torch.randn((16384, 4096), dtype=torch.float16, device="cuda")

ShapeProp(symbolic_traced).propagate(gradient_input, input)

# pass_print_graph(symbolic_traced, "./gemm.svg")

pass_gemm_fusion(symbolic_traced, symbolic_traced.graph)

symbolic_traced.recompile()

# pass_print_graph(symbolic_traced, "./gemm_optimized.svg")

for i in range(40):
    out = symbolic_traced(gradient_input, input)
for i in range(40):
    ref = module_reference(gradient_input, input)

for i in range(40):
    with nvtx.annotate("optimized"):
        out = symbolic_traced(gradient_input, input)

for i in range(40):
    with nvtx.annotate("origin"):
        ref = module_reference(gradient_input, input)