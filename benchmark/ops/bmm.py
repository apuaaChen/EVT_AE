import torch
from torch.fx import symbolic_trace
from torch.fx.passes.shape_prop import ShapeProp
import sys
sys.path.append("/workspace/sparseTraining/Codegen/compiler")
from passes import pass_gemm_fusion, pass_print_graph
import nvtx


class BmmDefault2(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
    
    def forward(self, view_default_18, view_default_19):
        bmm_default_2 = torch.ops.aten.bmm(view_default_18, view_default_19)
        return bmm_default_2


## module instances
module = BmmDefault2()
module_reference = BmmDefault2()

## run the compiler pass
symbolic_traced : torch.fx.GraphModule = symbolic_trace(module)
view_default_18 = torch.randn((512, 512, 64), dtype=torch.float16, device="cuda")
view_default_19 = torch.randn((512, 64, 512), dtype=torch.float16, device="cuda")

# The adder is wrong!!!!

ShapeProp(symbolic_traced).propagate(view_default_18, view_default_19)

pass_gemm_fusion(symbolic_traced, symbolic_traced.graph)

symbolic_traced.recompile()

pass_print_graph(symbolic_traced, "./gemm_optimized.svg")

for i in range(40):
    out = symbolic_traced(view_default_18, view_default_19)
for i in range(40):
    ref = module_reference(view_default_18, view_default_19)

for i in range(40):
    with nvtx.annotate("optimized"):
        out = symbolic_traced(view_default_18, view_default_19)

for i in range(40):
    with nvtx.annotate("origin"):
        ref = module_reference(view_default_18, view_default_19)