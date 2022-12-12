import torch
from torch.fx import symbolic_trace
from torch.fx.passes.shape_prop import ShapeProp
import sys
sys.path.append("/workspace/sparseTraining/Codegen/compiler")
from passes import pass_gemm_fusion, pass_print_graph
import unittest
import pycutlass

class GemmTestSm80(unittest.TestCase):
    def test_weight_gradient(self):
        ## Define model to trace
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
        gradient_input = torch.randn((16384, 1024), dtype=torch.float16, device="cuda")
        input = torch.randn((16384, 1024), dtype=torch.float16, device="cuda")
        
        ShapeProp(symbolic_traced).propagate(gradient_input, input)

        pass_print_graph(symbolic_traced, "./gemm.svg")

        pass_gemm_fusion(symbolic_traced, symbolic_traced.graph)

        symbolic_traced.recompile()

        pass_print_graph(symbolic_traced, "./gemm_optimized.svg")

        ref = module_reference(gradient_input, input)
        out = symbolic_traced(gradient_input, input)
        
        self.assertTrue(torch.allclose(out, ref, atol=1))
            
if __name__ == '__main__':
    pycutlass.get_memory_pool(manager="torch")
    pycutlass.compiler.nvcc()
    unittest.main()
