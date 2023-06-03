import torch
from torch.fx import symbolic_trace
from torch.fx.passes.shape_prop import ShapeProp
import sys
sys.path.append("/workspace/gtl/sparseTraining/Codegen/compiler")
from gtl.passes import pass_gemm_fusion, pass_print_graph
import unittest



class BmmTestSm80(unittest.TestCase):
    def test_bmm_default_2(self):
        ## Define model to trace
        class BmmDefault2(torch.nn.Module):
            def __init__(self) -> None:
                super().__init__()
            
            def forward(self, view_default_18, view_default_19, adder):
                permute_4 = torch.ops.aten.permute(view_default_18, [1, 0, 2])
                permute_default_1 = torch.ops.aten.permute(view_default_19, [1, 2, 0])
                bmm_default_2 = torch.ops.aten.bmm(permute_4, permute_default_1)
                view_default_21 = torch.ops.aten.view(bmm_default_2, [32, 16, 512, 512])
                mul_23 = torch.ops.aten.mul(view_default_21, 0.125)
                add_tensor_6 = torch.ops.aten.add(mul_23, adder)

                return add_tensor_6
        
        ## module instances
        module = BmmDefault2()
        module_reference = BmmDefault2()

        ## run the compiler pass
        symbolic_traced : torch.fx.GraphModule = symbolic_trace(module)
        view_default_18 = torch.randn((512, 512, 64), dtype=torch.float16, device="cuda")
        view_default_19 = torch.randn((512, 512, 64), dtype=torch.float16, device="cuda")

        # The adder is wrong!!!!
        adder = torch.randn((32, 1, 1, 512), dtype=torch.float16, device="cuda")

        ShapeProp(symbolic_traced).propagate(view_default_18, view_default_19, adder)

        pass_print_graph(symbolic_traced, "./gemm.svg")

        pass_gemm_fusion(symbolic_traced, symbolic_traced.graph)

        symbolic_traced.recompile()

        pass_print_graph(symbolic_traced, "./gemm_optimized.svg")

        out = symbolic_traced(view_default_18, view_default_19, adder)
        ref = module_reference(view_default_18, view_default_19, adder)

        print(out.view(-1))
        print(ref.view(-1))

        self.assertTrue(torch.allclose(out, ref, atol=1))

if __name__ == '__main__':
    unittest.main()