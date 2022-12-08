import torch
from torch.fx import symbolic_trace
from torch.fx.passes.shape_prop import ShapeProp
import sys
sys.path.append("/workspace/sparseTraining/Codegen/compiler")
from passes import pass_gemm_fusion, pass_print_graph
import unittest


class ReductionTestSm80(unittest.TestCase):
    def mm_default_4(self):
        ## Define model to trace
        class GemmReduction(torch.nn.Module):
            def __init__(self) -> None:
                super().__init__()
            
            def forward(self, gradient_input, tanh_output, weight):
                gradient_output = torch.ops.aten.mm(gradient_input, weight)
                tanh_backward = torch.ops.aten.tanh_backward(gradient_output, tanh_output)
                bias = torch.ops.aten.sum(tanh_backward, [0])
                return tanh_backward, bias
        
        ## module instances
        module = GemmReduction()
        module_reference = GemmReduction()

        ## run the compiler pass
        symbolic_traced : torch.fx.GraphModule = symbolic_trace(module)
        gradient_input = torch.randn((32, 2), dtype=torch.float16, device="cuda")
        weight = torch.randn((2, 1024), dtype=torch.float16, device="cuda")
        tanh_output = torch.tanh(torch.randn((32, 1024), dtype=torch.float16, device="cuda"))

        ShapeProp(symbolic_traced).propagate(gradient_input, tanh_output, weight)

        pass_print_graph(symbolic_traced, "./gemm.svg")

        pass_gemm_fusion(symbolic_traced, symbolic_traced.graph)

        symbolic_traced.recompile()

        pass_print_graph(symbolic_traced, "./gemm_optimized.svg")

        outs = symbolic_traced(gradient_input, tanh_output, weight)
        refs = module_reference(gradient_input, tanh_output, weight)

        for out, ref in zip(outs, refs):
            self.assertTrue(torch.allclose(out, ref, atol=1))

    def mm_default_12(self):
        ## Define model to trace
        class GemmReduction(torch.nn.Module):
            def __init__(self) -> None:
                super().__init__()
            
            def forward(self, gradient_input, gelu_input, weight):
                gradient_output = torch.ops.aten.mm(gradient_input, weight)
                view_default_49 = torch.ops.aten.view(gradient_output, [512, 32, 4096])
                gelu_backward = torch.ops.aten.gelu_backward(view_default_49, gelu_input, approximate="tanh")
                view_default_50 = torch.ops.aten.view(gelu_backward, [16384, 4096])
                bias = torch.ops.aten.sum(view_default_50, [0])
                return view_default_50, bias

        ## module instances
        module = GemmReduction()
        module_reference = GemmReduction()

        ## run the compiler pass
        symbolic_traced : torch.fx.GraphModule = symbolic_trace(module)
        gradient_input = torch.randn((16384, 1024), dtype=torch.float16, device="cuda")
        weight = torch.randn((1024, 4096), dtype=torch.float16, device="cuda")
        gelu_input = torch.tanh(torch.randn((512, 32, 4096), dtype=torch.float16, device="cuda"))

        ShapeProp(symbolic_traced).propagate(gradient_input, gelu_input, weight)

        pass_print_graph(symbolic_traced, "./gemm.svg")

        pass_gemm_fusion(symbolic_traced, symbolic_traced.graph)

        symbolic_traced.recompile()

        pass_print_graph(symbolic_traced, "./gemm_optimized.svg")

        outs = symbolic_traced(gradient_input, gelu_input, weight)
        refs = module_reference(gradient_input, gelu_input, weight)

        self.assertTrue(torch.allclose(outs[0], refs[0], atol=1))
        self.assertTrue(torch.allclose(outs[1], refs[1], atol=50))
    
    def test_bmm_default_4(self):
        ## Define model to trace
        class GemmReduction(torch.nn.Module):
            def __init__(self) -> None:
                super().__init__()
            
            def forward(self, gradient_input, weight):
                gradient_output = torch.ops.aten.bmm(gradient_input, weight)
                permute_14 = torch.ops.aten.permute(gradient_output, [1, 0, 2])
                clone_default_6 = torch.ops.aten.clone(permute_14, memory_format=torch.contiguous_format)
                view_default_7 = torch.ops.aten._unsafe_view(clone_default_6, [512, 32, 1024])
                view_default_60 = torch.ops.aten.view(view_default_7, [16384, 1024])
                bias = torch.ops.aten.sum(view_default_60, [0], True)
                return view_default_60, bias
        
        ## module instances
        module = GemmReduction()
        module_reference = GemmReduction()

        ## run the compiler pass
        symbolic_traced : torch.fx.GraphModule = symbolic_trace(module)
        gradient_input = torch.randn((512, 512, 512), dtype=torch.float16, device="cuda")
        weight = torch.randn((512, 512, 64), dtype=torch.float16, device="cuda")

        ShapeProp(symbolic_traced).propagate(gradient_input, weight)

        pass_print_graph(symbolic_traced, "./gemm.svg")

        pass_gemm_fusion(symbolic_traced, symbolic_traced.graph)

        symbolic_traced.recompile()

        pass_print_graph(symbolic_traced, "./gemm_optimized.svg")

        outs = symbolic_traced(gradient_input, weight)
        refs = module_reference(gradient_input, weight)

        self.assertTrue(torch.allclose(outs[0], refs[0], atol=1))
        self.assertTrue(torch.allclose(outs[1], refs[1], atol=50))
    
    def test_bmm_default_6(self):
        ## Define model to trace
        class GemmReduction(torch.nn.Module):
            def __init__(self) -> None:
                super().__init__()
            
            def forward(self, gradient_input, weight):
                gradient_output = torch.ops.aten.bmm(gradient_input, weight)
                permute_14 = torch.ops.aten.permute(gradient_output, [2, 0, 1])
                view_default_7 = torch.ops.aten._unsafe_view(permute_14, [512, 32, 1024])
                clone_default_6 = torch.ops.aten.clone(view_default_7, memory_format=torch.contiguous_format)
                view_default_60 = torch.ops.aten.view(clone_default_6, [16384, 1024])
                bias = torch.ops.aten.sum(view_default_60, [0], True)
                return view_default_60, bias
        
        ## module instances
        module = GemmReduction()
        module_reference = GemmReduction()

        ## run the compiler pass
        symbolic_traced : torch.fx.GraphModule = symbolic_trace(module)
        gradient_input = torch.randn((512, 64, 512), dtype=torch.float16, device="cuda")
        weight = torch.randn((512, 512, 512), dtype=torch.float16, device="cuda")

        ShapeProp(symbolic_traced).propagate(gradient_input, weight)

        pass_print_graph(symbolic_traced, "./gemm.svg")

        pass_gemm_fusion(symbolic_traced, symbolic_traced.graph)

        symbolic_traced.recompile()

        pass_print_graph(symbolic_traced, "./gemm_optimized.svg")

        outs = symbolic_traced(gradient_input, weight)
        refs = module_reference(gradient_input, weight)

        self.assertTrue(torch.allclose(outs[0], refs[0], atol=1))
        self.assertTrue(torch.allclose(outs[1], refs[1], atol=50))



if __name__ == '__main__':
    unittest.main()