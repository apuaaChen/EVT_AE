import torch
from torch.fx import symbolic_trace
from torch.fx.passes.shape_prop import ShapeProp
import sys
sys.path.append("/workspace/sparseTraining/Codegen/compiler")
from passes import pass_gemm_fusion, pass_print_graph
import unittest

class SoftmaxTestSm80(unittest.TestCase):
    def test_softmax_forward(self):
        ## Define model to trace
        class SoftmaxForward(torch.nn.Module):
            def __init__(self) -> None:
                super().__init__()
            
            def forward(self, input):
                softmax_output = torch.ops.aten._softmax(input, -1, False)
                return torch.ops.aten.div(softmax_output, 8.)

        ## Define reference model
        class SoftmaxForwardRef(torch.nn.Module):
            def __init__(self) -> None:
                super().__init__()
            
            def forward(self, input):
                # using float output from softmax kernels to reduce numeric error
                softmax_output = torch.ops.aten._softmax(input, -1, True)
                return torch.ops.aten.div(softmax_output, 8.).to(torch.float16)

        ## module instances
        module = SoftmaxForward()
        module_reference = SoftmaxForwardRef()

        ## run the compiler pass
        symbolic_traced : torch.fx.GraphModule = symbolic_trace(module)
        x = torch.randn((32, 16, 512, 512), dtype=torch.float16, device="cuda")

        ShapeProp(symbolic_traced).propagate(x)

        pass_print_graph(symbolic_traced, "./softmax.svg")

        pass_gemm_fusion(symbolic_traced, symbolic_traced.graph)

        symbolic_traced.recompile()

        pass_print_graph(symbolic_traced, "./softmax_optimized.svg")

        out = symbolic_traced(x)
        ref = module_reference(x)

        self.assertTrue(torch.allclose(out, ref, rtol=5e-2))
    
    def test_softmax_backward(self):
        # Define model to trace
        class SoftmaxBackward(torch.nn.Module):
            def __init__(self) -> None:
                super().__init__()
            
            def forward(self, grad_output, softmax_output):
                grad_input = torch.ops.aten._softmax_backward_data(
                    grad_output, softmax_output, -1, torch.float16
                )
                return torch.ops.aten.div(grad_input, 8.)

        ## module instances
        module = SoftmaxBackward()
        module_reference = SoftmaxBackward()

        ## run the compiler pass
        symbolic_traced : torch.fx.GraphModule = symbolic_trace(module)
        grad_output = torch.randn(
            (32, 16, 512, 512), dtype=torch.float16, device="cuda")
        softmax_output = torch.randn(
            (32, 16, 512, 512), dtype=torch.float16, device="cuda")
        
        ShapeProp(symbolic_traced).propagate(grad_output, softmax_output)

        pass_print_graph(symbolic_traced, "./softmax_backward.svg")

        pass_gemm_fusion(symbolic_traced, symbolic_traced.graph)

        symbolic_traced.recompile()

        pass_print_graph(symbolic_traced, "./softmax_backward_optimized.svg")

        out = symbolic_traced(grad_output, softmax_output)
        ref = module_reference(grad_output, softmax_output)

        self.assertTrue(torch.allclose(out, ref, atol=0.5))
    
    def test_softmax_dropout_forward(self):
        # Define model to trace
        class SoftmaxForward(torch.nn.Module):
            def __init__(self) -> None:
                super().__init__()
            
            def forward(self, input):
                softmax_output = torch.ops.aten._softmax(input, -1, False)
                drouput_output, mask = torch.ops.aten.native_dropout(
                    softmax_output, 0.5, True)
                return drouput_output, mask, softmax_output
        
        ## module instance
        module = SoftmaxForward()

        ## run the compiler pass
        symbolic_traced : torch.fx.GraphModule = symbolic_trace(module)
        x = torch.randn((32, 16, 512, 512), dtype=torch.float16, device="cuda")

        ShapeProp(symbolic_traced).propagate(x)

        pass_print_graph(symbolic_traced, "./softmax.svg")

        pass_gemm_fusion(symbolic_traced, symbolic_traced.graph)

        symbolic_traced.recompile()

        pass_print_graph(symbolic_traced, "./softmax_optimized.svg")

        dropout_out, mask, softmax_output = symbolic_traced(x)

        softmax_output_ref = torch.ops.aten._softmax(x, -1, True)
        dropout_out_ref = softmax_output_ref * mask.to(torch.float32) / 0.5

        self.assertTrue(
            torch.allclose(
                softmax_output, softmax_output_ref.to(torch.float16), rtol=5e-2
            )
        )

        self.assertTrue(
            torch.allclose(
                dropout_out, dropout_out_ref.to(torch.float16), rtol=5e-2
            )
        )

        print(torch.sum(mask) / mask.numel())

        self.assertTrue( 
            torch.allclose(torch.sum(mask) / mask.numel(), torch.Tensor([0.5]).to("cuda"), atol=0.05)
        )



if __name__ == '__main__':
    unittest.main()