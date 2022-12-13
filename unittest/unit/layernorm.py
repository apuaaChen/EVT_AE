import torch
from torch.fx import symbolic_trace
from torch.fx.passes.shape_prop import ShapeProp
import sys
sys.path.append("/workspace/sparseTraining/Codegen/compiler")
from passes import pass_gemm_fusion, pass_print_graph, pass_layernorm_preprocessing
import unittest

class LayerNormTestSm80(unittest.TestCase):
    def test_layernorm_forward(self):
        ## Define model to trace
        class LayerNormForwardBackward(torch.nn.Module):
            def __init__(self) -> None:
                super().__init__()
            
            def forward(self, input, normalized_shape, gamma, beta, weight):
                output, mean, std = torch.ops.aten.native_layer_norm(input, normalized_shape, gamma, beta, 1e-12)
                grad_out = torch.ops.aten.mm(output, weight)
                grad_input, grad_gamma, grad_beta = torch.ops.aten.native_layer_norm_backward(
                    grad_out, input, normalized_shape, mean, std, gamma, beta, [True, True, True]
                )
                return grad_input, grad_gamma, grad_beta
        
        ## module instance
        module = LayerNormForwardBackward()
        module_reference = LayerNormForwardBackward()

        ## run the compiler pass
        symbolic_traced : torch.fx.GraphModule = symbolic_trace(module)

        x = torch.randn(size=(32, 1024), dtype=torch.float16, device="cuda") * 2 + 1
        normalized_shape = [1024,]
        gamma = torch.randn(size=(1024,), dtype=torch.float16, device="cuda")
        beta = torch.randn(size=(1024,), dtype=torch.float16, device="cuda")
        weight = torch.randn(size=(1024, 1024), dtype=torch.float16, device="cuda")

        ShapeProp(symbolic_traced).propagate(x, normalized_shape, gamma, beta, weight)

        pass_print_graph(symbolic_traced, "./layernorm_fp.svg")
        
        pass_layernorm_preprocessing(symbolic_traced, symbolic_traced.graph)

        pass_gemm_fusion(symbolic_traced, symbolic_traced.graph)

        symbolic_traced.recompile()

        pass_print_graph(symbolic_traced, "./layernorm_fp_optimized.svg")

        grad_input, grad_gamma, grad_beta = symbolic_traced(x, normalized_shape, gamma, beta, weight)
        grad_input_ref, grad_gamma_ref, grad_beta_ref = module_reference(x, normalized_shape, gamma, beta, weight)

        print(torch.mean(x, dim=1))
        print(1./torch.std(x, dim=1))

        print(grad_gamma)
        print(grad_gamma_ref)

        print(grad_beta)
        print(grad_beta_ref)

        # self.assertTrue(torch.allclose(out, ref, rtol=5e-2))

if __name__ == '__main__':
    unittest.main()
