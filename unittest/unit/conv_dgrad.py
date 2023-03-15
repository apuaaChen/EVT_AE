import torch
from torch.fx import symbolic_trace
from torch.fx.passes.shape_prop import ShapeProp
import sys
sys.path.append("/workspace/sparseTraining/Codegen/compiler")
from passes import pass_conv_fusion, pass_print_graph, pass_suffix_elimination, pass_composed_op_breakdown
import unittest
import pycutlass
import operator


class ConvDgradTestSm80(unittest.TestCase):
    def test_conv2d_dgrad(self):
        ## Define model to trace
        class Conv2dDgrad(torch.nn.Module):
            def __init__(self) -> None:
                super().__init__()
            
            def forward(self, weight, input, grad_output):
                output =  torch.ops.aten.convolution_backward(
                    grad_output, input, weight, [0], [2, 2], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]
                )

                dgrad = operator.getitem(output, 0)
                wgrad = operator.getitem(output, 1)

                return dgrad, wgrad

                # data_gradient = torch.nn.grad.conv2d_input(
                #     (128, 256, 14, 14), weight, grad_output, [2, 2], [1, 1], [1, 1], 1
                # )
                # return dgrad, wgrad
        
        ## module instances
        module = Conv2dDgrad()
        module_reference = Conv2dDgrad()

        ## run the compiler pass
        symbolic_traced : torch.fx.GraphModule = symbolic_trace(module)
        weight = torch.randn([512, 256, 1, 1], dtype=torch.float16, device="cuda")
        input = torch.randn([128, 256, 14, 14], dtype=torch.float16, device="cuda")
        grad_output = torch.randn([128, 512, 7, 7], dtype=torch.float16, device="cuda")

        ShapeProp(symbolic_traced).propagate(weight, input, grad_output)

        pass_print_graph(symbolic_traced, "./gemm.svg")

        pass_suffix_elimination(symbolic_traced, symbolic_traced.graph)
        pass_composed_op_breakdown(symbolic_traced, symbolic_traced.graph)
        pass_conv_fusion(symbolic_traced, symbolic_traced.graph)

        symbolic_traced.recompile()

        pass_print_graph(symbolic_traced, "./gemm_optimized.svg")

        out, _ = symbolic_traced(weight, input, grad_output)
        ref, _ = module_reference(weight, input, grad_output)
        
        torch.cuda.synchronize()

        out = out.contiguous()

        print(ref.permute(0, 2, 3, 1).contiguous().view(-1, 256))
        # print(torch.nonzero(out.permute(0, 2, 3, 1).contiguous().view(-1, 256)).t()[0].view(-1, 256).t()[0].size())
        print(out.permute(0, 2, 3, 1).contiguous().view(-1, 256))

if __name__ == '__main__':
    pycutlass.get_memory_pool(manager="torch")
    pycutlass.compiler.nvcc()
    unittest.main()
