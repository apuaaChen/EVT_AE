import torch
import unittest
from torch.fx import Node, symbolic_trace, subgraph_rewriter, Interpreter
from torch.fx.passes.shape_prop import ShapeProp
from gtl.compiler.passes import pass_print_graph, pass_composed_op_breakdown
from torch._subclasses.fake_tensor import FakeTensorMode, FakeTensor
from torch.fx.experimental.proxy_tensor import py_sym_types
from torch.fx.node import map_aggregate
from torch.fx.passes.shape_prop import _extract_tensor_metadata
from typing import Optional
import re

class DeLogSoftmaxForward(torch.nn.Module):
    def __init__(self, dim, half2float) -> None:
        super().__init__()
        self.dim = dim
        self.half2float = half2float
    
    def forward(self, input):
        return torch.ops.aten.log(torch.ops.aten.softmax(input, self.dim, self.half2float))


class FakeTensorInfer(Interpreter):
    """
    Execute an FX graph Node-by-Node and record a fake tensor representing
    the metadata for the node.  Unlike ShapeProp, (1) this propagation
    is cheap--it does the propagation with meta tensors which do not actually
    store data, and (2) the fake tensors have much more fine grained information,
    e.g., they have accurate alias information that can be consulted by looking
    at the storages.

    Args:
         module (GraphModule): The module to be executed
         mode (Optional[FakeTensorMode]): The dispatch mode used to execute computation indicated by each FX Node.
    """
    def __init__(self, module: torch.fx.GraphModule, mode: Optional[FakeTensorMode] = None):
        super().__init__(module)
        if mode is None:
            mode = FakeTensorMode()
        self._mode = mode
        # dtype used when it cannot be infered
        self.dtype = torch.float16

    def run_node(self, n: Node):
        # when the node's tensor_meta is available, directly create fake tensor
        # from it
        if 'tensor_meta' in n.meta:
            meta = n.meta['tensor_meta']
            with self._mode:
                return torch.empty(size=meta.shape, dtype=meta.dtype)
        else:
            try:
                result = super().run_node(n)
            except:
                pass
            try:
                op_name = str(n.target).split(sep='.')[-1]
                result = getattr(self, op_name)(n)
            except:
                return
            
            def extract_val(obj):
                if isinstance(obj, FakeTensor):
                    return _extract_tensor_metadata(obj)
                elif isinstance(obj, torch.Tensor):
                    return _extract_tensor_metadata(self._mode.from_tensor(obj))
                elif isinstance(obj, py_sym_types):
                    return obj
                else:
                    return None
            meta = map_aggregate(result, extract_val)
            if meta is not None:
                n.meta['tensor_meta'] = meta
                n.meta['type'] = type(result)
            return result

    def infer(self):
        return super().run()

    # registered shape infer functions incompatible with fake tensor mode
    def one_hot(self, n: Node):
        with self._mode:
            return torch.empty(
                size=(
                    n.args[0].meta["tensor_meta"].shape[0], 
                    n.kwargs["num_classes"]
                ), dtype=self.dtype)


class Decomposition(unittest.TestCase):

    def __init__(self, methodName: str = "runTest") -> None:
        super().__init__(methodName)
        # Change to True to visualize the graph before and after decomposition
        self.visualize = True

    # Helper function for launching test
    def util_test_decomposition(self, cls, inputs):
        ## model instances
        model = cls()
        model_reference = cls()

        symbolic_traced: torch.fx.GraphModule = symbolic_trace(model)
        ShapeProp(symbolic_traced).propagate(*inputs)

        # Get snake case name
        name = cls.__name__
        words = re.findall(r'[A-Z][a-z0-9]*', name)
        snake_case_name = '_'.join(words).lower()

        if self.visualize:
            pass_print_graph(symbolic_traced, f"./{snake_case_name}.svg")
        
        pass_composed_op_breakdown(symbolic_traced, symbolic_traced.graph)
        symbolic_traced.recompile()

        if self.visualize:
            pass_print_graph(symbolic_traced, f"./{snake_case_name}_decomposed.svg")
        
        out = symbolic_traced(*inputs)
        ref = model_reference(*inputs)
        if isinstance(out, tuple):
            for o, r in zip(out, ref):
                self.assertTrue(torch.allclose(o, r, rtol=5e-2))
        else:
            self.assertTrue(torch.allclose(out, ref, rtol=5e-2))

    # decomposition of _log_softmax = aten.log(aten.softmax)
    def test_log_softmax_forward(self):
        # Model
        class LogSoftmaxForward(torch.nn.Module):
            def __init__(self) -> None:
                super().__init__()
            
            def forward(self, input):
                return torch.ops.aten._log_softmax(input, 1, False)
        
        # inputs
        x = torch.randn((512, 1024), dtype=torch.float16, device="cuda")

        self.util_test_decomposition(LogSoftmaxForward, [x,])
    
    def test_nll_loss_backward(self):
        # Model
        class NllLossBackward(torch.nn.Module):
            def __init__(self) -> None:
                super().__init__()
            
            def forward(
                    self, tangents, x, target):
                #
                log_softmax = torch.ops.aten._log_softmax(x, 1, False)
                nll_loss_forward = torch.ops.aten.nll_loss_forward(log_softmax, target, None, 2, -1)
                loss = nll_loss_forward[0]
                total_weight = nll_loss_forward[1]
                nll_loss_backward = torch.ops.aten.nll_loss_backward(
                    tangents, log_softmax, target, None, 
                    2, -1, total_weight)
                _log_softmax_backward_data = torch.ops.aten._log_softmax_backward_data(nll_loss_backward, log_softmax, 1, torch.float16)
                return loss, _log_softmax_backward_data
        
        # Inputs
        tangents = torch.randn((1,), dtype=torch.float16, device="cuda")
        x = torch.randn((512, 1024), dtype=torch.float16, device='cuda')
        target = torch.randint(low=0, high=1023, size=(512, ), dtype=torch.int64, device="cuda")

        self.util_test_decomposition(NllLossBackward, [tangents, x, target])
    
    def test_rsub(self):
        # Model
        class Rsub(torch.nn.Module):
            def __init__(self) -> None:
                super().__init__()
            
            def forward(self, x, y):
                return torch.ops.aten.rsub(x, y)
        
        # Inputs
        x = torch.randn((512, 1024), dtype=torch.float16, device='cuda')
        y = torch.randn((512, 1024), dtype=torch.float16, device='cuda')

        self.util_test_decomposition(Rsub, [x, y])
    
    def test_native_dropout_backward(self):
        # Model
        class NativeDropoutBackward(torch.nn.Module):
            def __init__(self) -> None:
                super().__init__()
            
            def forward(self, grad_y, mask):
                return torch.ops.aten.native_dropout_backward(grad_y, mask, 2.)
        
        # Inputs
        grad_y = torch.randn((512, 16, 1024), dtype=torch.float16, device='cuda')
        mask = torch.rand((512, 16, 1024), device='cuda') < 0.5

        self.util_test_decomposition(NativeDropoutBackward, [grad_y, mask])
    
    def test_threshold_backward(self):
        # Model
        class ThresholdBackward(torch.nn.Module):
            def __init__(self) -> None:
                super().__init__()
            
            def forward(self, grad_y, threshold_output):
                return torch.ops.aten.threshold_backward(grad_y, threshold_output, 0)
        
        # Inputs
        grad_y = torch.randn((512, 1024), dtype=torch.float16, device='cuda')
        threshold_output = torch.ops.aten.relu(torch.randn_like(grad_y))

        self.util_test_decomposition(ThresholdBackward, [grad_y, threshold_output])
        
    def test_addmm(self):
        # Model
        class Addmm(torch.nn.Module):
            def __init__(self) -> None:
                super().__init__()
            
            def forward(self, bias, lhs, rhs):
                return torch.ops.aten.addmm(bias, lhs, rhs)
        
        # Inputs
        lhs = torch.randn((16, 64), dtype=torch.float16, device='cuda')
        rhs = torch.randn((64, 16), dtype=torch.float16, device='cuda')
        bias = torch.randn((16,), dtype=torch.float16, device='cuda')

        self.util_test_decomposition(Addmm, [bias, lhs, rhs])
    
    def test_log_softmax_backward_data(self):
        # Model
        class LogSoftmaxBackwardData(torch.nn.Module):
            def __init__(self) -> None:
                super().__init__()
            
            def forward(self, grad_y, softmax):
                log_softmax = torch.ops.aten.log(softmax)
                return log_softmax, torch.ops.aten._log_softmax_backward_data(
                    grad_y, log_softmax, -1, torch.float32)
        
        # Inputs
        grad_y = torch.randn((2, 16, 24), dtype=torch.float32, device='cuda')
        x = torch.randn_like(grad_y)
        softmax = torch.ops.aten._softmax(x, -1, False)

        self.util_test_decomposition(LogSoftmaxBackwardData, [grad_y, softmax])


if __name__ == '__main__':
    unittest.main()