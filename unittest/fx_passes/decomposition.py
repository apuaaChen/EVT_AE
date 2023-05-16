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
    # decomposition of _log_softmax = aten.log(aten.softmax)
    def log_softmax_forward(self):
        # model
        class LogSoftmaxForward(torch.nn.Module):
            def __init__(self) -> None:
                super().__init__()
            
            def forward(self, input):
                return torch.ops.aten._log_softmax(input, 1, False)
        
        ## model instances
        model = LogSoftmaxForward()
        model_reference = LogSoftmaxForward()

        ## run the compiler pass
        symbolic_traced : torch.fx.GraphModule = symbolic_trace(model)
        x = torch.randn((512, 1024), dtype=torch.float16, device="cuda")

        ShapeProp(symbolic_traced).propagate(x)

        pass_print_graph(symbolic_traced, "./log_softmax_fp.svg")
        def pattern(input, dim, half2float):
            return torch.ops.aten._log_softmax(input, dim, half2float)
        
        def replacement(input, dim, half2float):
            softmax = torch.ops.aten._softmax(input, dim, half2float)
            return torch.ops.aten.log(softmax)

        # decomposition
        subgraph_rewriter.replace_pattern(symbolic_traced, pattern, replacement)
        # pass_infer_shape(symbolic_traced)
        FakeTensorInfer(symbolic_traced).infer()
        
        symbolic_traced.recompile()
        pass_print_graph(symbolic_traced, "./log_softmax_fp_optimized.svg")

        out = symbolic_traced(x)
        ref = model_reference(x)

        self.assertTrue(torch.allclose(out, ref, rtol=5e-2))
    
    def nll_loss_backward(self):
        # model
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
        
        ## model instance
        model = NllLossBackward()
        model_reference = NllLossBackward()

        ## run the compiler pass
        symbolic_traced : torch.fx.GraphModule = symbolic_trace(model)
        tangents = torch.randn((1,), dtype=torch.float16, device="cuda")
        x = torch.randn((512, 1024), dtype=torch.float16, device='cuda')
        target = torch.randint(low=0, high=1023, size=(512, ), dtype=torch.int64, device="cuda")
        
        ShapeProp(symbolic_traced).propagate(tangents, x, target)
        pass_print_graph(symbolic_traced, "./nll_bp_fp.svg")

        pass_composed_op_breakdown(symbolic_traced, symbolic_traced.graph)

        symbolic_traced.recompile()
        pass_print_graph(symbolic_traced, "./nll_bp_optimized.svg")

        out = symbolic_traced(tangents, x, target)
        ref = model_reference(tangents, x, target)

        self.assertTrue(torch.allclose(out[1], ref[1], rtol=5e-2))
    
    def test_rsub(self):
        # model
        class Rsub(torch.nn.Module):
            def __init__(self) -> None:
                super().__init__()
            
            def forward(self, x, y):
                return torch.ops.aten.rsub(x, y)
        
        ## model instance
        model = Rsub()
        model_reference = Rsub()

        ## run the compiler pass
        symbolic_traced : torch.fx.GraphModule = symbolic_trace(model)
        x = torch.randn((512, 1024), dtype=torch.float16, device='cuda')
        y = torch.randn((512, 1024), dtype=torch.float16, device='cuda')
        
        ShapeProp(symbolic_traced).propagate(x, y)
        pass_print_graph(symbolic_traced, "./rsub.svg")

        pass_composed_op_breakdown(symbolic_traced, symbolic_traced.graph)

        symbolic_traced.recompile()
        pass_print_graph(symbolic_traced, "./rsub.svg")

        out = symbolic_traced(x, y)
        ref = model_reference(x, y)

        self.assertTrue(torch.allclose(out, ref, rtol=5e-2))


if __name__ == '__main__':
    unittest.main()