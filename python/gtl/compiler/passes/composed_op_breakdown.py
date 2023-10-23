################################################################################
# Copyright [yyyy] [name of copyright owner]
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
################################################################################
import torch
from gtl.compiler.nodes import *
from torch.fx.passes.infra.pass_base import PassBase, PassResult
from torch.fx.graph_module import GraphModule
from torch.fx import Node, symbolic_trace, Interpreter
from torch.fx.subgraph_rewriter import replace_pattern
from torch._subclasses.fake_tensor import FakeTensorMode, FakeTensor
from typing import Optional
from torch.fx.passes.shape_prop import _extract_tensor_metadata
from torch.fx.experimental.proxy_tensor import py_sym_types
from torch.fx.node import map_aggregate
from gtl.compiler.passes.print_graph import pass_print_graph
import logging
from torch.fx.passes.tools_common import legalize_graph

################################################################################
# Graph-level pass to break down log softmax, nll_loss_forward, nll_loss_backward, 
# _log_softmax_backward_data
################################################################################
# TODO: the general substitution doesn't work for these cases

class nhwcWgradOnly:
    __name__ = "torch_nhwc_wgrad"
    def __init__(self) -> None:
        self.stream = None

    def __call__(self, grad_output, input, weight, bias_sizes_opt, stride, padding, dilation, transposed, output_padding, groups, output_mask):
        default_stream = torch.cuda.current_stream()
        if self.stream is not None:
            self.stream.wait_stream(default_stream)
            stream = self.stream
        else:
            stream = torch.cuda.current_stream()
        with torch.cuda.stream(stream):
            k, c, s, r = weight.size()
            weight_permuted = weight.view(k, r, s, c).permute(0, 3, 1, 2)
            weight_grad = torch.ops.aten.convolution_backward(grad_output, input, weight_permuted, bias_sizes_opt, stride, padding, dilation, transposed, output_padding, groups, output_mask)[1]
            weight_grad = weight_grad.permute(0, 2, 3, 1).view(k, c, r, s)

            if self.stream is not None:
                grad_output.record_stream(self.stream)
                input.record_stream(self.stream)
        return weight_grad

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
        # if mode is None:
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
                try:
                    if isinstance(meta, TensorMetadata):
                        return torch.empty(size=meta.shape, dtype=meta.dtype, device="meta")
                    elif isinstance(meta, tuple) and isinstance(meta[0], TensorMetadata):
                        return (torch.empty(size=m.shape, dtype=m.dtype, device="meta") for m in meta)
                except:
                    logging.DEBUG(f"unkown meta data {meta}")
                    print(meta)
                    exit()
                    return
        else:
            op_name = '_' + str(n.target).split(sep='.')[-1]
            if hasattr(self, op_name):
                result = getattr(self, op_name)(n)
            else:
                result = super().run_node(n)
                
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
    def _one_hot(self, n: Node):
        with self._mode:
            return torch.empty(
                size=(
                    n.args[0].meta["tensor_meta"].shape[0], 
                    n.kwargs["num_classes"]
                ), dtype=self.dtype, device="meta")
            

class Decomposition(PassBase):
    """
    GTL pass that decompose operations to registered patterns
    """
    def __init__(self) -> None:
        super().__init__()
        self.modified = False

        # concept: dict{key, list of tuples of function}
        self.pattern_replacement = {}
    
    def call(self, graph_module: GraphModule) -> PassResult:

        for key in self.pattern_replacement.keys():
            matches = replace_pattern(
                graph_module, self.pattern_replacement[key][0],
                self.pattern_replacement[key][1])
            assert len(matches)

        graph_module.recompile()

        return PassResult(graph_module, True)

    def requires(self, graph_module: GraphModule) -> None:
        """
        This function will be called before the pass is run and will check that
        the given graph module contains the preconditions needed to run the
        pass. 

        In the decomposition pass, it generates the pattern-replacement set based
        on the input graph module

        It also serves as the interface for decomposition registration

        Args:
            graph_module: The graph module we will run checks on
        """
        graph = graph_module.graph
        for node in graph.nodes:
            visitor_name = "requires_" + str(node.target).split(sep='.')[-1]
            if hasattr(self, visitor_name):
                getattr(self, visitor_name)(node)
    
    def ensures(self, graph_module: GraphModule) -> None:
        """
        This function will be called after the pass is run and will check that
        the given graph module contains the postconditions needed to run the
        pass. It is not required to implement this function.

        In the decomposition pass, it runs the shape infer pass as the metadata
        are lost during the subgraph substitutions

        Args:
            graph_module: The graph module we will run checks on
        """
        legalize_graph(graph_module)
        graph_module.graph.lint()
        # print(graph_module.code)
        FakeTensorInfer(graph_module).infer()
    
    def requires__log_softmax(self, node: torch.fx.Node):
        # Each registration requires a special key to avoid duplications
        # It usually starts with the function name and then a set of specified
        # arguments
        key = "aten._log_softmax"
        if key in self.pattern_replacement.keys():
            return

        # then a pair of pattern & replacement definition
        def pattern(input, dim, half2float):
            return torch.ops.aten._log_softmax(input, dim, half2float)
        
        def replacement(input, dim, half2float):
            softmax = torch.ops.aten._softmax(input, dim, half2float)
            return torch.ops.aten.log(softmax)
        
        # inject the pattern
        self.pattern_replacement[key] = (pattern, replacement)
    
    def requires_rsub(self, node: torch.fx.Node):
        key = "aten.rsub"
        if key in self.pattern_replacement.keys():
            return

        def pattern(x, y):
            return torch.ops.aten.rsub(x, y)

        def replacement(x, y):
            return torch.ops.aten.sub(y, x)
        
        # inject the pattern
        self.pattern_replacement[key] = (pattern, replacement)

        # node.target = torch.ops.aten.sub
        # node.args = (node.args[1], node.args[0])

    def requires_nll_loss_backward(self, node: torch.fx.Node):

        _, _log_softmax, _, _weight, _reduction_id, _ignore_idx, _ = node.args

        num_classes = _log_softmax.meta["tensor_meta"].shape[-1]

        if _weight is not None: return

        if _reduction_id == 1:
            reduction = "mean"
        elif _reduction_id == 2:
            reduction = "sum"
        else:
            return

        key = f"aten.nll_loss_backward_{reduction}_{_ignore_idx}"

        if key in self.pattern_replacement.keys():
            return
        
        def pattern(tangents, log_softmax, target, total_weight):
            return torch.ops.aten.nll_loss_backward(
                tangents, log_softmax, target, None, 
                _reduction_id, _ignore_idx, total_weight)
        
        if reduction == "sum":
            def replacement(tangents, log_softmax, target, total_weight):
                one_hot = torch.ops.aten.one_hot(target, num_classes=num_classes)
                neg = torch.ops.aten.neg(one_hot)
                mul = torch.ops.aten.mul(neg, tangents)

                ne = torch.ops.aten.ne(target, -1)
                unsqueeze = torch.ops.aten.unsqueeze(ne, 1)
                mul_mask = torch.ops.aten.mul(mul, unsqueeze)
                return mul_mask
        elif reduction == "mean":
            # TODO: more accurate number of sample is required based on the 
            # mask, this may lead to incorrect result
            num_sample = _log_softmax.meta["tensor_meta"].shape[0]
            def replacement(tangents, log_softmax, target, total_weight):
                one_hot = torch.ops.aten.one_hot(target, num_classes=num_classes)
                neg = torch.ops.aten.neg(one_hot)
                mul = torch.ops.aten.mul(neg, tangents)

                ne = torch.ops.aten.ne(target, -1)
                unsqueeze = torch.ops.aten.unsqueeze(ne, 1)
                mul_mask = torch.ops.aten.mul(mul, unsqueeze)
                mean = torch.ops.aten.div(mul_mask, num_sample)
                return mean
        
        self.pattern_replacement[key] = (pattern, replacement)

    def requires_native_dropout_backward(self, node: torch.fx.Node):
        key = "aten.native_dropout_backward"

        if key in self.pattern_replacement.keys():
            return

        def pattern(grad_y, mask, scale):
            return torch.ops.aten.native_dropout_backward(grad_y, mask, scale)
        
        def replacement(grad_y, mask, scale):
            grad_x = torch.ops.aten.mul(grad_y, mask)
            grad_x = torch.ops.aten.mul(grad_x, scale)
            return grad_x
        
        self.pattern_replacement[key] = (pattern, replacement)

    def requires_threshold_backward(self, node: torch.fx.Node):
        threshold = node.args[2]

        key = f"aten.threshold_backward_{threshold}"

        if key in self.pattern_replacement.keys():
            return

        def pattern(grad_Y, threshold_output):
            return torch.ops.aten.threshold_backward(grad_Y, threshold_output, threshold)
        
        def replacement(grad_Y, threshold_output):
            ne = torch.ops.aten.ne(threshold_output, threshold)
            return torch.ops.aten.mul(grad_Y, ne)
        
        self.pattern_replacement[key] = (pattern, replacement)

    def requires_addmm(self, node: torch.fx.Node):
        key = "aten.addmm"
        if key in self.pattern_replacement.keys():
            return
        
        def pattern(bias, lhs, rhs):
            return torch.ops.aten.addmm(bias, lhs, rhs)
    
        def replacement(bias, lhs, rhs):
            mm = torch.ops.aten.mm(lhs, rhs)
            return torch.ops.aten.add(mm, bias)

        self.pattern_replacement[key] = (pattern, replacement)
    
    def requires__log_softmax_backward_data(self, node: torch.fx.Node):
        key = f"aten._log_softmax_backward_data"

        if key in self.pattern_replacement.keys():
            return
        
        def pattern(grad_y, softmax, dim, dtype):
            log_softmax = torch.ops.aten.log(softmax)
            return log_softmax, torch.ops.aten._log_softmax_backward_data(grad_y, log_softmax, dim, dtype)
        
        def replacement(grad_y, softmax, dim, dtype):
            log_softmax = torch.ops.aten.log(softmax)
            sum = torch.ops.aten.sum(grad_y, dim)
            unsqueeze = torch.ops.aten.unsqueeze(sum, dim)
            mul = torch.ops.aten.mul(unsqueeze, softmax)
            sub = torch.ops.aten.sub(grad_y, mul)
            return log_softmax, sub

        self.pattern_replacement[key] = (pattern, replacement)

"""
    # TODO: the convolution backward is more tricky to construct the arg list
    def requires_convolution_backward(self, node: torch.fx.Node):
        # _grad_output, _input, _weight, _bias_sizes_opt, _stride, _padding, _dilation, _transposed, _output_padding, _groups, _output_mask = node.args
        _output_mask = node.args[-1]
        # n, c, h, w = _input.meta["tensor_meta"].shape
        # k, c_, r, s = _weight.meta["tensor_meta"].shape
        # assert c == c_
        if (not _output_mask[0]) and _output_mask[1] and (not _output_mask[2]):
            key = f"aten.convolution_backward_FTF"
            if key in self.pattern_replacement.keys():
                return

            def nhwc_wgrad_only(grad_output, input, weight, bias_sizes_opt, stride, padding, dilation, transposed, output_padding, groups):
                k, c, s, r = weight.size()
                weight_permuted = weight.view(k, r, s, c).permute(0, 3, 1, 2)
                weight_grad = torch.ops.aten.convolution_backward(grad_output, input, weight_permuted, bias_sizes_opt, stride, padding, dilation, transposed, output_padding, groups, [False, True, False])[1]
                weight_grad = weight_grad.permute(0, 2, 3, 1).view(k, c, r, s)
                return [None, weight_grad, None]
            
            def pattern(grad_output, input, weight, bias_sizes_opt, stride, padding, 
                        dilation, transposed, output_padding, groups):
                return torch.ops.aten.convolution_backward(
                    grad_output, input, weight, bias_sizes_opt, stride, padding, 
                    dilation, transposed, output_padding, groups, [False, True, False])

            def replacement(grad_output, input, weight, bias_sizes_opt, stride, padding, 
                        dilation, transposed, output_padding, groups):
                return nhwc_wgrad_only(grad_output, input, weight, bias_sizes_opt, stride, padding, dilation, transposed, output_padding, groups)
            
            self.pattern_replacement[key] = (pattern, replacement)
            

    # TODO: it may lead to several issues, we keep the old way temporarily
"""
        

def pass_composed_op_breakdown(module, graph, disabled_list=[]):
    decompose = Decomposition()

    module, modified = decompose(module)
    module.recompile()

    graph = module.graph
    

    softmax_node = None
    for node in graph.nodes:
        if node.op == "call_function":
            if node.target in disabled_list: continue
            if node.target == torch.ops.aten.convolution_backward:
                grad_output, input, weight, bias_sizes_opt, stride, padding, dilation, transposed, output_padding, groups, output_mask = node.args
                n, c, h, w = input.meta["tensor_meta"].shape
                k, c_, r, s = weight.meta["tensor_meta"].shape
                assert c == c_
                # inject dgrad node 
                if output_mask[0] and output_mask[1]:
                    dgrad_output = list(node.users.keys())[0]
                    wgrad_output = list(node.users.keys())[1]
                elif (not output_mask[0]) and output_mask[1]:
                    dgrad_output = None
                    wgrad_output = list(node.users.keys())[0]
                elif output_mask[0] and (not output_mask[1]):
                    dgrad_output = list(node.users.keys())[0]
                    wgrad_output = None
                else:
                    raise NotImplementedError
                
                if dgrad_output is not None:
                    dgrad_args = [
                        (n, c, h, w), weight, grad_output, stride, padding, dilation, groups
                    ]
                    graph.inserting_after(node)
                    dgrad_node = graph.call_function(torch.nn.grad.conv2d_input, args=tuple(dgrad_args))
                    dgrad_node.meta["tensor_meta"] = dgrad_output.meta["tensor_meta"]._replace()
                    dgrad_output.replace_all_uses_with(dgrad_node)
                
                if wgrad_output is not None:
                    if dgrad_output is not None:
                        torch_wgrad_func = nhwcWgradOnly()

                        graph.inserting_after(node)
                        wgrad_node = graph.call_function(torch_wgrad_func, args=(grad_output, input, weight, bias_sizes_opt, stride, padding, dilation, transposed, output_padding, groups, [False, True, False]))
                        wgrad_node.meta["tensor_meta"] = wgrad_output.meta["tensor_meta"]._replace()
                        wgrad_output.replace_all_uses_with(wgrad_node)
                    else:
                        # transfer the function to nhwc
                        def nhwc_wgrad_only(grad_output, input, weight, bias_sizes_opt, stride, padding, dilation, transposed, output_padding, groups, output_mask):
                            k, c, s, r = weight.size()
                            weight_permuted = weight.view(k, r, s, c).permute(0, 3, 1, 2)
                            weight_grad = torch.ops.aten.convolution_backward(grad_output, input, weight_permuted, bias_sizes_opt, stride, padding, dilation, transposed, output_padding, groups, output_mask)[1]
                            weight_grad = weight_grad.permute(0, 2, 3, 1).view(k, c, r, s)
                            return [None, weight_grad, None]
                        
                        node.target = nhwc_wgrad_only
                
    graph.eliminate_dead_code()
    graph.lint()
   