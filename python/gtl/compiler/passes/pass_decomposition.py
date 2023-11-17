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
from torch.fx import Node
from torch.fx.subgraph_rewriter import replace_pattern_with_filters
from torch._subclasses.fake_tensor import FakeTensorMode, FakeTensor
from typing import Optional
from torch.fx.passes.shape_prop import _extract_tensor_metadata
from torch.fx.experimental.proxy_tensor import py_sym_types
from torch.fx.node import map_aggregate
from torch.fx.passes.tools_common import legalize_graph
from torch.fx.passes.fake_tensor_prop import FakeTensorProp
from gtl.compiler.passes.pass_fake_shape_infer import pass_fake_shape_infer
import operator

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

@torch.fx.wrap
def spmm(adj_matrix, feature):
    # return feature
    return torch.ops.aten.mm(adj_matrix, feature)

@torch.fx.wrap
def convolution_forward_channel_last(input_channel_last, weight_channel_last, stride, padding, dilation, transposed, output_padding, groups):
    # K, C, R, S = weight_channel_last.size()
    # weight_channel_last = torch.ops.aten.view(weight_channel_last.contiguous(), [K, R, S, C])

    input = torch.ops.aten.permute(input_channel_last.contiguous(), [0, 3, 1, 2]).contiguous()
    weight = torch.ops.aten.permute(weight_channel_last.contiguous(), [0, 3, 1, 2]).contiguous()
    output = torch.ops.aten.convolution(input, weight, None, stride, padding, dilation, transposed, output_padding, groups)
    output_channel_last = torch.ops.aten.permute(output, [0, 2, 3, 1]).contiguous()
    return output_channel_last

@torch.fx.wrap
def convolution_backward_data_channel_last(grad_output_channel_last, input_channel_last, weight_channel_last, stride, padding, dilation, transpose, output_padding, groups):
    # K, C, R, S = weight_channel_last.size()
    # weight_channel_last = torch.ops.aten.view(weight_channel_last.contiguous(), [K, R, S, C])

    grad_output = torch.ops.aten.permute(grad_output_channel_last.contiguous(), [0, 3, 1, 2]).contiguous()
    input = torch.ops.aten.permute(input_channel_last.contiguous(), [0, 3, 1, 2]).contiguous()
    weight = torch.ops.aten.permute(weight_channel_last.contiguous(), [0, 3, 1, 2]).contiguous()
    grad_inputs = torch.ops.aten.convolution_backward(
        grad_output, input, weight, [0], stride, padding, dilation, transpose, output_padding, groups, [True, False, False]
    )
    grad_input = operator.getitem(grad_inputs, 0)#.contiguous()
    grad_input_channel_last = torch.ops.aten.permute(grad_input, [0, 2, 3, 1]).contiguous()
    return grad_input_channel_last

@torch.fx.wrap
def convolution_backward_weight_channel_last(grad_output_channel_last, input_channel_last, weight_channel_last, stride, padding, dilation, transpose, output_padding, groups):
    # K, C, R, S = weight_channel_last.size()
    # weight_channel_last = torch.ops.aten.view(weight_channel_last.contiguous(), [K, R, S, C])

    grad_output = torch.ops.aten.permute(grad_output_channel_last.contiguous(), [0, 3, 1, 2]).contiguous()
    input = torch.ops.aten.permute(input_channel_last.contiguous(), [0, 3, 1, 2]).contiguous()
    weight = torch.ops.aten.permute(weight_channel_last.contiguous(), [0, 3, 1, 2]).contiguous()
    grad_weights = torch.ops.aten.convolution_backward(
        grad_output, input, weight, [0], stride, padding, dilation, transpose, output_padding, groups, [False, True, False]
    )
    grad_weight = operator.getitem(grad_weights, 1)
    grad_weight_channel_last = torch.ops.aten.permute(grad_weight, [0, 2, 3, 1]).contiguous()#.view(K, C, R, S)#.contiguous().view(K, C, R, S)
    return grad_weight_channel_last

@torch.fx.wrap
def batch_norm_stat(input, reduction_factor):
    mean_vec = torch.ops.aten.mean(input, [0, 2, 3], dtype=torch.float32, keepdim=True)
    input_square = torch.ops.aten.square(input)
    mean_x2_vec = torch.ops.aten.mean(input_square, [0, 2, 3], dtype=torch.float32, keepdim=True)
    return mean_vec, mean_x2_vec

@torch.fx.wrap
def batch_norm_elemt(input, weight, bias, mean_vec, mean_x2_vec, eps, K):
    var = mean_x2_vec - mean_vec * mean_vec
    rstd = torch.ops.aten.rsqrt(var + eps)
    saved_mean = torch.ops.aten.squeeze(mean_vec)
    saved_rstd = torch.ops.aten.squeeze(rstd)

    gamma = torch.ops.aten.view(weight, [1, K, 1, 1])
    beta = torch.ops.aten.view(bias, [1, K, 1, 1])
    sub_mean = torch.ops.aten.sub(input, mean_vec)
    mul_gamma = torch.ops.aten.mul(gamma, sub_mean)
    mul_rstd = torch.ops.aten.mul(mul_gamma, rstd)
    add_bias = torch.ops.aten.add(mul_rstd, beta)
    output = torch.ops.aten.to(add_bias, torch.float16)
    
    return output.contiguous(), saved_mean, saved_rstd

@torch.fx.wrap
def batch_norm_backward_stat(y_grad, input, saved_mean, saved_rstd, K):
    saved_mean = torch.ops.aten.view(saved_mean, [1, K, 1, 1])
    saved_rstd = torch.ops.aten.view(saved_rstd, [1, K, 1, 1])

    sum_grad_y = torch.ops.aten.sum(y_grad, [0, 2, 3], dtype=torch.float32, keepdim=True)
    sub_saved_mean = torch.ops.aten.sub(input, saved_mean)
    x_hat = torch.ops.aten.mul(sub_saved_mean, saved_rstd)
    mul_x_hat = torch.ops.aten.mul(y_grad, x_hat)
    sum_grad_y_xhat = torch.ops.aten.sum(mul_x_hat, [0, 2, 3], dtype=torch.float32, keepdim=True)
    return sum_grad_y, sum_grad_y_xhat

@torch.fx.wrap
def batch_norm_backward_elemt(y_grad, input, saved_mean, saved_rstd, K, sum_grad_y, sum_grad_y_xhat, reduction_factor, gamma):
    saved_mean = torch.ops.aten.view(saved_mean, [1, K, 1, 1])
    saved_rstd = torch.ops.aten.view(saved_rstd, [1, K, 1, 1])
    sub_saved_mean = torch.ops.aten.sub(input, saved_mean)
    x_hat = torch.ops.aten.mul(sub_saved_mean, saved_rstd)
    gamma = torch.ops.aten.view(gamma, [1, K, 1, 1])

    mean_grad_y_xhat = torch.ops.aten.mul(sum_grad_y_xhat, reduction_factor)
    mean_grad_y = torch.ops.aten.mul(sum_grad_y, reduction_factor)

    var_grad_mean = -mean_grad_y_xhat * x_hat * saved_rstd * gamma
    x_grad = (y_grad - mean_grad_y) * gamma * saved_rstd + var_grad_mean
    x_grad = torch.ops.aten.to(x_grad, torch.float16)
    gamma_grad = torch.ops.aten.to(torch.ops.aten.squeeze(sum_grad_y_xhat), torch.float16)
    beta_grad = torch.ops.aten.to(torch.ops.aten.squeeze(sum_grad_y), torch.float16)
    return x_grad, gamma_grad, beta_grad


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

        # Seems that multiple round is expected until there are no more matchs
        num_matches = 1
        while (num_matches):
            for key in self.pattern_replacement.keys():
                matches = replace_pattern_with_filters(
                    graph_module, 
                    self.pattern_replacement[key][0],
                    self.pattern_replacement[key][1],
                    self.pattern_replacement[key][2] # filters
                    )
                num_matches = len(matches)

        # graph_module.recompile()

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
        pass_fake_shape_infer(graph_module, graph_module.graph)
    
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
        self.pattern_replacement[key] = (pattern, replacement, None)
    
    def requires_rsub(self, node: torch.fx.Node):
        key = "aten.rsub"
        if key in self.pattern_replacement.keys():
            return

        def pattern(x, y):
            return torch.ops.aten.rsub(x, y)

        def replacement(x, y):
            return torch.ops.aten.sub(y, x)
        
        # inject the pattern
        self.pattern_replacement[key] = (pattern, replacement, None)

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

        key = f"aten.nll_loss_backward_{reduction}_{_ignore_idx}_{num_classes}"

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
                unsqueeze = torch.ops.aten.unsqueeze(ne, -1)
                mul_mask = torch.ops.aten.mul(mul, unsqueeze)
                mean = torch.ops.aten.div(mul_mask, num_sample)
                return mean
        
        def filter(match, original_graph, pattern_graph):
            _num_cls = match.nodes_map[match.anchors[0]].meta["tensor_meta"].shape[-1]
            return _num_cls == num_classes
        
        self.pattern_replacement[key] = (pattern, replacement, [filter])

    def requires_native_dropout(self, node: torch.fx.Node):
        prob = node.args[1]
        scaling_factor = 1./(1-prob)

        key = f"aten.native_dropout_{prob}"

        if key in self.pattern_replacement.keys():
            return
        
        def pattern(x):
            dropout = torch.ops.aten.native_dropout(x, prob, True)
            output = operator.getitem(dropout, 0)
            mask = operator.getitem(dropout, 1)
            return output, mask
        
        def replacement(x):
            mask = torch.ops.aten.ge(torch.rand_like(x), prob)
            output = torch.ops.aten.mul(x, mask)
            scaled_output = torch.ops.aten.mul(output, scaling_factor)
            return scaled_output, mask
        
        def filter(match, original_graph, pattern_graph):
            for node in match.nodes_map:
                if node.target == torch.ops.aten.native_dropout:
                    if node.args[1] == prob:
                        return True
            
            return False
        
        self.pattern_replacement[key] = (pattern, replacement, [filter])

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
        
        self.pattern_replacement[key] = (pattern, replacement, None)

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
        
        self.pattern_replacement[key] = (pattern, replacement, None)

    def requires_addmm(self, node: torch.fx.Node):
        key = "aten.addmm"
        if key in self.pattern_replacement.keys():
            return
        
        def pattern(bias, lhs, rhs):
            return torch.ops.aten.addmm(bias, lhs, rhs)
    
        def replacement(bias, lhs, rhs):
            mm = torch.ops.aten.mm(lhs, rhs)
            return torch.ops.aten.add(mm, bias)

        self.pattern_replacement[key] = (pattern, replacement, None)
    
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

        self.pattern_replacement[key] = (pattern, replacement, None)
    
    def requires_t(self, node: torch.fx.Node):
        # Unify transposes to permutations
        key = f"aten.t"
        assert len(list(node.meta['tensor_meta'].shape)) == 2
        if key in self.pattern_replacement:
            return
        
        def pattern(x):
            return torch.ops.aten.t(x)
        
        def replacement(x):
            return torch.ops.aten.permute(x, [1, 0])
        
        self.pattern_replacement[key] = (pattern, replacement, None)
    
    def requires__unsafe_view(self, node: torch.fx.Node):
        # Unify _unsafe_view to view
        key = f"aten._unsafe_view"
        if key in self.pattern_replacement:
            return
        
        def pattern(x, shape):
            return torch.ops.aten._unsafe_view(x, shape)
        
        def replacement(x, shape):
            return torch.ops.aten.view(x, shape)
        
        self.pattern_replacement[key] = (pattern, replacement, None)
    
    def requires_transpose(self, node: torch.fx.Node):
        # Unify tranpose to permute
        shape = list(node.meta['tensor_meta'].shape)
        key = f"aten.transpose_{len(shape)}_{node.args[1]}_{node.args[2]}"
        if key in self.pattern_replacement:
            return

        dim = [i for i in range(len(shape))]
        dim[node.args[1]] = node.args[2]
        dim[node.args[2]] = node.args[1]
        
        def pattern(x, idx1, idx2):
            return torch.ops.aten.transpose(x, idx1, idx2)
        
        def replacement(x, idx1, idx2):
            return torch.ops.aten.permute(x, dim)
        
        def filter(match, original_graph, pattern_graph):
            node = match.nodes_map[match.anchors[0]]
            shape = list(node.meta['tensor_meta'].shape)
            _dim = [i for i in range(len(shape))]
            _dim[node.args[1]] = node.args[2]
            _dim[node.args[2]] = node.args[1]
            return _dim == dim
        
        self.pattern_replacement[key] = (pattern, replacement, [filter,])
    
    def requires_native_layer_norm(self, node: torch.fx.Node):
        key = f"aten.native_layer_norm"

        if node.args[2] is None:
            return False
        if node.args[3] is None:
            return False
        
        if key in self.pattern_replacement:
            return
        
        def pattern(input, normalized_shape, gamma, beta, eps):
            output, mean, rstd = torch.ops.aten.native_layer_norm(
                input, normalized_shape, gamma, beta, eps
            )

            return output, mean, rstd
        
        def replacement(input, normalized_shape, gamma, beta, eps):
            _output, mean, rstd = torch.ops.aten.native_layer_norm(
                input, normalized_shape, None, None, eps
            )
            output = torch.ops.aten.add(torch.ops.aten.mul(_output, gamma), beta)

            return output, mean, rstd

        def filter(match, original_graph, pattern_graph):
            layer_norm_node = None
            for node_ in match.nodes_map.keys():
                if node_.target == torch.ops.aten.native_layer_norm:
                    layer_norm_node = match.nodes_map[node_]
                    break
            
            if layer_norm_node.args[2] is None:
                return False
            if layer_norm_node.args[3] is None:
                return False
            
            return True
        
        self.pattern_replacement[key] = (pattern, replacement, [filter,])
    
    def requires_native_layer_norm_backward(self, node: torch.fx.Node):
        input_shape = node.args[1].meta["tensor_meta"].shape
        input_ndim = len(input_shape)
        normalized_shape = node.args[2]
        axis = input_ndim - len(normalized_shape)
        outer_dim_indices = list(range(0, axis))
        output_mask = node.args[-1]

        if (not output_mask[1]) or (not output_mask[2]):
            return

        key = f"aten.native_layer_norm_backward_{axis}"
        if key in self.pattern_replacement:
            return
        
        def pattern(input, normalized_shape, gamma, beta, grad_out, output_mask, mean, rstd):
            grad_input, grad_gamma, grad_beta = torch.ops.aten.native_layer_norm_backward(
                grad_out, input, normalized_shape, mean, rstd, gamma, beta,
                output_mask
            )
            return grad_input, grad_gamma, grad_beta
        
        def replacement(input, normalized_shape, gamma, beta, grad_out, output_mask, mean, rstd):
            grad_beta = torch.ops.aten.sum(grad_out, outer_dim_indices)
            sub_mean = torch.ops.aten.sub(input, mean)
            mul = torch.ops.aten.mul(sub_mean, rstd)
            mul_gamma = torch.ops.aten.mul(grad_out, mul)
            grad_gamma = torch.ops.aten.sum(mul_gamma, outer_dim_indices)
            grad_input, _, _ = torch.ops.aten.native_layer_norm_backward(
                grad_out, input, normalized_shape, mean, rstd, gamma, beta,
                [True, False, False]
            )
            return grad_input, grad_gamma, grad_beta

        def filter(match, original_graph, pattern_graph):
            layer_norm_backward_node = None
            for node_ in match.nodes_map.keys():
                if node_.target == torch.ops.aten.native_layer_norm_backward:
                    layer_norm_backward_node = match.nodes_map[node_]
                    break
            
            output_mask = layer_norm_backward_node.args[-1]

            if (not output_mask[1]) or (not output_mask[2]):
                return False
            
            input_shape_ = layer_norm_backward_node.args[1].meta["tensor_meta"].shape
            input_ndim_ = len(input_shape_)
            normalized_shape_ = layer_norm_backward_node.args[2]
            axis_ = input_ndim_ - len(normalized_shape_)
            return axis_ == axis
        
        self.pattern_replacement[key] = (pattern, replacement, [filter,])
    
    def requires_mm(self, node: torch.fx.Node):
        # Check if the mm is actually an spmm
        arg0 = node.args[0]
        if "tensor_meta" not in arg0.meta or arg0.meta["tensor_meta"].stride != (0, 0):
            return
        # Special pattern that apply spmm
        key = f"aten.mm_{node.name}"
        if key in self.pattern_replacement:
            return
        
        def pattern(adj_matrix, feature):
            return torch.ops.aten.mm(adj_matrix, feature)
        
        def replacement(adj_matrix, feature):
            return spmm(adj_matrix, feature)
        
        def filter(match, original_graph, pattern_graph):
            mm_node = None
            for n in match.nodes_map.keys():
                if n.target == torch.ops.aten.mm:
                    mm_node = match.nodes_map[n]
                    break
            return mm_node.name == node.name
            
        self.pattern_replacement[key] = (pattern, replacement, [filter,])

    def requires_convolution(self, node: torch.fx.Node):
        
        key = f"aten.convolution"
        if key in self.pattern_replacement:
            return
    
        def pattern(input, weight, stride, padding, dilation, transposed, output_padding, groups):
            return torch.ops.aten.convolution(input, weight, None, stride, padding, dilation, transposed, output_padding, groups)
        
        def replacement(input, weight, stride, padding, dilation, transposed, output_padding, groups):
            input_channel_last = torch.ops.aten.permute(input, [0, 2, 3, 1])
            weight = torch.ops.aten.permute(weight, [0, 2, 3, 1])
            output_channel_last = convolution_forward_channel_last(input_channel_last, weight, stride, padding, dilation, transposed, output_padding, groups)
            output = torch.ops.aten.permute(output_channel_last, [0, 3, 1, 2])
            return output
        
        self.pattern_replacement[key] = (pattern, replacement, None)
    
    def requires_convolution_backward(self, node: torch.fx.Node):
        
        output_mask = node.args[10]
        if output_mask == [True, True, False]:
            key = f"aten.convolution_backward"
        elif output_mask == [False, True, False]:
            key = f"aten.convolution_backward_weight_only"
        if key in self.pattern_replacement:
            return

        if key == f"aten.convolution_backward":
            def pattern(grad_output, input, weight, stride, padding, dilation, transpose, output_padding, groups):
                convolution_backward = torch.ops.aten.convolution_backward(
                    grad_output, input, weight, [0], stride, padding, dilation, transpose, output_padding, groups, [True, True, False]
                )
                grad_input = operator.getitem(convolution_backward, 0)
                grad_weight = operator.getitem(convolution_backward, 1)
                return grad_input, grad_weight

            def replacement(grad_output, input, weight, stride, padding, dilation, transpose, output_padding, groups):
                grad_output_channel_last = torch.ops.aten.permute(grad_output, [0, 2, 3, 1])
                input_channel_last = torch.ops.aten.permute(input, [0, 2, 3, 1])
                weight = torch.ops.aten.permute(weight, [0, 2, 3, 1])
            
                grad_input_channel_last = convolution_backward_data_channel_last(
                    grad_output_channel_last, input_channel_last, weight, 
                    stride, padding, dilation, transpose, output_padding, groups
                )
                grad_weight = convolution_backward_weight_channel_last(
                    grad_output_channel_last, input_channel_last, weight,
                    stride, padding, dilation, transpose, output_padding, groups
                )
                grad_input = torch.ops.aten.permute(grad_input_channel_last, [0, 3, 1, 2])
                grad_weight = torch.ops.aten.permute(grad_weight, [0, 3, 1, 2])
                return grad_input, grad_weight
        elif key == f"aten.convolution_backward_weight_only":
            def pattern(grad_output, input, weight, stride, padding, dilation, transpose, output_padding, groups):
                convolution_backward = torch.ops.aten.convolution_backward(
                    grad_output, input, weight, [0], stride, padding, dilation, transpose, output_padding, groups, [False, True, False]
                )
                grad_weight = operator.getitem(convolution_backward, 1)
                return grad_weight

            def replacement(grad_output, input, weight, stride, padding, dilation, transpose, output_padding, groups):
                grad_output_channel_last = torch.ops.aten.permute(grad_output, [0, 2, 3, 1])
                input_channel_last = torch.ops.aten.permute(input, [0, 2, 3, 1])
                weight = torch.ops.aten.permute(weight, [0, 2, 3, 1])

                grad_weight = convolution_backward_weight_channel_last(
                    grad_output_channel_last, input_channel_last, weight,
                    stride, padding, dilation, transpose, output_padding, groups
                )
                grad_weight = torch.ops.aten.permute(grad_weight, [0, 3, 1, 2])
                return grad_weight


        self.pattern_replacement[key] = (pattern, replacement, None)

    def requires_no_stats(self, node: torch.fx.Node):
        input = node.args[0]
        momentum = node.args[4]
        N, K, P, Q = list(input.meta["tensor_meta"].shape)
        reduction_length = N * P * Q
        reduction_factor = 1./reduction_length
        key = f"aten._native_batch_norm_legit.no_stats_{node.name}"

        if key in self.pattern_replacement:
            return

        def pattern(input, weight, bias, eps):
            output, mean, invstd = torch.ops.aten._native_batch_norm_legit.no_stats(
                input, weight, bias, True, momentum, eps)
            return output, mean, invstd
        
        def replacement(input, weight, bias, eps):
            mean_vec, mean_x2_vec = batch_norm_stat(input, reduction_factor)
            output, saved_mean, saved_rstd = batch_norm_elemt(input, weight, bias, mean_vec, mean_x2_vec, eps, K)
            return output, saved_mean, saved_rstd
        
        def filter(match, original_graph, pattern_graph):
            bn_node = None
            for n in match.nodes_map.keys():
                if n.target == torch.ops.aten._native_batch_norm_legit.no_stats:
                    bn_node = match.nodes_map[n]
                    break
            return bn_node.name == node.name
            
        self.pattern_replacement[key] = (pattern, replacement, [filter,])
    
    def requires_native_batch_norm_backward(self, node: torch.fx.Node):
        input = node.args[1]
        N, K, P, Q = list(input.meta["tensor_meta"].shape)
        reduction_length = N * P * Q
        reduction_factor = 1./reduction_length
        key = f"torch.ops.aten.native_batch_norm_backward_{node.name}"

        if key in self.pattern_replacement:
            return
        
        def pattern(y_grad, input, gamma, saved_mean, saved_rstd, epilon):
            grad_input, grad_gamma, grad_beta = torch.ops.aten.native_batch_norm_backward(
                y_grad, input, gamma, None, None, saved_mean, saved_rstd, True, epilon, [True, True, True]
            )
            return grad_input, grad_gamma, grad_beta
        
        def replacement(y_grad, input, gamma, saved_mean, saved_rstd, epsilon):
            sum_grad_y, sum_grad_y_xhat = batch_norm_backward_stat(y_grad, input, saved_mean, saved_rstd, K)
            x_grad, gamma_grad, beta_grad = batch_norm_backward_elemt(y_grad, input, saved_mean, saved_rstd, K, sum_grad_y, sum_grad_y_xhat, reduction_factor, gamma)
            return x_grad, gamma_grad, beta_grad
        
        def filter(match, original_graph, pattern_graph):
            bn_node = None
            for n in match.nodes_map.keys():
                if n.target == torch.ops.aten.native_batch_norm_backward:
                    bn_node = match.nodes_map[n]
                    break
            return bn_node.name == node.name
            
        self.pattern_replacement[key] = (pattern, replacement, [filter,])


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
        

def pass_decomposition(module, graph, disabled_list=[]):
    decompose = Decomposition()

    module, modified = decompose(module)
    # module.recompile()

    # graph = module.graph
    
    # TODO: strange it is not working
    # for node in graph.nodes:
    #     if node.op == "call_function":
    #         if node.target in disabled_list: continue
    #         if node.target == torch.ops.aten.convolution_backward:
    #             grad_output, input, weight, bias_sizes_opt, stride, padding, dilation, transposed, output_padding, groups, output_mask = node.args
    #             n, c, h, w = input.meta["tensor_meta"].shape
    #             k, c_, r, s = weight.meta["tensor_meta"].shape
    #             assert c == c_
    #             # inject dgrad node 
    #             if output_mask[0] and output_mask[1]:
    #                 dgrad_output = list(node.users.keys())[0]
    #                 wgrad_output = list(node.users.keys())[1]
    #             elif (not output_mask[0]) and output_mask[1]:
    #                 dgrad_output = None
    #                 wgrad_output = list(node.users.keys())[0]
    #             elif output_mask[0] and (not output_mask[1]):
    #                 dgrad_output = list(node.users.keys())[0]
    #                 wgrad_output = None
    #             else:
    #                 raise NotImplementedError
                
    #             if dgrad_output is not None:
    #                 dgrad_args = [
    #                     (n, c, h, w), weight, grad_output, stride, padding, dilation, groups
    #                 ]
    #                 graph.inserting_after(node)
    #                 dgrad_node = graph.call_function(torch.nn.grad.conv2d_input, args=tuple(dgrad_args))
    #                 dgrad_node.meta["tensor_meta"] = dgrad_output.meta["tensor_meta"]._replace()
    #                 dgrad_output.replace_all_uses_with(dgrad_node)
                
    #             if wgrad_output is not None:
    #                 if dgrad_output is not None:
    #                     torch_wgrad_func = nhwcWgradOnly()

    #                     graph.inserting_after(node)
    #                     wgrad_node = graph.call_function(torch_wgrad_func, args=(grad_output, input, weight, bias_sizes_opt, stride, padding, dilation, transposed, output_padding, groups, [False, True, False]))
    #                     wgrad_node.meta["tensor_meta"] = wgrad_output.meta["tensor_meta"]._replace()
    #                     wgrad_output.replace_all_uses_with(wgrad_node)
    #                 else:
    #                     # transfer the function to nhwc
    #                     def nhwc_wgrad_only(grad_output, input, weight, bias_sizes_opt, stride, padding, dilation, transposed, output_padding, groups, output_mask):
    #                         k, c, s, r = weight.size()
    #                         weight_permuted = weight.view(k, r, s, c).permute(0, 3, 1, 2)
    #                         weight_grad = torch.ops.aten.convolution_backward(grad_output, input, weight_permuted, bias_sizes_opt, stride, padding, dilation, transposed, output_padding, groups, output_mask)[1]
    #                         weight_grad = weight_grad.permute(0, 2, 3, 1).view(k, c, r, s)
    #                         return [None, weight_grad, None]
                        
    #                     node.target = nhwc_wgrad_only
    # graph.eliminate_dead_code()
    legalize_graph(module)
    # graph.lint()
   