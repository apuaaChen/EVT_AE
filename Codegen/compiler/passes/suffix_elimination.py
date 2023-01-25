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
from nodes import *
import operator

################################################################################
# Graph-level pass to eliminate suffix
################################################################################
suffix_dict = {
    torch.ops.aten.t.default: torch.ops.aten.t,
    torch.ops.aten.view.default: torch.ops.aten.view,
    torch.ops.aten._log_softmax.default: torch.ops.aten._log_softmax,
    torch.ops.aten.ne.Scalar: torch.ops.aten.ne,
    torch.ops.aten.mul.Tensor: torch.ops.aten.mul,
    torch.ops.aten.sum.default: torch.ops.aten.sum,
    torch.ops.aten.add.Tensor: torch.ops.aten.add,
    torch.ops.aten.expand.default: torch.ops.aten.expand,
    torch.ops.aten.neg.default: torch.ops.aten.neg,
    torch.ops.aten.div.Scalar: torch.ops.aten.div,
    torch.ops.aten.div_.Scalar: torch.ops.aten.div_,
    torch.ops.aten.mm.default: torch.ops.aten.mm,
    torch.ops.aten.sum.dim_IntList: torch.ops.aten.sum,
    torch.ops.aten.detach.default: torch.ops.aten.detach,
    torch.ops.aten.unsqueeze.default: torch.ops.aten.unsqueeze,
    torch.ops.aten.nll_loss_backward.default: torch.ops.aten.nll_loss_backward,
    torch.ops.aten._log_softmax_backward_data.default: torch.ops.aten._log_softmax_backward_data,
    torch.ops.aten.nll_loss_forward.default: torch.ops.aten.nll_loss_forward,
    torch.ops.aten.addmm.default: torch.ops.aten.addmm,
    torch.ops.aten._to_copy.default: torch.ops.aten._to_copy,
    torch.ops.aten.rsub.Scalar: torch.ops.aten.sub,
    torch.ops.aten.sub.Tensor: torch.ops.aten.sub,
    torch.ops.aten.arange.default: torch.ops.aten.arange,
    torch.ops.aten.embedding.default:torch.ops.aten.embedding,
    torch.ops.aten.native_layer_norm.default: torch.ops.aten.native_layer_norm,
    torch.ops.aten.native_dropout.default: torch.ops.aten.native_dropout,
    torch.ops.aten.transpose.int: torch.ops.aten.transpose,
    torch.ops.aten.clone.default: torch.ops.aten.clone,
    torch.ops.aten._unsafe_view.default: torch.ops.aten._unsafe_view,
    operator.getitem: operator.getitem,
    torch.ops.aten.permute.default: torch.ops.aten.permute,
    torch.ops.aten.bmm.default: torch.ops.aten.bmm,
    torch.ops.aten.div.Tensor: torch.ops.aten.div,
    torch.ops.aten._softmax.default: torch.ops.aten._softmax,
    torch.ops.aten.gelu.default: torch.ops.aten.gelu,
    torch.ops.aten.slice.Tensor: torch.ops.aten.slice,
    torch.ops.aten.select.int: torch.ops.aten.select,
    torch.ops.aten.tanh.default: torch.ops.aten.tanh,
    torch.ops.aten.nonzero.default: torch.ops.aten.nonzero,
    torch.ops.aten.squeeze.default: torch.ops.aten.squeeze,
    torch.ops.aten.index_select.default: torch.ops.aten.index_select,
    torch.ops.aten.is_same_size.default: torch.ops.aten.is_same_size,
    torch.ops.aten.native_layer_norm_backward.default: torch.ops.aten.native_layer_norm_backward,
    torch.ops.aten.gelu_backward.default: torch.ops.aten.gelu_backward,
    torch.ops.aten.new_empty.default: torch.ops.aten.new_empty,
    torch.ops.aten.zero_.default: torch.ops.aten.zero_,
    torch.ops.aten.index_add.default: torch.ops.aten.index_add,
    torch.ops.aten.tanh_backward.default: torch.ops.aten.tanh_backward,
    torch.ops.aten.select_backward.default: torch.ops.aten.select_backward,
    torch.ops.aten.slice_backward.default: torch.ops.aten.slice_backward,
    torch.ops.aten.native_dropout_backward.default: torch.ops.aten.native_dropout_backward,
    torch.ops.aten._softmax_backward_data.default: torch.ops.aten._softmax_backward_data,
    torch.ops.aten.embedding_dense_backward.default: torch.ops.aten.embedding_dense_backward,
    torch.ops.aten.relu.default: torch.ops.aten.relu,
    torch.ops.aten.sigmoid.default: torch.ops.aten.sigmoid,
    torch.ops.aten.threshold_backward.default: torch.ops.aten.threshold_backward,
    torch.ops.aten.convolution.default: torch.ops.aten.convolution,
    torch.ops.aten.native_batch_norm_backward.default: torch.ops.aten.native_batch_norm_backward,
    torch.ops.aten.native_batch_norm.default: torch.ops.aten.native_batch_norm,
    torch.ops.aten.relu_.default: torch.ops.aten.relu,
    torch.ops.aten.add_.Tensor: torch.ops.aten.add,
    torch.ops.aten.mean.dim: torch.ops.aten.mean,
    torch.ops.aten.native_batch_norm_backward.default: torch.ops.aten.native_batch_norm_backward,
    torch.ops.aten.convolution_backward.default: torch.ops.aten.convolution_backward
}

def pass_suffix_elimination(module, graph):
    for node in graph.nodes:
        if node.op == "call_function":
            try:
                node.target = suffix_dict[node.target]
                if node.target == torch.ops.aten.native_dropout:
                    node.meta["tensor_meta"] = node.args[0].meta["tensor_meta"]._replace()
            except:
                print(node.target)
    
    # insert constant as attribute tensors
    name_idx = 0
    for node in graph.nodes:
        if node.target in [torch.ops.aten.mul, torch.ops.aten.add]:
            if len(node.all_input_nodes) == 1:
                input_node = node.all_input_nodes[0]
                # get the constant value
                constant_value = None
                constant_idx = None
                for idx, arg in enumerate(node.args):
                    if arg != input_node:
                        constant_value = arg
                        constant_idx = idx
                constant_node = inject_get_attr(
                    input_node, module, graph,
                    torch.Tensor([constant_value,]).to("cuda").to(torch.float16),
                    "const_scalar%d" % name_idx
                )
                name_idx += 1
                graph.inserting_after(constant_node)
                scalar_node = graph.call_function(node.target, args=(input_node, constant_node))
                scalar_node.meta = {}
                scalar_node.meta['tensor_meta'] = node.meta['tensor_meta']._replace()
                node.replace_all_uses_with(scalar_node)
        elif node.target in [torch.ops.aten.sub, torch.ops.aten.div]:
            if len(node.all_input_nodes) == 1:
                if node.args[0] in node.all_input_nodes:
                    constant_node = inject_get_attr(
                        node.args[0], module, graph,
                        torch.Tensor([node.args[1],]).to("cuda").to(torch.float16),
                        "const_scalar%d" % name_idx
                    )
                    name_idx += 1
                    graph.inserting_after(constant_node)
                    scalar_node = graph.call_function(node.target, args=(node.args[0], constant_node))
                    scalar_node.meta = {}
                    scalar_node.meta['tensor_meta'] = node.meta['tensor_meta']._replace()
                    node.replace_all_uses_with(scalar_node)
                elif node.args[1] in node.all_input_nodes:
                    constant_node = inject_get_attr(
                        node.args[1], module, graph,
                        torch.Tensor([node.args[0],]).to("cuda").to(torch.float16),
                        "const_scalar%d" % name_idx
                    )
                    name_idx += 1
                    graph.inserting_after(constant_node)
                    scalar_node = graph.call_function(node.target, args=(constant_node, node.args[1]))
                    scalar_node.meta = {}
                    scalar_node.meta['tensor_meta'] = node.meta['tensor_meta']._replace()
                    node.replace_all_uses_with(scalar_node)
        elif node.target == torch.ops.aten.addmm:
            bias, lhs, rhs = node.args
            mm_node = inject_mm(node, graph, lhs, rhs)
            add_node = inject_add(mm_node, graph, mm_node, bias)
            node.replace_all_uses_with(add_node)

    graph.eliminate_dead_code()
    graph.lint()


