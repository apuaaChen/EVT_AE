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

################################################################################
# Graph-level pass to break down log softmax, nll_loss_forward, nll_loss_backward, 
# _log_softmax_backward_data
################################################################################
# TODO: the general substitution doesn't work for these cases

def pass_composed_op_breakdown(module, graph):
    softmax_node = None
    for node in graph.nodes:
        if node.op == "call_function":
            if node.target == torch.ops.aten._log_softmax:
                # break it down into softmax and log
                softmax_node = inject_softmax(
                    node, graph, node.args[0], node.args[1], node.args[2])
                log_node = inject_log(softmax_node, graph, softmax_node)
                node.replace_all_uses_with(log_node)
            elif node.target == torch.ops.aten.nll_loss_backward:
                if node.args[4] == 1:
                    reduction = "mean"
                elif node.args[4] == 2:
                    reduction = "sum"
                label_node = node.args[2]
                one_hot_node = inject_onehot(node, graph, node.meta["tensor_meta"].shape[1], label_node)
                neg_node = inject_neg(one_hot_node, graph, one_hot_node)
                mul_node = inject_mul(neg_node, graph, neg_node, node.args[0])
                if node.args[5] >= 0:
                    ne_node = inject_ne(mul_node, graph, label_node, node.args[5])
                    unsqueeze_node = inject_unsqueeze(ne_node, graph, ne_node, 1)
                    mul_mask_node = inject_mul(unsqueeze_node, graph, mul_node, unsqueeze_node)
                    if reduction == "mean":
                        div_node = inject_div(mul_mask_node, graph, mul_node, node.meta["tensor_meta"].shape[0])
                        node.replace_all_uses_with(div_node)
                    else:
                        node.replace_all_uses_with(mul_mask_node)
                        node.replace_all_uses_with(mul_mask_node)
                else:
                    if reduction == "mean":
                        div_node = inject_div(mul_node, graph, mul_node, node.meta["tensor_meta"].shape[0])
                        node.replace_all_uses_with(div_node)
                    else:
                        node.replace_all_uses_with(mul_node)
            elif node.target == torch.ops.aten._log_softmax_backward_data:
                sum_node = inject_sum(node, graph, node.args[0], 1)
                unsqueeze_node = inject_unsqueeze(sum_node, graph, sum_node, 1)
                mul_node = inject_mul(unsqueeze_node, graph, unsqueeze_node, softmax_node)
                sub_node = inject_sub(mul_node, graph, node.args[0], mul_node)
                node.replace_all_uses_with(sub_node)
            
    graph.eliminate_dead_code()
    graph.lint()