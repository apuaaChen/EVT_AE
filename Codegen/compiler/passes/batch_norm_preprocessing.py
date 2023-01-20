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
# Graph-level pass to preprocess layer norm for kernel fusion
################################################################################
def pass_batchnorm_preprocessing(module, graph):
    batch_norm_idx = 0
    for node in graph.nodes:
        if node.op == "call_function":
            if node.target == torch.ops.aten.native_batch_norm:
                if batch_norm_idx >=1: continue

                ################################################################
                # inject the epilogue nodes for reduction fusion
                input = node.args[0]
                N, K, P, Q = list(input.meta["tensor_meta"].shape)
                reduction_length = N * P * Q

                # div_node = inject_mul(input, graph, input, 1./reduction_length)
                div_node = inject_div(input, graph, input, reduction_length)
                mean_x_node = inject_sum(div_node, graph, div_node, [0, 2, 3], dtype=torch.float32)
                mul_node = inject_mul(div_node, graph, div_node, input)
                mean_x2_node = inject_sum(mul_node, graph, mul_node, [0, 2, 3], dtype=torch.float32)

                graph.inserting_after(node)
                bn_splitted = graph.call_function(node.target, args=(*node.args, mean_x_node, mean_x2_node))

                # bn_splitted.meta["tensor_meta"] = node.meta["tensor_meta"]._replace()
                node.replace_all_uses_with(bn_splitted)
                graph.erase_node(node)
                
                batch_norm_idx += 1