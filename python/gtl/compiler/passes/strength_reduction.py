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

################################################################################
# Graph-level pass to perform stength reduction
################################################################################
def pass_stength_reduction(module, graph):
    for node in graph.nodes:
        if node.op == "call_function":
            if node.target == torch.ops.aten.add_:
                node.target = torch.ops.aten.add
    
    for node in graph.nodes:
        if node.op == "get_attr":
            shape = list(node.meta["tensor_meta"].shape)
            numel = 1
            for dim in shape:
                numel *= dim
            if numel == 1 and "fake_loss" not in node.target:
                value = getattr(module, node.target).detach().cpu().item()
                node.replace_all_uses_with(value)
    
    # mul is much faster than div in epilogue
    for node in graph.nodes:
        if node.op == "call_function":
            if node.target == torch.ops.aten.div:
                if len(node.all_input_nodes) == 1:
                    if node.args[0] == node.all_input_nodes[0]:
                        mul_node = inject_mul(node, graph, node.args[0], 1./node.args[1])
                        node.replace_all_uses_with(mul_node)
                        graph.erase_node(node)