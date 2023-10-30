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
# Clean up pass: handle all the corner issues of the model
from typing import Optional
from torch.fx.graph_module import GraphModule
from torch.fx import Graph
from torch.fx.passes.infra.pass_base import PassBase, PassResult
import torch


class CleanUp(PassBase):
    def __init__(self) -> None:
        super().__init__()
    
    def call(self, graph_module: GraphModule) -> PassResult | None:
        graph: Graph = graph_module.graph
        for node in graph.nodes:
            # Case 1: one_hot node. the one-hot node by default produce tensor
            # with data type torch.int64_t. However, the torch.float16 is usually
            # expected.
            if node.target == torch.ops.aten.one_hot:
                users = list(node.users)
                if len(users) == 1 and users[0].target == torch.ops.aten.to:
                    continue
                # Insert the to node
                graph.inserting_after(node)
                to_node = graph.call_function(torch.ops.aten.to, args=(node, node.meta["tensor_meta"].dtype))
                for user in users:
                    user.replace_input_with(node, to_node)

def pass_clean_up(module, graph):
    clean_up = CleanUp()
    clean_up(module)
