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
from torch.fx.passes.infra.pass_base import PassBase, PassResult
from torch.fx.graph_module import GraphModule
from torch.fx.node import Node
import numpy as np
from torch.fx.passes.tools_common import legalize_graph
from gtl.compiler.passes.pass_fake_shape_infer import pass_fake_shape_infer


class PermuteProp(PassBase):
    """
    This pass tries to eliminate the permute nodes by propagating them along the
    edges. It is particularly helpful for convnets
    """
    COMMUTATIVE_TGTS = [
        torch.ops.aten.add, 
        torch.ops.aten.sub,
        torch.ops.aten.mul,
        torch.ops.aten.div,
        torch.ops.aten.ne,
        torch.ops.aten.native_dropout,
        torch.ops.aten.add,
        torch.ops.aten.rsub,
        torch.ops.aten.gelu,
        torch.ops.aten.tanh,
        torch.ops.aten.tanh_backward,
        torch.rand_like,
        torch.ops.aten.ge,
        torch.ops.aten.relu,
        torch.ops.aten.sigmoid,
        torch.ops.aten.to,
        torch.ops.aten.rsqrt
    ]
    def call(self, graph_module: GraphModule) -> PassResult | None:
        self.graph = graph_module.graph
        # Step 1: collect all the permutation nodes in the graph
        self.worklist = []
        for node in graph_module.graph.nodes:
            if node.target == torch.ops.aten.permute:
                self.worklist.append(node)
        
        # Step 2: process all the permutations in the worklist
        while len(self.worklist) > 0:
            permute_node: Node = self.worklist.pop()
            self.stack = []
            self.visited =  []
            users = list(permute_node.users)
            permute_node.replace_all_uses_with(permute_node.args[0])
            self.stack.append(permute_node.args[0])
            for user in users:
                self.propagate_permute_to_user(user, permute_node)
        
        # Step 3: post processing
        legalize_graph(graph_module)
        graph_module.graph.eliminate_dead_code()
        graph_module.recompile()
        pass_fake_shape_infer(graph_module, graph_module.graph)

    
    def propagate_permute_to_user(self, node: Node, permute_node: Node):
        if node in self.visited:
            return
        self.visited.append(node)

        if node.target in self.COMMUTATIVE_TGTS:
            if node.meta["tensor_meta"].shape != permute_node.meta["tensor_meta"].shape:
                breakpoint()
                raise NotImplementedError
            # Reset the metadata
            node.meta = {}
        elif node.target == torch.ops.aten.permute:
            # Meet another permute
            permute_indices = [permute_node.args[1][i] for i in node.args[1]]
            if permute_indices == list(range(len(permute_indices))):
                # delete the node
                node.replace_all_uses_with(node.args[0])
                if node in self.worklist:
                    self.worklist.remove(node)
                return
            else:
                node.args = (node.args[0], permute_indices)
                node.meta = {}
                return
        else:
            # Insert a copy of current node here
            self.graph.inserting_before(node)
            copied_permute_node = self.graph.call_function(
                torch.ops.aten.permute, 
                args=(self.stack[-1], permute_node.args[1]))
            node.replace_input_with(self.stack[-1], copied_permute_node)
            return

        # Propagate to inputs
        inv_permute_indices = list(np.argsort(permute_node.args[1]))
        inputs = list(node.all_input_nodes)
        for input in inputs:
            if input != self.stack[-1]:
                if len(input.meta["tensor_meta"].shape) in [0, 1]:
                    continue
                elif len(input.meta["tensor_meta"].shape) != len(permute_node.meta["tensor_meta"].shape):
                    breakpoint()
                    raise NotImplementedError()
                # Insert a inversed copy here
                self.graph.inserting_after(input)
                copied_permute_node = self.graph.call_function(
                    torch.ops.aten.permute, 
                    args=(input, inv_permute_indices))
                node.replace_input_with(input, copied_permute_node)
            
        # Propagate to users
        self.stack.append(node)
        users = list(node.users)
        for user in users:
            self.propagate_permute_to_user(user, permute_node)
        self.stack.pop()


def pass_permute_propagation(module, graph):
    PermuteProp()(module)

