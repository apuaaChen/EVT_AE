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
from typing import Optional
from torch.fx.graph_module import GraphModule
from gtl.compiler.nodes import *
from torch.fx.passes.tools_common import legalize_graph
from torch.fx.passes.infra.pass_base import PassBase, PassResult

class LocalCSE(PassBase):
    """
    GTL pass that eliminates the common subexpressions in local basic block
    The current implementation exploits the local value numbering by assigning
    unique hash value to different tensors and callees. The complexity is O(n)
    """
    def __init__(self) -> None:
        super().__init__()
        self.modified = False
        # to dictionaries matching nodes with hash value
        self.hash2node = {}
        self.node2hash = {}
    
    def is_commutative(self, node):
        assert node.op == "call_function"

        if len(node.args) <= 1: return False

        # register commutative operations here
        if node.target in [
            torch.ops.aten.add, torch.ops.aten.mul,
            torch.ops.aten.ne
        ]:
            return True
        
        return False
    
    def hash(self, node: torch.fx.Node):
        if node.op in ["placeholder", "get_attr"]:
            return hash(str(node.target))
        elif node.op == "call_function":
            # callee
            target_hash = [hash(str(node.target)),]
            # arg list
            args_hash = []
            for arg in node.args:
                if isinstance(arg, torch.fx.Node):
                    args_hash.append(self.node2hash[arg])
                else:
                    args_hash.append(hash(str(arg)))
            if self.is_commutative(node):
                args_hash.sort()
            # kwargs list
            kwargs_hash = []
            for key in node.kwargs.keys():
                kwarg = node.kwargs[key]
                if isinstance(kwarg, torch.fx.Node):
                    kwargs_hash.append(hash((kwarg, self.node2hash[arg])))
                else:
                    kwargs_hash.append(hash((kwarg, str(arg))))
            if self.is_commutative(node):
                kwargs_hash.sort()  
            return hash(tuple(target_hash + args_hash + kwargs_hash))
        elif node.op == "output":
            return hash("output")
        else:
            raise NotImplementedError(node.op)
    
    def call(self, graph_module: GraphModule) -> PassResult:
        graph = graph_module.graph
        self.modified = False

        for node in graph.nodes:
            hash_val = self.hash(node)
            self.node2hash[node] = hash_val
            if hash_val in self.hash2node.keys():
                node.replace_all_uses_with(self.hash2node[hash_val])
                self.modified = True
            else:
                self.hash2node[hash_val] = node
        
        return PassResult(graph_module, self.modified)
    
    def requires(self, graph_module: GraphModule) -> None:
        """
        This function will be called before the pass is run and will check that
        the given graph module contains the preconditions needed to run the
        pass. 

        In the CSE pass, it ensures the graph is topologically sorted

        Args:
            graph_module: The graph module we will run checks on
        """
        graph = graph_module.graph

        # the graph must be topologically sorted
        graph.lint()
    
    def ensures(self, graph_module: GraphModule) -> None:
        """
        This function will be called after the pass is run and will check that
        the given graph module contains the postconditions needed to run the
        pass. It is not required to implement this function.

        In the CSE pass, it performs the dead code elimination

        Args:
            graph_module: The graph module we will run checks on
        """
        graph_module.graph.eliminate_dead_code()

################################################################################
# Graph-level pass to merge duplicated nodes
################################################################################

def pass_remove_duplicated_node(module, graph):
    cse = LocalCSE()

    module, modified = cse(module)
