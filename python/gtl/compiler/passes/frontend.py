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
from torch.fx.passes.infra.pass_base import PassBase, PassResult
import operator
import torch
from torch import ops
import logging

################################################################################
# Compiler Frontend that formalize the compute graph representation & reduce
# operation space
################################################################################

class GTLFrontend(PassBase):
    """
    Compiler Frontend that formalize the compute graph representation 
    & reduce operation space
    """
    def __init__(self) -> None:
        # flag tracking if the module is updated
        self.modified = False
        super().__init__()
    
    def call(self, graph_module: GraphModule) -> Optional[PassResult]:
        self.modified = False
        """
        Optimization 1: eliminate the surffix of target in torch
        example:
            torch.op.aten.t.default -> aten.t
        """
        graph = graph_module.graph
        for node in graph.nodes:
            self.visit(node)
        graph_module.recompile()

        return PassResult(graph_module, self.modified)
    def visit(self, node: torch.fx.Node):
        # step 1: eliminate surffix
        target_name = str(node.target)
        suffix = target_name.split(sep='.')[-1]
        if suffix in ["default", "Scalar", "Tensor", "dim_IntList", "int", "dim"]:
            target_name_wo_suffix = target_name[:-len(suffix) - 1]
            target_wo_suffix = getattr(ops, target_name_wo_suffix)
            node.target = target_wo_suffix
            self.modified = True

    
    def requires(self, graph_module: GraphModule) -> None:
        return super().requires(graph_module)
    
    def ensures(self, graph_module: GraphModule) -> None:
        return super().ensures(graph_module)

################################################################################
# Registered mapping between targets and gtl-registered targets
################################################################################

# suffix_dict = {
#     aten.div_.Scalar: aten.div_,
#     aten.rsub.Scalar: aten.sub,
#     aten.relu_.default: aten.relu,
#     aten.add_.Tensor: aten.add,
# }