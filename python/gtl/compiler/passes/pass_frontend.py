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
from torch.utils._python_dispatch import _pop_mode_temporarily

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
        # List of nodes that can be safely removed without consequences
        self.transparent_nodes = [
            torch.ops.aten.detach,
            torch.ops.aten.expand,  # usually handled by broadcast directly
        ]
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

        graph = graph_module.graph
        pass_suffix_elimination_(graph_module, graph)
        graph.eliminate_dead_code()
        graph.lint()
        graph_module.recompile()

        return PassResult(graph_module, self.modified)
    
    def visit(self, node: torch.fx.Node):
        if node.op == "call_function" and node.target in self.transparent_nodes:
            node.replace_all_uses_with(node.args[0])
            return

        # step 1: eliminate surffix
        target_name = str(node.target).split(sep='.')
        # this pass is designed specifically for aten operations
        if target_name[0] != "aten": return
        suffix = target_name[-1]
        if suffix in ["default", "Scalar", "Tensor", "dim_IntList", "int", "dim"]:
            op_name = target_name[-2]
            try:
                target_wo_suffix = getattr(torch.ops.aten, op_name)
                assert callable(target_wo_suffix), \
                    f"{target_wo_suffix} is not callable"
                node.target = target_wo_suffix
                self.modified = True
            except AttributeError:
                logging.debug(
                    f"torch.ops has no operator {op_name}")
        # now it is assumed that no operation bear a suffix

        # step 2: eliminate the inplace suffix "_" in operation name 
        target_name = str(node.target).split(sep=".")
        op_name = target_name[1]

        if op_name[-1] == "_":
            target_name_wo_op_suffix = target_name[:-2]
            try:
                target_wo_op_suffix = getattr(
                    torch.ops.aten, target_name_wo_op_suffix)
                assert callable(target_wo_suffix), \
                    f"{target_wo_op_suffix} is not callable"
                node.target = target_name_wo_op_suffix
                self.modified = True
            except AttributeError:
                logging.debug(
                    f"torch.ops has no operator {target_wo_op_suffix}")
    
    def requires(self, graph_module: GraphModule) -> None:
        return super().requires(graph_module)
    
    def ensures(self, graph_module: GraphModule) -> None:
        return super().ensures(graph_module)


################################################################################
# Unoptimized node
################################################################################
from gtl.compiler.nodes import *

def pass_suffix_elimination_(module, graph):
    for node in graph.nodes:
        if node.op == "call_function":
            if node.target == torch.ops.aten.native_dropout:
                node.meta["tensor_meta"] = node.args[0].meta["tensor_meta"]._replace()
    
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
                with _pop_mode_temporarily():
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
        elif node.target in [torch.ops.aten.sub, torch.ops.aten.div, torch.ops.aten.rsub]:
            if len(node.all_input_nodes) == 1:
                if node.args[0] in node.all_input_nodes:
                    with _pop_mode_temporarily():
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
                    with _pop_mode_temporarily():
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

    graph.eliminate_dead_code()
    graph.lint()