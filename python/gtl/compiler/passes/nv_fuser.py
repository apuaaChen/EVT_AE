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
from torch.fx.graph_module import GraphModule
from torch.fx.passes.tools_common import legalize_graph
from torch.fx import Graph, Node
import torch
from torch._functorch.partitioners import _extract_graph_with_inputs_outputs
import operator
from gtl.compiler.passes.pass_clean_up import CleanUp


class FusedInductorOp:
    """
    Implementation of fused inductor op
    """
    __name__ = "inductor_fused_kernel"
    def __init__(self, gm: GraphModule) -> None:
        self.inputs = [node for node in gm.graph.nodes if node.op == "placeholder"]
        self.input_names = [node.name for node in self.inputs]
        output_node = [node for node in gm.graph.nodes if node.op == "output"][0]
        self.outputs = list(output_node.args[0])
        self.output_names = [node.name for node in self.outputs]
        CleanUp()(gm)
        gm.recompile()
        self.inductor_module = torch.compile(gm)
    
    def __call__(self, *args):
        inputs = {}
        for name, tensor in zip(self.input_names, args):
            inputs[name] = tensor

        outputs = self.inductor_module(**inputs)
        if not isinstance(outputs, list):
            outputs = [outputs,]
        
        return outputs


class NVFuser:
    """
    The entry point of the default fuser from torch inductor
    """
    def __init__(self) -> None:
        pass
    
    def get_node_with_name(name, graph_module):
        for node in graph_module.graph.nodes:
            if node.name == name:
                return node
        raise ValueError(f"{name} is not in the graph.")
    
    def trace(self, graph_module: GraphModule, partition_: 'list[Node]'):
        self.partition_names = partition_
        try:
            graph_module.graph.lint()
        except:
            legalize_graph(graph_module)
        partition = [NVFuser.get_node_with_name(n, graph_module) for n in partition_]

        # Step 2: extract the subgraph
        inputs = set()
        outputs = set()
        for node in partition:
            for input in node.all_input_nodes:
                if input not in partition:
                    inputs.add(input)
            has_external_user = False
            for user in node.users:    
                if user not in partition:
                    has_external_user = True
                    break
            if has_external_user:
                outputs.add(node)
        
        subgraph: Graph = _extract_graph_with_inputs_outputs(graph_module.graph,
                                                            inputs, outputs)
        
        sub_gm = GraphModule(graph_module, subgraph)

        fused_op = FusedInductorOp(sub_gm)
        input_nodes = [NVFuser.get_node_with_name(n, graph_module) for n in fused_op.input_names]
        output_nodes = [NVFuser.get_node_with_name(n, graph_module) for n in fused_op.output_names]

        fused_node = graph_module.graph.call_function(fused_op, args=tuple(input_nodes))
        fused_node.meta = {}
        graph_module.graph.inserting_after(fused_node)

        for idx, output in enumerate(output_nodes):
            get_item_node = graph_module.graph.call_function(operator.getitem, args=(fused_node, idx))
            get_item_node.meta = {}
            get_item_node.meta["tensor_meta"] = output.meta["tensor_meta"]._replace()
            output.replace_all_uses_with(get_item_node)
            graph_module.graph.inserting_after(get_item_node)
