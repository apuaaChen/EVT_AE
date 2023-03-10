import torch
import torch.fx as fx
from functorch._src.partitioners import _is_primal, _extract_fwd_bwd_outputs, \
    _extract_graph_with_inputs_outputs, _extract_graph_with_inputs_outputs, \
    _extract_fwd_bwd_modules
from gat_pass_manager import *
from passes import *



def partition_func(joint_module: fx.GraphModule, _joint_inputs):
    pre_partition_optimization(joint_module)

    primal_inputs = list(filter(_is_primal, joint_module.graph.nodes))
    fwd_outputs, bwd_outputs = _extract_fwd_bwd_outputs(joint_module)

    forward_only_graph = _extract_graph_with_inputs_outputs(joint_module.graph, primal_inputs, fwd_outputs)
    forward_node_names = set([node.name for node in forward_only_graph.nodes if node.op != 'output'])

    def node_saved(node):
        return node.name in forward_node_names and 'tensor_meta' in node.meta
    saved_values = [node for node in joint_module.graph.nodes if node_saved(node)]
    return _extract_fwd_bwd_modules(joint_module, saved_values)

def compiler_fn(fx_module: torch.fx.GraphModule, _):
    print(fx_module.code)
    return fx_module