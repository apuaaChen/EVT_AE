import torch
import torch.fx as fx
from functorch._src.partitioners import _is_primal, _extract_fwd_bwd_outputs, \
    _extract_graph_with_inputs_outputs, _extract_graph_with_inputs_outputs, \
    _extract_fwd_bwd_modules
from passes import *
from vit_pass_manager import *
from functorch._src.compilers import ts_compile
import logging


def partition_func(joint_module: fx.GraphModule, _joint_inputs, enabled_passes=["fusion", "uturn", "stream"]):
    pre_partition_optimization(joint_module, enabled_passes)

    primal_inputs = list(filter(_is_primal, joint_module.graph.nodes))
    fwd_outputs, bwd_outputs = _extract_fwd_bwd_outputs(joint_module)

    forward_only_graph = _extract_graph_with_inputs_outputs(joint_module.graph, primal_inputs, fwd_outputs)
    forward_node_names = set([node.name for node in forward_only_graph.nodes if node.op != 'output'])

    def node_saved(node):
        return node.name in forward_node_names and 'tensor_meta' in node.meta
    saved_values = [node for node in joint_module.graph.nodes if node_saved(node)]
    return _extract_fwd_bwd_modules(joint_module, saved_values)

def compiler_fn(fx_module: torch.fx.GraphModule, _):
    logging.debug("============Optimized Source Code============")
    logging.debug(fx_module.code)
    return fx_module

def compiler_fn_nvfuser(fx_module: torch.fx.GraphModule, _):
    graph = fx_module.graph
    # fix the reshape bug in nvfuser
    for node in graph.nodes:
        if node.target == torch.ops.aten.view.default:
            node.target = torch.ops.aten.reshape
    
    fx_module.recompile()
    # print(fx_module.code)

    return ts_compile(fx_module, _)
