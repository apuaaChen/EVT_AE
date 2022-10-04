import torch
from passes import *
from nodes import *
from functorch._src.aot_autograd import _is_primal, _is_tangent, _extract_graph_with_inputs_outputs


def pre_partition_optimization(joint_module):
    # get graph
    graph = joint_module.graph

    # pass: eliminate expand
    pass_eliminate_transparent_node(
        joint_module, graph, 
        [torch.ops.aten.expand, torch.ops.aten.detach]
    )

    # pass: loss elimination
    pass_loss_elimination(joint_module, graph)

    # pass: graph substitution
    # pass_graph_substitution(joint_module, graph)
    # pass_shape_prop(joint_module, graph)

    # pass: composed op breakdown
    pass_composed_op_breakdown(joint_module, graph)

    # pass: remove duplicated nodes
    pass_remove_duplicated_node(joint_module, graph)

    # pass: merge common factor of add / sub operators
    pass_merge_common_factor(joint_module, graph)

    # pass: update attributes
    pass_update_attributes(joint_module, graph)

    # pass: constant reduction
    pass_constant_folding(joint_module, graph)

    # pass: update attributes
    pass_update_attributes(joint_module, graph)

    # pass: strength reduction
    pass_stength_reduction(joint_module, graph)

    # pass: gemm fusion
    pass_gemm_fusion(joint_module, graph)

    # recompile graph
    joint_module.recompile()

    # visualize graph
    pass_print_graph(joint_module, "./joint_graph.svg")

