import torch
import sys
sys.path.append("/workspace/sparseTraining/Codegen/compiler")
from passes import *
from nodes import *

def pre_partition_optimization(joint_module):
    # get graph
    graph = joint_module.graph

    pass_suffix_elimination(joint_module, graph)

    # pass: eliminate expand
    pass_eliminate_transparent_node(
        joint_module, graph, 
        [torch.ops.aten.detach,]
    )

    # pass: loss elimination
    pass_loss_elimination(joint_module, graph)

    # # pass: graph substitution
    # # pass_graph_substitution(joint_module, graph)
    # # pass_shape_prop(joint_module, graph)

    # # pass: composed op breakdown
    # pass_composed_op_breakdown(joint_module, graph)
 
    # pass: remove duplicated nodes
    pass_remove_duplicated_node(joint_module, graph)

    # pass: replace tranpose 3D with permute
    pass_trans_2_permute(joint_module, graph)

    # # # pass: merge common factor of add / sub operators
    # pass_merge_common_factor(joint_module, graph)

    # # pass: update attributes
    # pass_update_attributes(joint_module, graph)

    # # pass: constant reduction
    # pass_constant_folding(joint_module, graph)

    # # pass: update attributes
    # pass_update_attributes(joint_module, graph)

    # # pass: strength reduction
    # pass_stength_reduction(joint_module, graph)

    # pass: mark permutation in epilogue
    pass_mark_epilogue_permutations(joint_module, graph)

    # pass: gemm fusion
    pass_gemm_fusion(joint_module, graph)

    # pass: softmax fusion
    pass_softmax_fusion(joint_module, graph)

    # pass: assign stream
    # pass_assign_stream(joint_module, graph)

    # recompile graph
    joint_module.recompile()

    # visualize graph
    pass_print_graph(joint_module, "./joint_graph.svg")