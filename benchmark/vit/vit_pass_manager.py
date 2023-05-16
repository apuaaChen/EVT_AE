import torch
from gtl.compiler.passes import *
from gtl.compiler.nodes import *
import logging


def pre_partition_optimization(joint_module, enabled_passes=["fusion", "uturn", "stream"]):
    frontend = GTLFrontend()
    joint_module, modified = frontend(joint_module)

    graph = joint_module.graph

    # joint_module.recompile()
    # print(joint_module.code)
    # exit()

    # pass: eliminate expand
    # pass_eliminate_transparent_node(
    #     joint_module, graph, 
    #     [torch.ops.aten.detach, torch.ops.aten.expand]
    # )

    # pass: loss elimination
    pass_loss_elimination(joint_module, graph)

    if "uturn" in enabled_passes:
        disabled_list = []
    else:
        disabled_list = [
            torch.ops.aten._log_softmax,
            torch.ops.aten.nll_loss_backward,
            torch.ops.aten._log_softmax_backward_data
        ]
    # pass: composed op breakdown
    pass_composed_op_breakdown(joint_module, graph, disabled_list)
 
    # pass: remove duplicated nodes
    pass_remove_duplicated_node(joint_module, graph)

    # pass: merge common factor of add / sub operators
    pass_merge_common_factor(joint_module, graph)

    # pass: update attributes
    pass_update_attributes(joint_module, graph)

    # pass: constant reduction
    pass_constant_folding(joint_module, graph)

    # pass: replace tranpose 3D with permute
    pass_trans_2_permute(joint_module, graph)


    # pass: update attributes
    pass_update_attributes(joint_module, graph)

    # pass: strength reduction
    pass_stength_reduction(joint_module, graph)

    # pass: mark permutation in epilogue
    pass_mark_epilogue_permutations(joint_module, graph)

    # pass: preprocess layernorm
    pass_layernorm_preprocessing(joint_module, graph)

    # pass: gemm fusion
    pass_gemm_fusion(joint_module, graph)

    pass_permute_view_fix(joint_module, graph)

    if "stream" in enabled_passes:
        logging.info("[PASS] Multi-Stream Pass")
        # pass: assign stream
        pass_assign_stream(joint_module, graph)

    # recompile graph
    joint_module.recompile()

    # visualize graph
    # pass_print_graph(joint_module, "./vit.svg")