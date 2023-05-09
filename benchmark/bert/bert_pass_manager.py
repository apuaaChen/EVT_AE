import torch
from gtl.compiler.passes import *
from gtl.compiler.nodes import *
import logging

def pre_partition_optimization(joint_module, enabled_passes=["fusion", "uturn", "stream"]):
    # get graph
    graph = joint_module.graph

    # frontend = GTLFrontend()
    # frontend(joint_module)

    pass_suffix_elimination(joint_module, graph)

    # pass: eliminate expand
    pass_eliminate_transparent_node(
        joint_module, graph, 
        [torch.ops.aten.detach,]
    )

    logging.info("[PASS] Loss Elimination")
    # pass: loss elimination
    pass_loss_elimination(joint_module, graph)

    logging.info("[PASS] Composed Op Breaking down & Constant Folding")
    if "uturn" in enabled_passes:
        disabled_list = []
    else:
        disabled_list = [
            torch.ops.aten._log_softmax,
            torch.ops.aten.nll_loss_backward,
            torch.ops.aten._log_softmax_backward_data
        ]
    # # pass: composed op breakdown
    pass_composed_op_breakdown(joint_module, graph, disabled_list)
 
    # pass: remove duplicated nodes
    pass_remove_duplicated_node(joint_module, graph)

    # pass: merge common factor of add / sub operators
    pass_merge_common_factor(joint_module, graph)

    # pass: update attributes
    pass_update_attributes(joint_module, graph)

    # pass: constant reduction
    pass_constant_folding(joint_module, graph)

    logging.info("[PASS] Permute Preprocessing")
    # pass: replace tranpose 3D with permute
    pass_trans_2_permute(joint_module, graph)

    # pass: update attributes
    pass_update_attributes(joint_module, graph)

    # pass: strength reduction
    pass_stength_reduction(joint_module, graph)

    # pass: mark permutation in epilogue
    pass_mark_epilogue_permutations(joint_module, graph)

    logging.info("[PASS] Layernorm Preprocessing")
    # pass: preprocess layernorm
    pass_layernorm_preprocessing(joint_module, graph)

    logging.info("[PASS] Fusion")
    # # pass: gemm fusion
    pass_gemm_fusion(joint_module, graph)

    if "stream" in enabled_passes:
        logging.info("[PASS] Multi-Stream Pass")
        # pass: assign stream
        pass_assign_stream(joint_module, graph)

    # pass weight grad tuner
    # pass_weight_gradient_tuner(joint_module, graph)

    # recompile graph
    joint_module.recompile()

    # visualize graph
    # pass_print_graph(joint_module, "./joint_graph.svg")