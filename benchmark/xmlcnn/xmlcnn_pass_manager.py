import torch
import sys
sys.path.append("/workspace/gtl/sparseTraining/Codegen/compiler")
from passes import *
from nodes import *
import logging

def pre_partition_optimization(joint_module, enabled_passes=["fusion", "uturn", "stream"]):
    # get graph
    graph = joint_module.graph

    pass_suffix_elimination(joint_module, graph)

    # pass: loss elimination
    pass_loss_elimination(joint_module, graph)

    if "uturn" in enabled_passes:
        logging.info("[PASS] Uturn Pass")
        disabled_list = [torch.ops.aten.convolution_backward,]
    else:
        disabled_list = [
            torch.ops.aten.convolution_backward,
            torch.ops.aten._log_softmax,
            torch.ops.aten.nll_loss_backward,
            torch.ops.aten._log_softmax_backward_data
        ]
    # pass: composed op breakdown
    pass_composed_op_breakdown(joint_module, graph, disabled_list=disabled_list)

    # pass: eliminate expand
    pass_eliminate_transparent_node(
        joint_module, graph, 
        [torch.ops.aten.detach,]
    )

    # pass: constant reduction
    pass_constant_folding(joint_module, graph)

    # pass: strength reduction
    pass_stength_reduction(joint_module, graph)
    
    if "fusion" in enabled_passes:
        logging.info("[PASS] Fusion Pass")
        # pass gemm fusion
        pass_gemm_fusion(joint_module, graph)

    if "stream" in enabled_passes:
        logging.info("[PASS] Multi-Stream Pass")
        # pass: assign stream
        pass_assign_stream(joint_module, graph, 1)

    # recompile graph
    joint_module.recompile()

    # visualize graph
    # pass_print_graph(joint_module, "./joint_graph.svg")
    
