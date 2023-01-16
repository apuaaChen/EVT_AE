import torch
import sys
sys.path.append("/workspace/sparseTraining/Codegen/compiler")
from passes import *
from nodes import *

def pre_partition_optimization(joint_module):
    # get graph
    graph = joint_module.graph

    pass_suffix_elimination(joint_module, graph)

    # pass: loss elimination
    pass_loss_elimination(joint_module, graph)

    # pass: composed op breakdown
    pass_composed_op_breakdown(joint_module, graph)

    # pass: eliminate expand
    pass_eliminate_transparent_node(
        joint_module, graph, 
        [torch.ops.aten.detach,]
    )

    # pass: constant reduction
    pass_constant_folding(joint_module, graph)

    # pass: strength reduction
    pass_stength_reduction(joint_module, graph)
    

    # pass gemm fusion
    pass_gemm_fusion(joint_module, graph)

    # pass: assign stream
    pass_assign_stream(joint_module, graph, 1)

    # recompile graph
    joint_module.recompile()

    # visualize graph
    pass_print_graph(joint_module, "./joint_graph.svg")
    
