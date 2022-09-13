import torch
from torch.fx.passes import graph_drawer


################################################################################
# Graph-level pass to print fx graph to disk as svg file
################################################################################
def pass_print_graph(module):
    g = graph_drawer.FxGraphDrawer(module, "dynamic_classifier")
    with open("./joint_graph.svg", "wb") as f:
        f.write(g.get_dot_graph().create_svg())

################################################################################
# Graph-level pass to replase loss node with a static Tensor 1.0, 
# Eliminates the original loss with dead code elimination
################################################################################
def pass_loss_elimination(module, graph):
    # eliminate loss
    for node in graph.nodes:
        if node.op == "output":
            # loss is at input_node[0]
            loss_node = node.all_input_nodes[0]
            # locate the insertion place
            graph.inserting_after(loss_node)
            # add fake loss as register_buffer
            module.register_buffer(
                "_fake_loss_0", 
                torch.Tensor([1.0,]).to("cuda").requires_grad_())
            # create node
            fake_loss_node = graph.get_attr("_fake_loss_0")
            # insert node
            loss_node.replace_all_uses_with(fake_loss_node)
    graph.eliminate_dead_code()
    graph.lint()


def pre_partition_optimization(joint_module):
    # get graph
    graph = joint_module.graph
    # pass: loss elimination
    pass_loss_elimination(joint_module, graph)
    
    # recompile graph
    joint_module.recompile()

    # visualize graph
    pass_print_graph(joint_module)


