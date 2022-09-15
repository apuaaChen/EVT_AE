from cProfile import label
import torch
from torch.fx.passes import graph_drawer
import torch.nn.functional as F


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
    for node in graph.nodes:
        if node.op == "output":
            # loss is at input_node[0]
            loss_node = node.all_input_nodes[0]
            # locate the insertion place
            graph.inserting_after(loss_node)
            # add fake loss as register_buffer
            module.register_buffer(
                "_fake_loss_0", 
                torch.Tensor([1.0,]).to("cuda").requires_grad_().to(torch.float16))
            # create node
            fake_loss_node = graph.get_attr("_fake_loss_0")
            # insert node
            loss_node.replace_all_uses_with(fake_loss_node)
    graph.eliminate_dead_code()
    graph.lint()

################################################################################
# TODO: pass, break down log softmax, nll_loss_forward, nll_loss_backward, 
# _log_softmax_backward_data
################################################################################
def pass_composed_op_breakdown(module, graph):
    label_node = None
    softmax_node = None
    for node in graph.nodes:
        if node.op == "call_function":
            if node.target == torch.ops.aten._log_softmax:
                # break it down into softmax and log
                graph.inserting_after(node)
                softmax_node = graph.call_function(torch.ops.aten._softmax, args=node.args)
                softmax_node.meta = node.meta
                graph.inserting_after(softmax_node)
                log_node = graph.call_function(torch.ops.aten.log, args=(softmax_node,))
                log_node.meta = node.meta
                node.replace_all_uses_with(log_node)
            elif node.target == torch.ops.aten.nll_loss_forward:
                label_node = node.args[1]
            elif node.target == torch.ops.aten.nll_loss_backward:
                graph.inserting_after(node)
                one_hot_node = graph.call_function(torch.ops.aten.one_hot, args=(label_node,), kwargs={"num_classes": node.meta["tensor_meta"].shape[1]})
                one_hot_node.meta = node.meta
                graph.inserting_after(one_hot_node)
                neg_node = graph.call_function(torch.ops.aten.neg, args=(one_hot_node,))
                neg_node.meta = neg_node.meta
                graph.inserting_after(neg_node)
                mul_node = graph.call_function(torch.ops.aten.mul, args=(neg_node, node.args[0]))
                mul_node.meta = neg_node.meta

                # get padding mask
                graph.inserting_after(mul_node)
                ne_node = graph.call_function(torch.ops.aten.ne, args=(label_node, node.args[5]))
                ne_node.meta = label_node.meta

                graph.inserting_after(ne_node)
                unsqueeze_node = graph.call_function(torch.ops.aten.unsqueeze, args=(ne_node, 1))
                graph.inserting_after(unsqueeze_node)
                mul_mask_node = graph.call_function(torch.ops.aten.mul, args=(mul_node, unsqueeze_node))
                
                node.replace_all_uses_with(mul_mask_node)
            elif node.target == torch.ops.aten._log_softmax_backward_data:
                # TODO: Expand this operator
                graph.inserting_after(node)
                sum_node = graph.call_function(torch.ops.aten.sum, args=(node.args[0], 1))
                graph.inserting_after(sum_node)
                unsqueeze_node = graph.call_function(torch.ops.aten.unsqueeze, args=(sum_node, 1))
                graph.inserting_after(unsqueeze_node)
                mul_node = graph.call_function(torch.ops.aten.mul, args=(unsqueeze_node, softmax_node))
                graph.inserting_after(mul_node)
                sub_node = graph.call_function(torch.ops.aten.sub, args=(node.args[0], mul_node))
                node.replace_all_uses_with(sub_node)
                # pass
            
    graph.eliminate_dead_code()
    graph.lint()

def pre_partition_optimization(joint_module):
    # get graph
    graph = joint_module.graph
    # pass: loss elimination
    pass_loss_elimination(joint_module, graph)

    # pass: composed op breakdown
    pass_composed_op_breakdown(joint_module, graph)
    
    # recompile graph
    joint_module.recompile()

    # visualize graph
    pass_print_graph(joint_module)


