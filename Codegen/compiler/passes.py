import torch
from torch.fx.passes import graph_drawer
from nodes import *


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

            fake_loss_node = inject_get_attr(
                loss_node, module, graph, 
                torch.Tensor([1.0,]).to("cuda").requires_grad_().to(torch.float16), 
                "_fake_loss_0")

            loss_node.replace_all_uses_with(fake_loss_node)
    graph.eliminate_dead_code()
    graph.lint()

################################################################################
# Graph-level pass to break down log softmax, nll_loss_forward, nll_loss_backward, 
# _log_softmax_backward_data
################################################################################
def pass_composed_op_breakdown(module, graph):
    label_node = None
    softmax_node = None
    for node in graph.nodes:
        if node.op == "call_function":
            if node.target == torch.ops.aten._log_softmax:
                # break it down into softmax and log
                softmax_node = inject_softmax(
                    node, graph, node.args[0], node.args[1], node.args[2])
                log_node = inject_log(softmax_node, graph, softmax_node)
                node.replace_all_uses_with(log_node)
            elif node.target == torch.ops.aten.nll_loss_forward:
                label_node = node.args[1]
            elif node.target == torch.ops.aten.nll_loss_backward:
                one_hot_node = inject_onehot(node, graph, node.meta["tensor_meta"].shape[1], label_node)
                neg_node = inject_neg(one_hot_node, graph, one_hot_node)
                mul_node = inject_mul(neg_node, graph, neg_node, node.args[0])
                ne_node = inject_ne(mul_node, graph, label_node, node.args[5])
                unsqueeze_node = inject_unsqueeze(ne_node, graph, ne_node, 1)
                mul_mask_node = inject_mul(unsqueeze_node, graph, mul_node, unsqueeze_node)
                node.replace_all_uses_with(mul_mask_node)
            elif node.target == torch.ops.aten._log_softmax_backward_data:
                sum_node = inject_sum(node, graph, node.args[0], 1)
                unsqueeze_node = inject_unsqueeze(sum_node, graph, sum_node, 1)
                mul_node = inject_mul(unsqueeze_node, graph, unsqueeze_node, softmax_node)
                sub_node = inject_sub(mul_node, graph, node.args[0], mul_node)
                node.replace_all_uses_with(sub_node)
            
    graph.eliminate_dead_code()
    graph.lint()

################################################################################
# Graph-level pass to merge duplicated nodes
################################################################################
def node_equal(node1, node2):
    if node1.op != node2.op:
        return False
    if node1.target != node2.target:
        return False
    if node1.args != node2.args:
        return False
    return True
    
def pass_remove_duplicated_node(module, graph):
    modified = True
    while(modified):
        modified = False
        for i, u in enumerate(graph.nodes):
            for j, v in enumerate(graph.nodes):
                if i == j: continue
                if node_equal(u, v):
                    v.replace_all_uses_with(u)
                    graph.eliminate_dead_code()
                    modified = True
                    break
    
    graph.lint()

################################################################################
# Graph-level pass to merge common factor of add / sub operators
################################################################################

def extract_factor(node):
    # return numerator and demoninator factors
    if isinstance(node, float) or isinstance(node, int):
        return [node,], []
    if isinstance(node, torch.fx.Node):
        if node.op == "call_function":
            # mul & div
            if node.target == torch.ops.aten.mul:
                lhs_factor_n, lhs_factor_d = extract_factor(node.args[0])
                rhs_factor_n, rhs_factor_d = extract_factor(node.args[1])
                return lhs_factor_n + rhs_factor_n, lhs_factor_d + rhs_factor_d
            elif node.target == torch.ops.aten.div:
                lhs_factor_n, lhs_factor_d = extract_factor(node.args[0])
                rhs_factor_n, rhs_factor_d = extract_factor(node.args[1])
                return lhs_factor_n + rhs_factor_d, lhs_factor_d + rhs_factor_n
            elif node.target == torch.ops.aten.neg:
                factor_n, factor_d = extract_factor(node.args[0])
                return [-1,] + factor_n, factor_d
            # transparent nodes
            elif node.target in [torch.ops.aten.view, torch.ops.aten.sum, torch.ops.aten.expand, torch.ops.aten.unsqueeze]:
                return extract_factor(node.args[0])
            else:
                return [node, ], []
        elif node.op in ["get_attr", "placeholder"]:
            return [node, ], []
    else:
        print(node)
        raise TypeError


def remove_factor_denominator(node, common_factor):
    if isinstance(node, torch.fx.Node):
        if node.op == "call_function":
            # mul & div
            if node.target == torch.ops.aten.mul:
                remove_factor_denominator(node.args[0], common_factor)
                remove_factor_denominator(node.args[1], common_factor)
            elif node.target == torch.ops.aten.div:
                if node.args[1] in common_factor:
                    node.replace_all_uses_with(node.arg[0])
                    remove_factor_denominator(node.args[0], common_factor)
                else:
                    remove_factor_numerator(node.args[1], common_factor)
                    remove_factor_denominator(node.args[0], common_factor)
            elif node.target == torch.ops.aten.neg:
                if -1 in common_factor:
                    node.replace_all_uses_with(node.args[0])
                    remove_factor_denominator(node.args[0])
            # transparent nodes
            if node.target in [torch.ops.aten.view, torch.ops.aten.sum, torch.ops.aten.expand, torch.ops.aten.unsqueeze]:
                return remove_factor_denominator(node.args[0], common_factor)

def remove_factor_numerator(node, common_factor):
    # This function takes a node and a list of common factors. 
    # If the node is common factor itself, return True
    if isinstance(node, torch.fx.Node):
        if node in common_factor:
            return True
        if node.op == "call_function":
            # mul & div
            if node.target == torch.ops.aten.mul:
                if (node.args[0] in common_factor 
                    and node.args[1] not in common_factor):
                    #
                    node.replace_all_uses_with(node.args[1])
                    return remove_factor_numerator(node.args[1], common_factor)
                elif (node.args[0] not in common_factor 
                    and node.args[1] in common_factor):
                    #
                    node.replace_all_uses_with(node.args[0])
                    return remove_factor_numerator(node.args[0], common_factor)
                elif (node.args[0] in common_factor 
                    and node.args[1] in common_factor):
                    # 
                    raise NotImplementedError
                else:
                    lhs_empty = remove_factor_numerator(node.args[0], common_factor)
                    rhs_empty = remove_factor_numerator(node.args[1], common_factor)
                    if lhs_empty and not rhs_empty:
                        node.replace_all_uses_with(node.args[1])
                    elif rhs_empty and not lhs_empty:
                        node.replace_all_uses_with(node.args[0])
                    else:
                        return (lhs_empty and rhs_empty)
            elif node.target == torch.ops.aten.div:
                if node.args[0] in common_factor:
                    node.replace_input_with(node.args[0], 1)
                    return remove_factor_denominator(node.args[1], common_factor)
                else:
                    lhs_empty = remove_factor_numerator(node.args[0], common_factor)
                    rhs_empty = remove_factor_denominator(node.args[1], common_factor)
                    return (lhs_empty and rhs_empty)
            elif node.target == torch.ops.aten.neg:
                if -1 in common_factor:
                    node.replace_all_uses_with(node.args[0])
                    return remove_factor_numerator(node.args[0], common_factor)
                else:
                    return False
            # transparent nodes
            elif node.target in [torch.ops.aten.view, torch.ops.aten.sum, torch.ops.aten.expand, torch.ops.aten.unsqueeze]:
                return remove_factor_numerator(node.args[0], common_factor)
            
def intersection(lst1, lst2):
    return [value for value in lst1 if value in lst2]

def pass_merge_common_factor(module, graph):
    tmp_node = None
    print(dir(graph.nodes))
    for idx, node in enumerate(graph.nodes):
        if idx == 0:
            tmp_node = node
        if node.op == "call_function":
            if node.target == torch.ops.aten.add:
                # extract factors of lhs child and rhs child
                lhs_factors_n, lhs_factors_d = extract_factor(node.args[0])
                rhs_factors_n, rhs_factors_d = extract_factor(node.args[1])
                common_numerator_n = intersection(lhs_factors_n, rhs_factors_n)
                remove_factor_numerator(node.args[0], [common_numerator_n[0], common_numerator_n[1], common_numerator_n[2]])
                remove_factor_numerator(node.args[1], [common_numerator_n[0], common_numerator_n[1], common_numerator_n[2]])

                injecting_node = node

                # # inject common factor after the node
                for factor in common_numerator_n:
                    print(factor)
                    if factor == -1:
                        print("get neg!!!!")
                        neg_node = inject_neg(injecting_node, graph, injecting_node, tmp_node)
                        injecting_node.replace_all_uses_with(neg_node)
                        neg_node.replace_input_with(tmp_node, injecting_node)
                        injecting_node = neg_node
                    else:
                        # check dimension match
                        injecting_node_shape = list(
                            injecting_node.meta['tensor_meta'].shape)
                        if isinstance(factor, fx.Node):
                            factor_shape = list(
                                factor.meta['tensor_meta'].shape)
                        else:
                            factor_shape = []
                        if (len(factor_shape) == len(injecting_node_shape)
                            or len(factor_shape) == 0
                            or len(injecting_node_shape) == 0):
                            mul_node = inject_mul(injecting_node, graph, injecting_node, factor, tmp_lhs=tmp_node)
                            injecting_node.replace_all_uses_with(mul_node)
                            mul_node.replace_input_with(tmp_node, injecting_node)
                            injecting_node = mul_node
                        else:
                            # get the dimensions to unsqueeze
                            unsqueeze_dims = []
                            factor_idx = 0
                            for idx, injecting_dim in enumerate(injecting_node_shape):
                                factor_dim = factor_shape[factor_idx]
                                if (factor_dim != injecting_dim
                                    and factor_dim != 1 and injecting_dim != 1):
                                    unsqueeze_dims.append(idx)
                            
                            # inject unsqueeze node
                            unsqueeze_node = injecting_node
                            for dim in unsqueeze_dims:
                                unsqueeze_node = inject_unsqueeze(unsqueeze_node, graph, factor, dim)

                            # inject mul node
                            mul_node = inject_mul(unsqueeze_node, graph, injecting_node, unsqueeze_node, tmp_lhs=tmp_node)
                            injecting_node.replace_all_uses_with(mul_node)
                            mul_node.replace_input_with(tmp_node, injecting_node)
                            injecting_node = mul_node
    
    graph.eliminate_dead_code()
    graph.lint()



################################################################################
# Graph-level pass to update data type of module attributes
################################################################################

def pass_update_attributes(module, graph):
    for node in graph.nodes:
        if node.op == "get_attr":
            attr = node.target
            setattr(module, attr, getattr(module, attr).to(torch.float16).to("cuda"))


def pre_partition_optimization(joint_module):
    # get graph
    graph = joint_module.graph
    # pass: loss elimination
    pass_loss_elimination(joint_module, graph)

    # pass: composed op breakdown
    pass_composed_op_breakdown(joint_module, graph)

    # pass: remove duplicated nodes
    pass_remove_duplicated_node(joint_module, graph)

    # pass: merge common factor of add / sub operators
    pass_merge_common_factor(joint_module, graph)

    # pass: update attributes
    pass_update_attributes(joint_module, graph)
    
    # recompile graph
    joint_module.recompile()

    # visualize graph
    pass_print_graph(joint_module)


