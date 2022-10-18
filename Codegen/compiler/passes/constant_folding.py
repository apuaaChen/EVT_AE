################################################################################
# Copyright [yyyy] [name of copyright owner]
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
################################################################################
import torch
from nodes import *
from functorch._src.partitioners import _is_primal, _is_tangent, _extract_graph_with_inputs_outputs
from passes.print_graph import pass_print_graph
from passes.extract_common_factor import extract_factor


################################################################################
# Graph-level pass to perform constant folding
################################################################################

def neighbor_constant_folding(module, graph):
    """
    Folding direct constant node with pattern “op(const)”
    """
    for node in graph.nodes:
        if node.op == "call_function":
            # Look for call_functions nodes
            input_nodes = node.all_input_nodes
            _is_constant_node = True
            # check if all its input nodes are constant
            for input in input_nodes:
                if input.op != "get_attr":
                    _is_constant_node = False
            # for constant node: inject new constant attr
            if _is_constant_node:
                args = node.args
                func_args = []
                for arg in args:
                    if arg in input_nodes:
                        func_args.append(getattr(module, arg.target))
                    else:
                        func_args.append(arg)
                tensor = node.target(*func_args)
                const_node = inject_get_attr(node, module, graph, tensor, tensor_name="const_" + str(node))
                node.replace_all_uses_with(const_node)
            else:
                # mul with 1 and add with 0 are also constant nodes
                if node.target == torch.ops.aten.mul:
                    lhs = node.args[0]
                    rhs = node.args[1]
                    if lhs.op == "get_attr":
                        tensor = getattr(module, lhs.target)
                        if tensor.numel() == 1:
                            if tensor.item() == 1:
                                node.replace_all_uses_with(rhs)
                    
                    if rhs.op == "get_attr":
                        tensor = getattr(module, rhs.target)
                        if tensor.numel() == 1:
                            if tensor.item() == 1:
                                node.replace_all_uses_with(lhs)
                elif node.target == torch.ops.aten.div:
                    denominator = node.args[1]
                    if denominator.op == "get_attr":
                        tensor = getattr(module, denominator.target)
                        if tensor.numel() == 1:
                            if tensor.item() == 1:
                                node.replace_all_uses_with(denominator)
                if node.target in [torch.ops.aten.add, torch.ops.aten.sub]:
                    lhs = node.args[0]
                    rhs = node.args[1]
                    if lhs.op == "get_attr":
                        tensor = getattr(module, lhs.target)
                        if tensor.numel() == 1:
                            if tensor.item() == 0:
                                node.replace_all_uses_with(rhs)
                    
                    if rhs.op == "get_attr":
                        tensor = getattr(module, rhs.target)
                        if tensor.numel() == 1:
                            if tensor.item() == 0:
                                node.replace_all_uses_with(lhs)
                
                # constant fusing of transpose
                if node.target == torch.ops.aten.t:
                    if node.args[0].target == torch.ops.aten.t:
                        node.replace_all_uses_with(node.args[0].args[0])


def get_constant_factor(node):
    """
    Recursively extract constant factors to reduce other constant factor
    For each branch, we extract the nearest factor in each branch
    The numerator is explored first, then denominator
    """
    # factors in both numerator and denominator can be considered
    factors_numerator, factors_denumerator = extract_factor(node)
    # traverse the numerator first
    for factor in reversed(factors_numerator):
        if isinstance(factor, fx.Node):
            if factor.op == "get_attr":
                return [factor,], []
            elif factor.op == "call_function":
                if factor.target in [torch.ops.aten.add, torch.ops.aten.sub]:
                    lhs_n, lhs_d = get_constant_factor(factor.args[0])
                    rhs_n, rhs_d = get_constant_factor(factor.args[1])
                    if len(lhs_n) + len(lhs_d) > 0 and len(rhs_n) + len(rhs_d) > 0:
                        return lhs_n + rhs_n, lhs_d + rhs_d
    
    for factor in reversed(factors_denumerator):
        if isinstance(factor, fx.Node):
            if factor.op == "get_attr":
                return [], [factor,]
            elif factor.op == "call_function":
                if factor.target in [torch.ops.aten.add, torch.ops.aten.sub]:
                    lhs_n, lhs_d = get_constant_factor(factor.args[0])
                    rhs_n, rhs_d = get_constant_factor(factor.args[1])
                    if len(lhs_n) + len(lhs_d) > 0 and len(rhs_n) + len(rhs_d) > 0:
                        return lhs_d + rhs_d, lhs_n + rhs_n
    
    return [], []
    

denominator_funcs = {
    torch.ops.aten.div: torch.ops.aten.mul,
    torch.ops.aten.sum: torch.ops.aten.div,
    torch.ops.aten.neg: torch.ops.aten.neg
}


def propagate_constant_folding(module, graph):
    """
    Folding indirect constant node with pattern “op(op(const, var))”
    """
    tmp_node = None
    for idx, node in enumerate(graph.nodes):
        if idx == 0:
            tmp_node = node
        # reduction of constant factors
        if node.target in [torch.ops.aten.div, torch.ops.aten.mul, torch.ops.aten.neg]:
            _partial_constant_node = False
            # check if all its input nodes are constant
            non_constant_node = None
            input_nodes = node.all_input_nodes
            for input in input_nodes:
                if input.op == "get_attr":
                    _partial_constant_node = True
                else:
                    non_constant_node = input
            
            if _partial_constant_node and non_constant_node is not None:
                # get constant factors to update
                factors_numerator, factors_denominator = get_constant_factor(non_constant_node)
                # apply factors to numerator and denominators
                if len(factors_numerator) + len(factors_denominator) > 0:
                    args = node.args
                    func_args = []
                    for arg in args:
                        if arg == non_constant_node:
                            func_args.append(getattr(module, factor.target))
                        else:
                            func_args.append(arg)
                    for factor in factors_numerator:
                        setattr(module, factor.target, node.target(*func_args))
                    for factor in factors_denominator:
                        setattr(
                            module, factor.target, 
                            denominator_funcs[node.target](*func_args))
                    node.replace_all_uses_with(non_constant_node)
        # TODO: add / sub nodes
        # sum node
        elif node.target in [torch.ops.aten.sum,]:
            # sum node is propagated to reduce the tensor size as early as possible.
            parent = node.all_input_nodes[0]
            # sum node can only be propagated to single child parent
            if len(parent.users) != 1: continue
            # get reduction dimension
            reduction_dim = node.args[1]
            if len(reduction_dim) == 1:
                reduction_dim = node.args[1][0]
            # get the length of reduction
            reduction_length = 1
            try:
                reduction_length *= parent.meta['tensor_meta'].shape[reduction_dim]
            except:
                for dim in reduction_dim:
                    reduction_length *= parent.meta['tensor_meta'].shape[dim]
                

            if parent.target in [
                torch.ops.aten.mul, torch.ops.aten.div, 
                torch.ops.aten.add, torch.ops.aten.sub]:
                #
                lhs = parent.args[0]
                rhs = parent.args[1]
                if isinstance(lhs, fx.Node):
                    try:
                        lhs_dim = lhs.meta['tensor_meta'].shape[reduction_dim]
                    except:
                        lhs_dim = 1
                else: 
                    lhs_dim = 1

                if isinstance(rhs, fx.Node): 
                    try:
                        rhs_dim = rhs.meta['tensor_meta'].shape[reduction_dim]
                    except:
                        rhs_dim = 1
                else: 
                    rhs_dim = 1

                # sum node is not switchable in this case
                if lhs_dim != 1 and rhs_dim != 1: continue

                if lhs_dim != 1:
                    node.replace_all_uses_with(parent)
                    sum_node = inject_sum(lhs, graph, lhs, reduction_dim, tmp_node)
                    lhs.replace_all_uses_with(sum_node)
                    sum_node.replace_input_with(tmp_node, lhs)
                elif rhs_dim != 1:
                    node.replace_all_uses_with(parent)
                    sum_node = inject_sum(rhs, graph, rhs, reduction_dim, tmp_node)
                    rhs.replace_all_uses_with(sum_node)
                    sum_node.replace_input_with(tmp_node, rhs)
                
                if parent.target in [torch.ops.aten.add, torch.ops.aten.sub]:
                    if lhs_dim == 1:
                        mul_node = inject_mul(lhs, graph, lhs, reduction_length, tmp_lhs=tmp_node)
                        lhs.replace_all_uses_with(mul_node)
                        mul_node.replace_input_with(tmp_node, lhs)
                    elif rhs_dim == 1:
                        mul_node = inject_mul(rhs, graph, rhs, reduction_length, tmp_rhs=tmp_node)
                        rhs.replace_all_uses_with(mul_node)
                        mul_node.replace_input_with(tmp_node, rhs)
            elif parent.target == torch.ops.aten.one_hot:
                reduced_onehot = inject_get_attr(
                    parent, module, graph, 
                    torch.Tensor([1.0,]).to("cuda").requires_grad_().to(torch.float16), 
                    "reduced_onehot")
                
                node.replace_all_uses_with(reduced_onehot)



def inject_subgraph(inject_point, replaced_node, module, graph, submodule, subgraph):
    """
    Helper function to replace a node (and its parent link) with a subgraph
    """
    # get original place holders
    placeholders = {}
    for node in graph.nodes:
        if node.op == "placeholder":
            placeholders[node.target] = node

    env = {}
    for node in subgraph.nodes:
        # Skip placeholder nodes
        if node.op == "placeholder":
            env[node] = placeholders[node.target]
        elif node.op == "call_function":
            graph.inserting_after(inject_point)
            env[node] = graph.node_copy(node, lambda x: env[x])
            inject_point = env[node]
        elif node.op == "get_attr":
            graph.inserting_after(inject_point)
            attr = node.target
            if not hasattr(module, attr):
                setattr(module, attr, getattr(submodule, attr))
            env[node] = graph.node_copy(node)
            inject_point = env[node]
        elif node.op == "output":
            replaced_node.replace_all_uses_with(env[node.args[0][0]])


def constant_graph_folding(module, graph):
    """
    Indirect folding a constant node by analyzing its generating graph
    It is used to explore opportunities for non-single child 
    (propagation doesn't work in these case)
    """
    for node in graph.nodes:
        # identify non-single child
        parents = node.all_input_nodes
        branched_parent = False
        for parent in parents:
            if len(parent.users) != 1:
                branched_parent = True
                break
        if not branched_parent: continue
        # TODO: The rule could be applied to general node cases
        if node.target in [torch.ops.aten.sum,]:
            # extract input nodes of the graph
            primal_inputs = list(filter(_is_primal, module.graph.nodes))
            tangent_inputs = list(filter(_is_tangent, module.graph.nodes))
            num_input_nodes = len(primal_inputs) + len(tangent_inputs)
            # extract the submodule and graph
            subgraph = _extract_graph_with_inputs_outputs(graph, primal_inputs + tangent_inputs, [node,])
            submodule = fx.GraphModule(module, subgraph)

            # pass_print_graph(submodule, "./sub_module.svg")

            # optimize the sub module
            # The submodule should be optimized recursively.
            propagate_constant_folding(submodule, submodule.graph)
            submodule.graph.eliminate_dead_code()
            submodule.graph.lint()

            propagate_constant_folding(submodule, submodule.graph)
            submodule.graph.eliminate_dead_code()
            submodule.graph.lint()

            propagate_constant_folding(submodule, submodule.graph)
            submodule.graph.eliminate_dead_code()
            submodule.graph.lint()

            propagate_constant_folding(submodule, submodule.graph)
            submodule.graph.eliminate_dead_code()
            submodule.graph.lint()

            propagate_constant_folding(submodule, submodule.graph)
            submodule.graph.eliminate_dead_code()
            submodule.graph.lint()

            neighbor_constant_folding(submodule, submodule.graph)
            submodule.graph.eliminate_dead_code()
            submodule.graph.lint()

            neighbor_constant_folding(submodule, submodule.graph)
            submodule.graph.eliminate_dead_code()
            submodule.graph.lint()

            num_subgraph_nodes = len(subgraph.nodes) - num_input_nodes - 1
            if num_subgraph_nodes == 1:
                # only inject the subgraph nodes if it is simpler (only one node)
                # pass_print_graph(submodule, "./sub_module.svg")
                inject_subgraph(node, node, module, graph, submodule, subgraph)
            



def pass_constant_folding(module, graph):
    neighbor_constant_folding(module, graph)
    graph.eliminate_dead_code()
    neighbor_constant_folding(module, graph)
    graph.eliminate_dead_code()

    propagate_constant_folding(module, graph)
    graph.eliminate_dead_code()

    constant_graph_folding(module, graph)
    graph.eliminate_dead_code()

    neighbor_constant_folding(module, graph)
    graph.eliminate_dead_code()
    
    graph.lint()