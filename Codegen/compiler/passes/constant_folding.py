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
from functorch._src.aot_autograd import _is_primal, _is_tangent, _extract_graph_with_inputs_outputs
from passes.print_graph import pass_print_graph


################################################################################
# Graph-level pass to perform constant folding
################################################################################

def neighbor_constant_folding(module, graph):
    for node in graph.nodes:
        if node.op == "get_attr":
            attr = node.target
            users = list(node.users.keys())
            if len(users) == 1:
                user_op = users[0].target
                if user_op in [torch.ops.aten.unsqueeze,]:
                    setattr(module, attr, user_op(getattr(module, attr), *users[0].args[1:]))
                    users[0].replace_all_uses_with(node)
                elif user_op in [torch.ops.aten.div,]:
                    if len(users[0].all_input_nodes) == 1:
                        args = users[0].args
                        func_args = []
                        for arg in args:
                            if arg == node: func_args.append(getattr(module, attr))
                            else: func_args.append(arg)

                        setattr(module, attr, user_op(*func_args))
                        users[0].replace_all_uses_with(node)
        elif node.op == "call_function":
            input_nodes = node.all_input_nodes
            _is_constant_node = True
            for input in input_nodes:
                if input.op != "get_attr":
                    _is_constant_node = False
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




def propagate_constant_folding(module, graph):
    tmp_node = None
    for idx, node in enumerate(graph.nodes):
        if idx == 0:
            tmp_node = node
        if node.target in [torch.ops.aten.div, torch.ops.aten.mul, torch.ops.aten.neg]:
            if len(node.all_input_nodes) == 1: 
                # propagate the single-argument node closer to constant
                parent = node.all_input_nodes[0]
                if len(parent.users) != 1: continue
                if parent.target == torch.ops.aten.add:
                    lhs_factors_n, lhs_factors_d = extract_factor(parent.args[0])
                    rhs_factors_n, rhs_factors_d = extract_factor(parent.args[1])
                    lhs_constant_factor = None
                    rhs_constant_factor = None
                    for factor in lhs_factors_n:
                        if isinstance(factor, fx.Node):
                            if factor.op == "get_attr":
                                lhs_constant_factor = factor
                    
                    for factor in rhs_factors_n:
                        if isinstance(factor, fx.Node):
                            if factor.op == "get_attr":
                                rhs_constant_factor = factor
                    
                    if lhs_constant_factor is not None and rhs_constant_factor is not None:
                        args = node.args
                        lhs_func_args = []
                        rhs_func_args = []
                        for arg in args:
                            if arg == parent: 
                                lhs_func_args.append(getattr(module, lhs_constant_factor.target))
                                rhs_func_args.append(getattr(module, rhs_constant_factor.target))
                            else:
                                lhs_func_args.append(arg)
                                rhs_func_args.append(arg)
                        setattr(module, lhs_constant_factor.target, node.target(*lhs_func_args))
                        setattr(module, rhs_constant_factor.target, node.target(*rhs_func_args))
                        node.replace_all_uses_with(parent)
                elif parent.target in [torch.ops.aten.view, torch.ops.aten.sum, torch.ops.aten.expand, torch.ops.aten.unsqueeze]:
                    raise NotImplementedError
        elif node.target in [torch.ops.aten.sum,]:
            parent = node.all_input_nodes[0]
            if len(parent.users) != 1: continue
            reduction_dim = node.args[1]
            reduction_length = parent.meta['tensor_meta'].shape[reduction_dim]
            if parent.target in [torch.ops.aten.mul, torch.ops.aten.add]:
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
                
                if parent.target in [torch.ops.aten.add,]:
                    print(reduction_length)
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
            print(getattr(submodule, attr))
            env[node] = graph.node_copy(node)
            inject_point = env[node]
        elif node.op == "output":
            replaced_node.replace_all_uses_with(env[node.args[0][0]])
    pass

def constant_graph_folding(module, graph):
    for node in graph.nodes:
        if node.target in [torch.ops.aten.sum,]:
            primal_inputs = list(filter(_is_primal, module.graph.nodes))
            tangent_inputs = list(filter(_is_tangent, module.graph.nodes))
            subgraph = _extract_graph_with_inputs_outputs(graph, primal_inputs + tangent_inputs, [node,])
            submodule = fx.GraphModule(module, subgraph)

            # optimize the sub module
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

            # pass_print_graph(submodule, "./sub_module.svg")
            inject_subgraph(node, node, module, graph, submodule, subgraph)
            break



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