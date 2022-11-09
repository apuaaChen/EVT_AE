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

################################################################################
# Graph-level pass to merge common factor of add / sub operators
################################################################################

def extract_factor(node):
    """
    Helper function to extract the factors of a node in numerator and denominator
    Params:
        node: the node whose factors to extract
    Returns:
        numerator_factor (list), denominator_factor (list)
    """
    # return numerator and demoninator factors
    if isinstance(node, float) or isinstance(node, int):
        return [node,], []
    if isinstance(node, torch.fx.Node):
        if node.op == "call_function":
            # element-wise mul / div node: 
            # join the numerator and denominator factor list
            if node.target == torch.ops.aten.mul:
                lhs_factor_n, lhs_factor_d = extract_factor(node.args[0])
                rhs_factor_n, rhs_factor_d = extract_factor(node.args[1])
                return lhs_factor_n + rhs_factor_n, lhs_factor_d + rhs_factor_d
            elif node.target == torch.ops.aten.div:
                lhs_factor_n, lhs_factor_d = extract_factor(node.args[0])
                rhs_factor_n, rhs_factor_d = extract_factor(node.args[1])
                return lhs_factor_n + rhs_factor_d, lhs_factor_d + rhs_factor_n
            # neg node is equivalent with multiplying with -1
            elif node.target == torch.ops.aten.neg:
                factor_n, factor_d = extract_factor(node.args[0])
                return [-1,] + factor_n, factor_d
            # the factors are transparent to these nodes
            elif node.target in [torch.ops.aten.view, torch.ops.aten.expand, torch.ops.aten.unsqueeze]:
                return extract_factor(node.args[0])
            # sum node is only transparent to the factors along non-reduction dimensions
            elif node.target in [torch.ops.aten.sum]:
                factor_n, factor_d = extract_factor(node.args[0])
                new_factor_n = []
                new_factor_d = []
                redution_dim = node.args[1][0]
                for factor in factor_n:
                    factor_shape = get_shape(factor)
                    try:
                        if factor_shape[redution_dim] != 1:
                            continue
                    except:
                        pass
                    new_factor_n.append(factor)
                for factor in factor_d:
                    factor_shape = get_shape(factor)
                    try:
                        if factor_shape[redution_dim] != 1:
                            continue
                    except:
                        pass
                    new_factor_d.append(factor)
                return new_factor_n, new_factor_d
            else:
                return [node, ], []
        elif node.op in ["get_attr", "placeholder"]:
            return [node, ], []
    else:
        print(node)
        raise TypeError

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
                    # case: lhs node is common factor
                    #       replace the mul node with rhs node
                    #       continue checking rhs node
                    node.replace_all_uses_with(node.args[1])
                    return remove_factor_numerator(node.args[1], common_factor)
                elif (node.args[0] not in common_factor 
                    and node.args[1] in common_factor):
                    # case: rhs node is common factor
                    #       replace the mul node with lhs node
                    #       continue checkong lhs node
                    node.replace_all_uses_with(node.args[0])
                    return remove_factor_numerator(node.args[0], common_factor)
                elif (node.args[0] in common_factor 
                    and node.args[1] in common_factor):
                    # case: both lhs and rhs are common factor
                    return True
                else:
                    # case: no direct common factor in lhs and rhs branch
                    lhs_empty = remove_factor_numerator(node.args[0], common_factor)
                    rhs_empty = remove_factor_numerator(node.args[1], common_factor)
                    if lhs_empty and not rhs_empty:
                        # case: lhs is indirect common factor
                        node.replace_all_uses_with(node.args[1])
                        return rhs_empty
                    elif rhs_empty and not lhs_empty:
                        # case: rhs is indirect common factor
                        node.replace_all_uses_with(node.args[0])
                        return lhs_empty
                    else:
                        # case: both are or are not common factors
                        return (lhs_empty and rhs_empty)
            elif node.target == torch.ops.aten.div:
                if node.args[0] in common_factor:
                    # case: numerator is common factor
                    #       replace numerator with 1
                    #       remove common factor in denominator's denominator
                    node.replace_input_with(node.args[0], 1)
                    return remove_factor_denominator(node.args[1], common_factor)
                else:
                    # case: others
                    #       remove common factor in numerator of numerator
                    #       remove common factor in denominator of denominator
                    lhs_empty = remove_factor_numerator(node.args[0], common_factor)
                    rhs_empty = remove_factor_denominator(node.args[1], common_factor)
                    if lhs_empty and not rhs_empty:
                        # case: numerator is empty
                        node.replace_input_with(node.args[0], 1)
                        return rhs_empty
                    elif rhs_empty and not lhs_empty:
                        # case: denominator is empty
                        node.replace_all_uses_with(node.args[1])
                        return lhs_empty
                    else:
                        ## case: both are or are not common factors
                        return (lhs_empty and rhs_empty)
            elif node.target == torch.ops.aten.neg:
                # case neg node
                if -1 in common_factor:
                    node.replace_all_uses_with(node.args[0])
                    return remove_factor_numerator(node.args[0], common_factor)
                else:
                    empty_child = remove_factor_numerator(node.args[0], common_factor)
                    if empty_child:
                        node.replace_all_uses_with(-1)
                    return False
            # transparent nodes
            elif node.target in [torch.ops.aten.view, torch.ops.aten.sum, torch.ops.aten.expand, torch.ops.aten.unsqueeze]:
                return remove_factor_numerator(node.args[0], common_factor)

def remove_factor_denominator(node, common_factor):
    if isinstance(node, torch.fx.Node):
        # We can only get common factor in denominator in div node
        if node.op == "call_function":
            # mul & div
            if node.target == torch.ops.aten.mul:
                # case: mul node
                lhs_empty = remove_factor_denominator(node.args[0], common_factor)
                rhs_empty = remove_factor_denominator(node.args[1], common_factor)
                if lhs_empty and not rhs_empty:
                    node.replace_all_uses_with(node.args[1])
                elif rhs_empty and not lhs_empty:
                    node.replace_all_uses_with(node.args[0])

                return (rhs_empty and lhs_empty)
            elif node.target == torch.ops.aten.div:
                # case: div node
                if node.args[1] in common_factor:
                    # case: denominator is common factor
                    node.replace_all_uses_with(node.args[0])
                    return remove_factor_denominator(node.args[0], common_factor)
                else:
                    lhs_empty = remove_factor_denominator(node.args[0], common_factor)
                    rhs_empty = remove_factor_numerator(node.args[1], common_factor)
                    if lhs_empty and not rhs_empty:
                        node.replace_input_with(node.args[0], 1)
                        return rhs_empty
                    elif rhs_empty and not lhs_empty:
                        node.replace_all_uses_with(node.args[1])
                        return lhs_empty
                    else:
                        return (lhs_empty and rhs_empty)
                    
            # transparent nodes
            if node.target in [
                torch.ops.aten.neg, torch.ops.aten.view, torch.ops.aten.sum, 
                torch.ops.aten.expand, torch.ops.aten.unsqueeze]:
                #
                return remove_factor_denominator(node.args[0], common_factor)
    else:
        return False
                     
def intersection(lst1, lst2):
    return [value for value in lst1 if value in lst2]

def add_factor(node, graph, factors, inject_op, tmp_node):
    injecting_node = node

    # # inject common factor after the node
    for factor in factors:
        if factor == -1:
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
                mul_node = inject_op(injecting_node, graph, injecting_node, factor, tmp_lhs=tmp_node)
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
                mul_node = inject_op(unsqueeze_node, graph, injecting_node, unsqueeze_node, tmp_lhs=tmp_node)
                injecting_node.replace_all_uses_with(mul_node)
                mul_node.replace_input_with(tmp_node, injecting_node)
                injecting_node = mul_node

def pass_merge_common_factor(module, graph):
    """
    Extract the common factor of add/sub nodes
    Params:
        module: fx.GraphModule
        graph: fx.Graph
    Returns: None
    """
    # The tmp node
    tmp_node = None
    for idx, node in enumerate(graph.nodes):
        if idx == 0:
            # register tmp node
            tmp_node = node
        if node.op == "call_function":
            # get the add and sub nodes
            if node.target in [torch.ops.aten.add, torch.ops.aten.sub]:
                # extract factors of lhs child and rhs child
                lhs_factors_n, lhs_factors_d = extract_factor(node.args[0])
                rhs_factors_n, rhs_factors_d = extract_factor(node.args[1])

                # Step 1: simplify the denominator
                lhs_common_factor = intersection(lhs_factors_n, lhs_factors_d)
                rhs_common_factor = intersection(rhs_factors_n, rhs_factors_d)
                remove_factor_numerator(node.args[0], lhs_common_factor)
                remove_factor_denominator(node.args[0], lhs_common_factor)
                remove_factor_numerator(node.args[1], rhs_common_factor)
                remove_factor_denominator(node.args[1], rhs_common_factor)

                # Step 2: remove common numerators
                common_numerator = intersection(lhs_factors_n, rhs_factors_n)
                remove_factor_numerator(node.args[0], common_numerator)
                remove_factor_numerator(node.args[1], common_numerator)

                # Step 3: remove common denominators
                common_denominator = intersection(lhs_factors_d, rhs_factors_d)
                remove_factor_denominator(node.args[0], common_denominator)
                remove_factor_denominator(node.args[1], common_denominator)

                # Step 4: append the factors
                add_factor(node, graph, common_numerator, inject_mul, tmp_node)
                add_factor(node, graph, common_denominator, inject_div, tmp_node)
    
    graph.eliminate_dead_code()
    graph.lint()
