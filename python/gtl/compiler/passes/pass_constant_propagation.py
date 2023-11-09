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
# Tensor Constant Propagation Pass
"""
The constant folding pass takes two major steps:
1. ReductionElimination: Simplify the reduction nodes (e.g. sum) to element-wise functions
2. Reassociation: General constant-folding + expression reassociation
"""
from typing import Optional, Union
from torch.fx.graph_module import GraphModule
from torch.fx import Node, Graph
from torch.fx.passes.tools_common import legalize_graph
from torch.fx.passes.infra.pass_base import PassBase, PassResult
from enum import Enum
import torch
from torch._functorch.partitioners import _extract_graph_with_inputs_outputs
from torch.fx.subgraph_rewriter import replace_pattern
from torch.utils._python_dispatch import _pop_mode_temporarily, _len_torch_dispatch_stack
from contextlib import nullcontext
from gtl.compiler.nodes import inject_get_attr, inject_squeeze
from gtl.compiler.passes.pass_fake_shape_infer import pass_fake_shape_infer
from torch.fx.passes.shape_prop import TensorMetadata


def get_shape(node: Union[Node, float, int]) -> list:
    if isinstance(node, Node):
        if "tensor_meta" in node.meta:
            meta = node.meta["tensor_meta"]
            if isinstance(meta, TensorMetadata):
                return list(node.meta["tensor_meta"].shape)
            else:
                return list(node.meta["tensor_meta"][0].shape)
        else:
            return [1]  # return a fake tensor shape
    else:
        return [1]


################################################################################
# Permutation Progataion Pass
################################################################################

class ReshapeAbv:
    """
    The abstract value of the nodes
    Each node is represented as a tensor (fx.Node) and its permutation
    """
    def __init__(self, tensor: Node, shape: list=None):
        self.tensor: Node = tensor
        self.tensor_shape = get_shape(tensor)
        if shape is None:
            self.shape = get_shape(tensor)
        else:
            self.shape = shape
    
    def reshape(self, new_shape):
        return ReshapeAbv(self.tensor, new_shape)
    
    def try_folding(self, node: Node, graph):
        if self.shape == self.tensor_shape:
            node.replace_all_uses_with(self.tensor)
        else:
            graph.inserting_before(node)
            view_node = graph.call_function(torch.ops.aten.view,
                                               args=(self.tensor, self.shape))
            view_node.meta = {}
            view_node.meta["tensor_meta"] = node.meta["tensor_meta"]._replace()
            node.replace_all_uses_with(view_node)


class ReshapeFolding(PassBase):
    """
    This pass eliminate a chain of view by finding constant in it
    """
    def call(self, graph_module: GraphModule) -> PassResult | None:
        graph: Graph = graph_module.graph
        self.workspace = {}
        for node in graph.nodes:
            if node.target == torch.ops.aten.view:
                tensor_abv = self.workspace[node.args[0]]
                try:
                    self.workspace[node] = tensor_abv.reshape(node.args[1])
                except:
                    breakpoint()
            else:
                self.workspace[node] = ReshapeAbv(node)
        
        # Fold the permutations based on the abstract values
        for node in graph.nodes:
            if node.target == torch.ops.aten.view:
                self.workspace[node].try_folding(node, graph)
    
    def ensures(self, graph_module: GraphModule) -> None:
        graph_module.graph.eliminate_dead_code()

################################################################################
# Permutation Progataion Pass
################################################################################

class PermuteAbv:
    """
    The abstract value of the nodes
    Each node is represented as a tensor (fx.Node) and its permutation
    """
    def __init__(self, tensor: Node, permutation: list=None):
        self.tensor: Node = tensor
        self.shape = get_shape(tensor)
        if permutation is None:
            self.permutation = list(range(len(self.shape)))
        else:
            self.permutation = permutation
    
    def permute(self, indices):
        permutation = [self.permutation[idx] for idx in indices]
        return PermuteAbv(self.tensor, permutation)
    
    def try_folding(self, node: Node, graph):
        if self.permutation != list(range(len(self.shape))):
            # Fold to tensor->permute
            graph.inserting_before(node)
            permute_node = graph.call_function(torch.ops.aten.permute,
                                               args=(self.tensor, self.permutation))
            permute_node.meta = {}
            permute_node.meta["tensor_meta"] = node.meta["tensor_meta"]._replace()
            node.replace_all_uses_with(permute_node)
        else:
            # Fold to tensor
            node.replace_all_uses_with(self.tensor)
    
    def is_contiguous(self):
        return self.permutation == list(range(len(self.shape)))



class PermutationFolding(PassBase):
    """
    This pass eliminate a chain of permutation by finding constant in it
    for instance:
        Tensor[4096, 1024] - permute([1, 0]) - permute([0, 1])
                                   |                 |
                                   A                 B
    can be folded to
        Tensor[4096, 1024] - permute([1, 0])
                 |                 |
                 B                 A
    """
    def call(self, graph_module: GraphModule) -> PassResult | None:
        graph: Graph = graph_module.graph
        self.workspace = {}
        for node in graph.nodes:
            if node.target == torch.ops.aten.permute:
                tensor_abv = self.workspace[node.args[0]]
                try:
                    self.workspace[node] = tensor_abv.permute(node.args[1])
                except:
                    breakpoint()
            else:
                self.workspace[node] = PermuteAbv(node)
        
        # Fold the permutations based on the abstract values
        for node in graph.nodes:
            if node.target == torch.ops.aten.permute:
                self.workspace[node].try_folding(node, graph)
    
    def ensures(self, graph_module: GraphModule) -> None:
        graph_module.graph.eliminate_dead_code()

################################################################################
# Constant Propagation Pass
################################################################################
    
# # Define the semilattice
class SL(Enum):
    UNDEF = 0    # Top: undefined type
    NAC = 1      # Bottom: not a constant

class ConstantPropagation(PassBase):
    def call(self, graph_module: GraphModule) -> PassResult | None:
        graph = graph_module.graph

        for stmt in graph.nodes:
            self.transfer_function(stmt)
        
        # Inject the constant args
        for node in graph.nodes:
            if node.op == "call_function":
                if self.workspace[node] not in [SL.UNDEF, SL.NAC]:
                    const_node = inject_get_attr(
                        node, self.module, self.module.graph,
                        self.workspace[node],
                        "const_" + node.name)
                    node.replace_all_uses_with(const_node)
        # Special case for Div for strength reduction
        for node in graph.nodes:
            if node.target == torch.ops.aten.div:
                divisor = node.args[1]
                if isinstance(divisor, Node) and divisor.op == "get_attr":
                    with (_pop_mode_temporarily() 
                        if _len_torch_dispatch_stack() > 0 else nullcontext()):
                            multiplier = inject_get_attr(
                                divisor, self.module, self.module.graph,
                                1./getattr(self.module, divisor.target),
                                "inv_" + divisor.name
                            )
                else:
                    multiplier = 1./divisor
                
                graph.inserting_after(node)
                mul_node = graph.call_function(torch.ops.aten.mul, args=(node.args[0], multiplier))
                node.replace_all_uses_with(mul_node)


    def binary_transfer(self, abv1, abv2):
        if abv1 is None:
            if abv2 in [SL.UNDEF, SL.NAC]:
                return abv2
            else:
                return [abv2, ]
        if abv1 == SL.UNDEF:
            if abv2 == SL.NAC:
                return SL.NAC
            else:
                return SL.UNDEF
        elif abv1 == SL.NAC:
            return SL.NAC
        else:
            # abv1 is a specific constant tensor
            if abv2 in [SL.NAC, SL.UNDEF]:
                return abv2
            else:
                assert isinstance(abv1, list)
                return abv1 + [abv2,]

    def get_abv(self, node: Union[Node, int, float]):
        if isinstance(node, Node):
            return self.workspace[node]
        else:
            return node # constant immediate number  
    
    def transfer_function(self, stmt: Node):
        if stmt.op == "placeholder":
            self.workspace[stmt] = SL.NAC
        elif stmt.op == "get_attr":
            self.workspace[stmt] = getattr(self.module, stmt.target)
        elif stmt.op == "output":
            self.workspace[stmt] = SL.UNDEF
        elif stmt.op == "call_function":
            stmt_abv = None
            for arg in stmt.args:
                stmt_abv = self.binary_transfer(stmt_abv, self.get_abv(arg))
            if stmt_abv in [SL.UNDEF, SL.NAC]:
                self.workspace[stmt] = stmt_abv
            elif isinstance(stmt_abv, list):
                with (_pop_mode_temporarily() 
                if _len_torch_dispatch_stack() > 0 else nullcontext()):
                    self.workspace[stmt] = stmt.target(*stmt_abv, **stmt.kwargs)
            else:
                # A function without args
                with (_pop_mode_temporarily() 
                if _len_torch_dispatch_stack() > 0 else nullcontext()):
                    self.workspace[stmt] = stmt.target(**stmt.kwargs)
        else:
            raise ValueError(f"Unknown op code {stmt.op}")


    def requires(self, graph_module: GraphModule) -> None:
        self.module = graph_module
        self.workspace = {}
        
        graph = graph_module.graph
        # Initialize the workspace with UNDEF
        for node in graph.nodes:
            self.workspace[node] = SL.UNDEF


################################################################################
# Common factor extraction Pass
################################################################################

class FactorAbv:
    """
    The abstract value of the nodes
    Each node is represented as numerators / denominators
    both numerator and denominator are dict of {factor: num_occurance}, so that 
    we can model cases where the same factor is multiplied by multiple times
    
    In the following notations, we denode numerator of a as na, and denominator 
    of b as nb
    """
    def __init__(self, numerators: dict, denominators: dict) -> None:
        assert isinstance(numerators, dict)
        assert isinstance(denominators, dict)

        self.numerators: dict = numerators
        self.denominators: dict = denominators
    
    def mul(self, others):
        """
        (na/da) * (nb/db). We first find the common factor between (na, db) and 
        (da, nb) and eliminate the common factors in each branch
        Returns:
            the abstract value of the multiplication result
            factors to eliminate in na
            factors to eliminate in da
            factors to eliminate in nb
            factors to eliminate in db
        """
        assert isinstance(others, FactorAbv)
        # get common factor of (na, db) and (da, nb)
        inter_na_db = self.intersect_dicts(self.numerators, others.denominators)
        inter_da_nb = self.intersect_dicts(self.denominators, others.numerators)
        
        return FactorAbv(
            self.union_dicts(
                self.subtract_dicts(self.numerators, inter_na_db),
                self.subtract_dicts(others.numerators, inter_da_nb)
            ),
            self.union_dicts(
                self.subtract_dicts(self.denominators, inter_da_nb),
                self.subtract_dicts(others.denominators, inter_na_db)
            )
        ), inter_na_db, inter_da_nb, inter_da_nb, inter_na_db

    def div(self, others):
        """
        (na/da) / (nb/db). We first find the common factor between (na, nb) and 
        (da, db) and eliminate the common factors in each branch
        Returns:
            the abstract value of the division result
            factors to eliminate in na
            factors to eliminate in da
            factors to eliminate in nb
            factors to eliminate in db
        """
        assert isinstance(others, FactorAbv)
        inter_na_nb = self.intersect_dicts(self.numerators, others.numerators)
        inter_da_db = self.intersect_dicts(self.denominators, others.denominators)

        return FactorAbv(
            self.union_dicts(
                self.subtract_dicts(self.numerators, inter_na_nb),
                self.subtract_dicts(others.denominators, inter_da_db)
            ),
            self.union_dicts(
                self.subtract_dicts(self.denominators, inter_da_db),
                self.subtract_dicts(others.numerators, inter_na_nb)
            )
        ), inter_na_nb, inter_da_db, inter_na_nb, inter_da_db

    def add(self, others, node):
        """
        (na/da) + (nb/db) = common_factor(na, nb) / common_factor(da,db) * (..+..)
        Returns:
            the abstract value of the division result
            factors to eliminate in na
            factors to eliminate in da
            factors to eliminate in nb
            factors to eliminate in db
        """
        assert isinstance(others, FactorAbv)
        inter_na_nb = self.intersect_dicts(self.numerators, others.numerators)
        inter_da_db = self.intersect_dicts(self.denominators, others.denominators)
        
        return FactorAbv({node: 1}, {}), inter_na_nb, inter_da_db, inter_na_nb, inter_da_db

    def __eq__(self, others) -> bool:
        """
        Check if two abstract values are equal
        """
        # Check numerator
        for key, value in self.numerators.items():
            if key not in others.numerators:
                return False
            if others.numerators[key] != value:
                return False
        # Check denominator
        for key, value in self.denominators.items():
            if key not in others.denominators:
                return False
            if others.denominators[key] != value:
                return False
        return True

    #
    # Helper functions
    #

    def remove_numerator_factor(self, factor):
        """remove factor from numerator """
        assert factor in self.numerators
        if self.numerators[factor] == 1:
            del self.numerators[factor]
        else:
            self.numerators[factor] -= 1
    
    def remove_denominator_factor(self, factor):
        """ remove factor from denominator """
        assert factor in self.denominators
        if self.denominators[factor] == 1:
            del self.denominators[factor]
        else:
            self.denominators[factor] -= 1
    
    def union_dicts(self, dict1, dict2):
        """ get the union of two dicts """
        new_dict = {}
        for key in dict1:
            if key not in new_dict:
                new_dict[key] = dict1[key]
            else:
                new_dict[key] += dict1[key]
        
        for key in dict2:
            if key not in new_dict:
                new_dict[key] = dict2[key]
            else:
                new_dict[key] += dict2[key]
        return new_dict
    
    def intersect_dicts(self, dict1, dict2):
        """ get the intersect of two dicts """
        new_dict = {}
        for key in dict1:
            if key in dict2:
                new_dict[key] = min(dict1[key], dict2[key])
        return new_dict

    def subtract_dicts(self, dict1, dict2):
        """ dict 1 - dict 2 """
        new_dict = {}
        for key in dict1:
            if key in dict2:
                num_occur = dict1[key] - dict2[key]
                assert num_occur >= 0
                if num_occur > 0:
                    new_dict[key] = num_occur
            else:
                new_dict[key] = dict1[key]
        
        for key in dict2:
            assert key in dict1
        
        return new_dict
    
    def copy(self):
        numerators = {key: value for key, value in self.numerators.items()}
        denominators = {key: value for key, value in self.denominators.items()}
        return FactorAbv(numerators, denominators)


class CommonFactorExtraction(PassBase):
    """
    GTL pass that extract the common factor of the lhs & rhs operands of add &
    sub operators

    Method:
        The abstract value of each node is its factorization numerator & denominator. 
        e.g. n: a * b, abstract value: (a, b), ()
             n: a / b, abstract value: (a), (b)
             n: sin(a), abstract value sin(a)
    """
    def call(self, graph_module: GraphModule) -> PassResult | None:
        graph: Graph = graph_module.graph
        self.graph = graph
        self.workspace = {}
        for node in graph.nodes:
            if node.op == "call_function":
                target = str(node.target).split(sep='.')[-1]
                if hasattr(self, target):
                    getattr(self, target)(node)
                    continue
            self.workspace[node] = FactorAbv({node: 1}, {})
    
    def get_abv(self, arg):
        if isinstance(arg, Node):
            return self.workspace[arg]
        else:
            return FactorAbv({arg: 1}, {})

    def mul(self, node: Node):
        lhs, rhs = node.args
        abv_lhs = self.get_abv(lhs)
        abv_rhs = self.get_abv(rhs)
        abv_rst, rm_na, rm_da, rm_nb, rm_db = abv_lhs.mul(abv_rhs)
        self.stack = [node,]
        self.remove_factors_from_numerator(lhs, rm_na)
        self.remove_factors_from_numerator(rhs, rm_nb)
        self.remove_factors_from_denominator(lhs, rm_da)
        self.remove_factors_from_denominator(rhs, rm_db)
        self.workspace[node] = abv_rst
    
    def div(self, node: Node):
        lhs, rhs = node.args
        abv_lhs = self.get_abv(lhs)
        abv_rhs = self.get_abv(rhs)
        abv_rst, rm_na, rm_da, rm_nb, rm_db = abv_lhs.div(abv_rhs)
        self.stack = [node,]
        self.remove_factors_from_numerator(lhs, rm_na)
        self.remove_factors_from_numerator(rhs, rm_nb)
        self.remove_factors_from_denominator(lhs, rm_da)
        self.remove_factors_from_denominator(rhs, rm_db)
        self.workspace[node] = abv_rst
    
    def add(self, node: Node):
        lhs, rhs = node.args
        abv_lhs = self.get_abv(lhs)
        abv_rhs = self.get_abv(rhs)
        abv_rst, rm_na, rm_da, rm_nb, rm_db = abv_lhs.add(abv_rhs, node)
        self.stack = [node,]
        numerators = self.remove_factors_from_numerator(lhs, rm_na)
        self.remove_factors_from_numerator(rhs, rm_nb)
        denominators = self.remove_factors_from_denominator(lhs, rm_da)
        self.remove_factors_from_denominator(rhs, rm_db)
        self.workspace[node] = abv_rst

        self.add_factors_to_numerator(node, numerators)
        self.add_factors_to_denominator(node, denominators)
    
    def unsqueeze(self, node: Node):
        self.workspace[node] = self.workspace[node.args[0]].copy()
    
    def squeeze(self, node: Node):
        self.workspace[node] = self.workspace[node.args[0]].copy()

    
    def remove_factors_from_numerator(self, node: Node, factors):
        """
        Helper function that removes a list of factors from a branch's numerator
        and returns the list of removed factors
        """
        removed_numerator_factors = []
        for factor in factors:
            num_occur = factors[factor]
            for _ in range(num_occur):
                result = self.remove_numerator_factor_from_node(node, factor)
                assert result
                removed_numerator_factors.append(result)
        return removed_numerator_factors
    
    def remove_factors_from_denominator(self, node: Node, factors):
        """
        Helper function that removes a list of factors from a branch's denominator
        and returns the list of removed factors
        """
        removed_denominator_factors = []
        for factor in factors:
            num_occur = factors[factor]
            for _ in range(num_occur):
                result = self.remove_denominator_factor_from_node(node, factor)
                assert result
                removed_denominator_factors.append(result)
        return removed_denominator_factors

    def remove_numerator_factor_from_node(self, node: Node, factor):
        """
        Remove a factor from a given node's numerator. A particular trick is used
        to handle the squeeze and unsqueeze nodes.

        For example, c = a * b. if we find that the abstract value of `a` equals
        to FactorAbv({factor: 1}, {}), then we directly replace c with b. So that
        even if the `a` is actually a squeeze/unsqueeze of a, we can still
        eliminate it without getting into it. 

        However, it may cause an issue that the factor's shape missmatch when we
        reapply the factor back. To address this issue, we directly return `a` 

        when moving the factors forward through a squeeze/unsqueeze node, we apply
        the node to it to make the dimension compatible
        """
        # If the factor is not in the numerator, return False
        if not factor in self.workspace[node].numerators:
            return False

        # For mul nodes
        if node.target == torch.ops.aten.mul:
            arg1, arg2 = node.args
            # Case 1: arg1 == factor
            if self.get_abv(arg1) == FactorAbv({factor: 1}, {}):
                self.stack[-1].replace_input_with(node, arg2)
                return arg1
            # Case 2: arg2 == factor
            elif self.get_abv(arg2) == FactorAbv({factor: 1}, {}):
                self.stack[-1].replace_input_with(node, arg1)
                return arg2
            else:
                # Case 3: the factor is removed from children branches
                self.stack.append(node)
                # first try lhs
                result = self.remove_numerator_factor_from_node(arg1, factor)
                if result:
                    self.workspace[node].remove_numerator_factor(factor)
                    self.stack.pop()
                    return result
                result = self.remove_numerator_factor_from_node(arg2, factor)
                if result:
                    self.workspace[node].remove_numerator_factor(factor)
                    self.stack.pop()
                    return result
                else:
                    return False

        elif node.target == torch.ops.aten.div:
            arg1, arg2 = node.args
            # Case 1: arg1 == factor
            if self.get_abv(arg1) == FactorAbv({factor: 1}, {}):
                self.stack[-1].replace_input_with(node, arg2)
                return arg1
            else:
                # Case 2: the factor is removed from children branches
                self.stack.append(node)
                result = self.remove_numerator_factor_from_node(arg1, factor)
                if result:
                    self.workspace[node].remove_numerator_factor(factor)
                    self.stack.pop()
                    return result
                result = self.remove_denominator_factor_from_node(arg2, factor)
                if result:
                    self.workspace[node].remove_numerator_factor(factor)
                    self.stack.pop()
                    return result
                else:
                    return False
        elif node.target in [torch.ops.aten.unsqueeze, torch.ops.aten.squeeze]:
            self.stack.append(node)
            result = self.remove_numerator_factor_from_node(node.args[0], factor)
            if result:
                self.workspace[node].remove_numerator_factor(factor)
                self.stack.pop()
                self.graph.inserting_after(result)
                new_node = self.graph.call_function(node.target, args=(result, *node.args[1:]))
                self.workspace[new_node]=self.get_abv(result)
                return new_node
            else:
                return False
        else:
            raise ValueError()
    
    def remove_denominator_factor_from_node(self, node: Node, factor):
        """
        Remove a factor from a given node's denominator
        """
        if not factor in self.workspace[node].denominators:
            return False

        if node.target == torch.ops.aten.mul:
            arg1, arg2 = node.args
            self.stack.append(node)
            # first try lhs
            result = self.remove_denominator_factor_from_node(arg1, factor)
            if result:
                self.workspace[node].remove_denominator_factor(factor)
                self.stack.pop()
                return result
            result = self.remove_denominator_factor_from_node(arg2, factor)
            if result:
                self.workspace[node].remove_denominator_factor(factor)
                self.stack.pop()
                return result
            else:
                return False
        
        elif node.target == torch.ops.aten.div:
            arg1, arg2 = node.args
            # Case 1: arg1 == factor
            if self.get_abv(arg2) == FactorAbv({factor: 1}, {}):
                self.stack[-1].replace_input_with(node, arg1)
                return arg2
            else:
                # Case 2: the factor is removed from children branches
                self.stack.append(node)
                # first try lhs
                result = self.remove_denominator_factor_from_node(arg1, factor)
                if result:
                    self.workspace[node].remove_denominator_factor(factor)
                    self.stack.pop()
                    return result
                result = self.remove_numerator_factor_from_node(arg2, factor)
                if result:
                    self.workspace[node].remove_denominator_factor(factor)
                    self.stack.pop()
                    return result
                else:
                    return False

        elif node.target in [torch.ops.aten.unsqueeze, torch.ops.aten.squeeze]:
            self.stack.append(node)
            result = self.remove_denominator_factor_from_node(node.args[0], factor)
            if result:
                self.workspace[node].remove_denominator_factor(factor)
                self.stack.pop()
                self.graph.inserting_after(result)
                new_node = self.graph.call_function(node.target, args=(result, *node.args[1:]))
                return new_node
            else:
                return False
        else:
            raise ValueError()
    
    def add_factors_to_numerator(self, node: Node, factors: list):
        for factor in factors:
            self.add_numerator_factor_to_node(node, factor)
    
    def add_factors_to_denominator(self, node: Node, factors: list):
        for factor in factors:
            self.add_denominator_factor_to_node(node, factor)

    def add_numerator_factor_to_node(self, node: Node, factor):
        self.graph.inserting_after(node)
        users = list(node.users)
        mul_node = self.graph.call_function(torch.ops.aten.mul, args=(node, factor))
        for user in users:
            user.replace_input_with(node, mul_node)
    
    def add_denominator_factor_to_node(self, node: Node, factor):
        self.graph.inserting_after(node)
        users = list(node.users)
        div_node = self.graph.call_function(torch.ops.aten.div, args=(node, factor))
        for user in users:
            user.replace_input_with(node, div_node)  


################################################################################
# Reduction Elimination Pass
################################################################################

class Reassociation(PassBase):
    """
    This pass if largely inspired by the implementation at
    https://opensource.apple.com/source/llvmCore/llvmCore-2118/lib/Transforms/Scalar/Reassociate.cpp.auto.html
    It reassociates commutative expressions in an order that is designed to 
    promote better constant propagation

    For example: 4 + (x + 5) -> x + (4 + 5)
    """

    def call(self, graph_module: GraphModule) -> PassResult:
        self.rank_map = {}
        ## Preprocessing
        # This step unifies the neg to mul, and sub to add to simplify logics
        self.replace_neg_with_mul(graph_module)
        self.replace_sub_with_add(graph_module)

        self.module = graph_module
        graph = graph_module.graph

        # Step 1: perform common factor extraction A*B+A*C = A*(B+C)
        # Unlike the original implementation that does this during the
        # reassociation, we perform it at the very begining.
        cfe = CommonFactorExtraction()
        cfe(graph_module)


        # Step 2: associate ranks to each node in the graph. The general rule is
        # as follows
        # 1. The input arguments (placeholders) are ranked incrementally from 2
        # 2. Constants are ranked as 0
        # 3. Functions are assigned the highest rank of its arguments
        # 4. Special case: one-hot is assigned rank-1, so if it exists, we can
        #    find it easily
        arg_rank = 2
        # Step 1: rank the nodes
        for node in graph.nodes:
            if node.op == "placeholder":
                self.rank_map[node] = arg_rank
                arg_rank += 1
            elif node.op == "call_function":
                if node.target == torch.ops.aten.one_hot:
                    # Special case for one-hot node
                    self.rank_map[node] = -1
                    self.rank_map[node.args[0]] = -1
                else:
                    rank = 0
                    for arg in node.args:
                        if isinstance(arg, Node):
                            rank = max(rank, self.rank_map[arg])
                        else:
                            # Constant
                            rank = max(rank, 0)
                    self.rank_map[node] = rank
            elif node.op == "get_attr":
                self.rank_map[node] = 0

        # Reassociate the input graph and perform constant folding
        # Inspect all of the nodes in this graph, reasociating them as we go
        for node in graph.nodes:
            # If this instruction is a commutative binary operator, process it
            if not self.is_associative(node):
                continue
            
            # If this is an interior node of a reassociable tree, ignore it 
            # until we get to the root of the tree, to avoid N^2 analysis
            users = list(node.users)
            if (len(users) == 1 and 
                self.is_reassociable_op(node, users[0].target)):
                continue
            
            self.reassociate_expression(node)
        
        # Constant propagation
        cp = ConstantPropagation()
        cp(self.module)

    def ensures(self, graph_module: GraphModule) -> None:
        # Cleanup
        legalize_graph(graph_module)
        graph_module.graph.eliminate_dead_code()
        graph_module.recompile()
        pass_fake_shape_infer(graph_module, graph_module.graph)
    
    #
    # Helper function
    #
    def get_rank(self, node: Union[Node, float, int]):
        if isinstance(node, Node):
            return self.rank_map[node]
        else:
            return 0

    #
    # Preprocessing helper functions
    #
    def replace_neg_with_mul(self, graph_module):
        def pattern(x):
            return torch.ops.aten.neg(x)
        
        def replacement(x):
            return torch.ops.aten.mul(x, -1.)   
        replace_pattern(graph_module, pattern, replacement)
    
    def replace_sub_with_add(self, graph_module):
        def pattern(x, y):
            return torch.ops.aten.sub(x, y)
        
        def replacement(x, y):
            neg_y = torch.ops.aten.mul(y, -1.)
            return torch.ops.aten.add(x, neg_y)
        replace_pattern(graph_module, pattern, replacement)
    
    #
    # Linearize the reassociable expression
    #

    def linearize_expr(self, node: Node):
        """
        For expression (A+B)+(C+D) where the node is the `+` in the middle
        Both of its lhs and rhs expressions are reassociable
        """
        lhs = node.args[0]
        rhs = node.args[1]

        assert self.is_reassociable_op(lhs, node.target)
        assert self.is_reassociable_op(rhs, node.target)

        # Change to node: (A+B) + C
        node.args = (lhs, rhs.args[0])
        # rhs = (A+B) + D
        rhs.args = (lhs, rhs.args[1])
        # node: ((A+B) + D) + C
        node.args = (rhs, node.args[1])

        # continue reassociate C is it is not reassociable
        if self.is_reassociable_op(node.args[1], node.target):
            self.linearize_expr(node)

    def linearize_expr_tree(self, root: Node, ops: list):
        """
        Given an associative binary expression tree, traverse all of the uses
        puting it into canonical form. This forces a left-linear form of the
        expression (((a+b)+c)+d), and collects information about the rank of
        the non-tree operands

        Note: this intentionally destroys the expression tree operands (
        turning them into undef values) to reduce #uses of the values. This 
        means that the caller MUST use something like rewirte_expr_tree to put
        the values back in
        """
        lhs = root.args[0]
        rhs = root.args[1]
        target = root.target

        # First step, linearize the expression if it is in ((A+B)+(C+D)) form
        lhsbo = self.is_reassociable_op(lhs, target)
        rhsbo = self.is_reassociable_op(rhs, target)

        if not lhsbo:
            if not rhsbo:
                # Either this LHS or RHS as part of the tree, thus this is a 
                # leaf. As such, just remember these operands and their rank
                ops.append((self.get_rank(lhs), lhs))
                ops.append((self.get_rank(rhs), rhs))
                root.args = (None, None)
                return
            else:
                # Turn X+(Y+Z) -> (Y+Z)+X
                root.args = (rhs, lhs)
                lhs, rhs = rhs, lhs
                lhsbo, rhsbo = rhsbo, lhsbo
        elif rhsbo:
            # Turn (A+B)+(C+D) -> (((A+B)+C)+D)
            self.linearize_expr(root)
            lhs = lhsbo = root.args[0]
            rhs = root.args[1]
            rhsbo = False
        
        assert not self.is_reassociable_op(rhs, target)
        self.linearize_expr_tree(lhs, ops)
        ops.append((self.get_rank(rhs), rhs))
        root.args = (root.args[0], None)
    
    def rewrite_expr_tree(self, node, ops, i):
        if i == 1:
            node.args = (ops[0][1], ops[1][1])
            return
        
        assert i>1
        node.args = (node.args[0], ops[i][1])
        node.meta = {}
        self.rewrite_expr_tree(node.args[0], ops, i-1)

    def optimize_expression(self, node: Node, ops: list):
        """
        Now that we have the linearized expression tree, try to optimize it
        Start by folding any constants that we found
        """
        # The linearized expression is like A+B+C+D+...
        # where ops=[A,B,C,D] that are sorted by rank

        # Optimization 1: Fold constant One-hot & Sum
        # as one-hot node is always assigned rank -1, if a 
        # one-hot node exist, it must be A
        if (isinstance(ops[0][1], Node) and
            ops[0][1].target == torch.ops.aten.one_hot and 
            len(node.users) == 1 and 
            list(node.users)[0].target == torch.ops.aten.sum):
            # Verify the reduction dimension
            sum_node = list(node.users)[0]
            onehot_node = ops[0][1]
            shape_onehot = get_shape(onehot_node)
            meta = onehot_node.meta["tensor_meta"]
            if sum_node.args[1] in [-1, len(shape_onehot) - 1]:
                # insert the constant node
                with _pop_mode_temporarily() if _len_torch_dispatch_stack() > 0 else nullcontext():
                    reduced_onehot = inject_get_attr(
                        onehot_node, self.module, self.module.graph,
                        torch.ones(
                            size=tuple([1] * (len(shape_onehot) - 1) + [1,]), # an additional dimension is added to match the dimensions
                            dtype=meta.dtype, device="cuda"),
                        "const_" + onehot_node.name)
                onehot_node = reduced_onehot
                # The sum_node requires a squeeze_node
                squeeze_node = inject_squeeze(
                    node, self.module.graph,node, sum_node.args[1]
                )
                sum_node.replace_all_uses_with(squeeze_node)
                self.module.graph.erase_node(sum_node)
                ops[0] = (0, reduced_onehot)
                self.rank_map[reduced_onehot] = 0

    def reassociate_expression(self, node: Node):
        # First, walk the expression tree, linearizing the tree, collecting
        ops = []
        # The ops contain tuples (rank, op)
        self.linearize_expr_tree(node, ops)

        ops = sorted(ops, key=lambda item: item[0])

        # optimization
        self.optimize_expression(node, ops)

        # Rewrite the expr tree
        self.rewrite_expr_tree(node, ops, len(ops)-1)
        self.module = legalize_graph(self.module)
        # Eliminate potential dead code
        self.module.graph.eliminate_dead_code()

    def is_associative(self, node: Node):
        # Only add & mul are reassociable so far
        if node.target in [torch.ops.aten.mul, torch.ops.aten.add]:
            return True
        return False
    
    def is_reassociable_op(self, node: Union[Node, float, int], target):
        if not isinstance(node, Node):
            return False
        if len(node.users) <= 1 and node.target == target:
            return node
        return False
        


################################################################################
# Reduction Elimination Pass
################################################################################

class ReductionElimination(PassBase):
    """
    Simplify the reduction nodes (e.g. sum) to element-wise functions
    The reduction nodes usually breaks the flow of fusion as they are only fusible
    for a perfectly-nested-loop compiler under the atomic reduction
    Seeking ways of representing the reduction with other forms of computation 
    would unlock new fusion opportunities
    """
    def call(self, graph_module: GraphModule) -> PassResult:
        graph: Graph = graph_module.graph

        sum_nodes = []

        for node in graph.nodes:
            if node.target in [torch.ops.aten.sum,]:
                sum_nodes.append(node)
        
        for node in sum_nodes:
            # Extract the subgraph producing the node
            submodule = self.extract_subgraph(graph_module, node)

            # Simplify the submodule with reassociation + constant propagation
            reas = Reassociation()
            reas(submodule)
            submodule.recompile()

            # Check if sum node is eliminated
            eliminated = True
            for n in submodule.graph.nodes:
                if n.target == torch.ops.aten.sum:
                    eliminated = False
                    break
            if not eliminated:
                # The reduction node still exists. Skip this node
                continue

            # Copy the whole subgraph into the original graph module
            # val_map: Dict[Node, Node] mapping from nodes in g to nodes in self
            val_map = {}
            # match the input nodes:
            subgraph_placeholders = {n.name: n for n in submodule.graph.nodes if n.op == "placeholder"}
            subgraph_attrs = [n.target for n in submodule.graph.nodes if n.op == "get_attr"]

            # When constructing the submodule, the placeholders inherit the names
            for input in self.input_nodes:
                val_map[subgraph_placeholders[input.name]] = input

            with graph.inserting_before(node):
                copied_returning_nodes = graph.graph_copy(submodule.graph, val_map)
            
            node.replace_all_uses_with(copied_returning_nodes[0])

            # Copy the attributes (folded constants) to graph_module
            for attr in subgraph_attrs:
                if not hasattr(graph_module, attr):
                    setattr(graph_module, attr, getattr(submodule, attr))
            
            # Remove the sum node
            graph.erase_node(node)

    def ensures(self, graph_module: GraphModule) -> None:
        # Cleanup 
        legalize_graph(graph_module)
        graph_module.graph.eliminate_dead_code()
        graph_module.recompile()
        pass_fake_shape_infer(graph_module, graph_module.graph)
    
    #
    # Helper functions
    #

    def backward_dfs(self, node: Node):
        """
        Find the input nodes of the current submodule
        """
        # Skip the visited nodes
        if node in self.visited:
            return
        self.visited.append(node)
        # Traverse the inputs of the current node
        for input in node.all_input_nodes:
            # If the input node is reassociable with sum, include it
            if self.is_reassociable_op(input, self.reduction_node):
                self.backward_dfs(input)
            # Otherwise, store the input node
            else:
                self.input_nodes.append(input)

    def extract_subgraph(self, graph_module: GraphModule, node: Node):
        """
        Extract the subgraph producing the current reduction node through a 
        series of reassociable ops with the current reduction node

        The idea is to perform backward DFS from the current reduction node.
        The DFS stops at unassociable nodes, and these unassociable are treated 
        as input nodes of the subgraph.

        With the input and output nodes, we use the built-in function 
        `_extract_graph_with_inputs_outputs` to generate the submodule to work on
        """

        # Step 1: get the input & output nodes
        self.input_nodes = []
        self.visited = []  # dfs buffer
        self.reduction_node: Node = node
        self.backward_dfs(node)
        output_nodes = [node]
        # Extract subgraph
        subgraph = _extract_graph_with_inputs_outputs(
            graph_module.graph, self.input_nodes, output_nodes
        )
        return GraphModule(graph_module, subgraph)

    def get_canonical_broadcast_shapes(self, target: Node, args: 'list[Node]'):
        shape_target = get_shape(target)
        shape_args = [get_shape(arg) for arg in args]
        shape_args = [
            [1] * (len(shape_target) - len(shape_arg)) + shape_arg 
            for shape_arg in shape_args]
        
        return shape_args


    def is_reassociable_op(self, node: Union[Node, float, int], target_node: Node):
        """
        Return true is node is reassociable with the target node
        """
        assert target_node.target == torch.ops.aten.sum
        # Only fx.Nodes are reassociable
        if not isinstance(node, Node):
            return False

        # Special cases when considering reassociation with `sum` node
        target_shape = get_shape(target_node.args[0])
        # Note: reduction dim could be a list of indices
        # We convert it to list to unify the logics
        reduction_dim = target_node.args[1]
        if not isinstance(reduction_dim, list):
            reduction_dim = [reduction_dim, ]

        # Placeholders, get_attrs are not associable
        if node.op != "call_function":
            return False
        
        ####################################################################
        # Case 1: mul
        # `mul` is reassociable with the reduction node if either of its
        # multiplicand or multiplier is broadcasted along the reduction 
        # dimension
        if node.target == torch.ops.aten.mul:
            shape_product = get_shape(node)
            # They are reassociable when they are at the same shape
            if shape_product != target_shape:
                return False
            
            # Get the shape of multiplicand and multiplier
            shape_args = self.get_canonical_broadcast_shapes(node, node.args)
            for shape in shape_args:
                match = True
                for dim in reduction_dim:
                    if shape[dim] != 1:
                        match = False
                        break
                # As long as there is one match, return True
                if match:
                    return True
            return False
        ####################################################################
        # Case 2: neg is always reassociable
        elif node.target == torch.ops.aten.neg:
            return True
        ####################################################################
        # Case 3: one_hot is a special case when its dimension matchs the 
        #         reduction dim
        elif node.target == torch.ops.aten.one_hot:
            shape_onehot = get_shape(node)
            if len(reduction_dim) == 1 and len(shape_onehot) - 1 == reduction_dim[0]:
                return True
            return False
        else:
            return False


def pass_constant_propagation(module, graph):
    # First eliminate the reductions
    res = ReductionElimination()
    res(module)

    # Then try to fold any constant expressions + reassociation
    reas = Reassociation()
    reas(module)

    # permutation propagation
    pf = PermutationFolding()
    pf(module)