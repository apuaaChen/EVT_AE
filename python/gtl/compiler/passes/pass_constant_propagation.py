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
from typing import Optional, Union
from torch.fx.graph_module import GraphModule
from torch.fx import Node, Graph
from torch.fx.passes.tools_common import legalize_graph
from torch.fx.passes.infra.pass_base import PassBase, PassResult
from enum import Enum
import torch
from torch._functorch.partitioners import _extract_graph_with_inputs_outputs
from gtl.compiler.passes.pass_print_graph import pass_print_graph
from torch.fx.subgraph_rewriter import replace_pattern
from torch.utils._python_dispatch import _pop_mode_temporarily, _len_torch_dispatch_stack
from contextlib import nullcontext
from gtl.compiler.nodes import inject_get_attr, inject_squeeze
from gtl.compiler.passes.pass_fake_shape_infer import pass_fake_shape_infer

# # Define the semilattice
class SL(Enum):
    UNDEF = 0    # Top: undefined type
    NAC = 1      # Bottom: not a constant

class ConstantPropagation(PassBase):
    def __init__(self) -> None:
        super().__init__()
    
    def binary_transfer(self, abv1, abv2):
        if abv1 is None:
            if abv2 in [SL.UNDEF, SL.NAC]:
                return abv2
            else:
                assert not isinstance(abv2, list)
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
    
    def ensures(self, graph_module: GraphModule) -> None:
        return super().ensures(graph_module)


# The key idea is to eliminate the reduction nodes
# The reduction nodes usually breaks the flow of fusion as they are only fusible
# for a perfectly-nested-loop compiler under the atomic reduction
# Seeking ways of representing the reduction with other forms of computation 
# would unlock new fusion opportunities

# Helper function for associativity
def get_shape(node: Union[Node, float, int]) -> list:
    if isinstance(node, Node):
        return list(node.meta["tensor_meta"].shape)
    else:
        return [1]


def get_canonical_broadcast_shapes(target: Node, args: 'list[Node]'):
    shape_target = get_shape(target)
    shape_args = [get_shape(arg) for arg in args]
    shape_args = [
        [1] * (len(shape_target) - len(shape_arg)) + shape_arg 
        for shape_arg in shape_args]
    
    return shape_args

def is_reassociable_op(node: Union[Node, float, int], target_node: Node):
    """
    Return true is node is reassociable with the current reduction node
    Let's keep it simple for now
    """
    if not isinstance(node, Node):
        return False

    if target_node.target == torch.ops.aten.sum:
        target_shape = get_shape(target_node.args[0])
        # Note: reduction dim could be a list of indices
        reduction_dim = target_node.args[1]
        if not isinstance(reduction_dim, list):
            reduction_dim = [reduction_dim, ]

    if node.op == "call_function":
        if target_node.target == torch.ops.aten.sum:
            ####################################################################
            # Case 1: mul
            # `mul` is reassociable with the reduction node if either of its
            # multiplicand or multiplier is broadcasted along the reduction 
            # dimension
            if node.target == torch.ops.aten.mul:
                if target_node.target == torch.ops.aten.sum:
                    shape_product = get_shape(node)
                    if shape_product == target_shape:
                        shape_args = get_canonical_broadcast_shapes(
                            node, node.args
                        )
                        for shape in shape_args:
                            match = True
                            for dim in reduction_dim:
                                if shape[dim] != 1:
                                    match = False
                                    break
                            if match:
                                return True
                        return False
                    else:
                        # TODO: there might be another way
                        return False
            
            elif node.target == torch.ops.aten.neg:
                return True
            elif node.target == torch.ops.aten.one_hot:
                shape_onehot = get_shape(node)
                if len(reduction_dim) == 1 and len(shape_onehot) - 1 == reduction_dim[0]:
                    return True
                else:
                    return False
            else:
                return False
        else:
            if len(node.users) <= 1 and node.target == target_node.target:
                return True
            return False
    else:
        return False


class Reassociation(PassBase):
    def __init__(self) -> None:
        super().__init__()
        self.rank_map = {}
        self.ops = []
    
    def replace_neg_with_mul(self, graph_module):
        def pattern(x):
            return torch.ops.aten.neg(x)
        
        def replacement(x):
            return torch.ops.aten.mul(x, -1.)   
        replace_pattern(graph_module, pattern, replacement)

    def linearize_expr(self, node: Node):
        lhs = node.args[0]
        rhs = node.args[1]

        assert is_reassociable_op(lhs, node)
        assert is_reassociable_op(rhs, node)

        node.args = (lhs, rhs.args[0])
        rhs.args = (lhs, rhs.args[1])
        node.args = (rhs, node.args[1])

        if self.is_reassociable_op(node.args[1], node.target):
            self.linearize_expr(node)
    
    def get_rank(self, node: Union[Node, float, int]):
        if isinstance(node, Node):
            return self.rank_map[node]
        else:
            return 0

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
        if (ops[0][1].target == torch.ops.aten.one_hot and 
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
        # Constant folding
        cp = ConstantPropagation()
        cp(self.module)
        self.module.graph.eliminate_dead_code()
        

    
    def is_associative(self, node: Node):
        if node.target in [torch.ops.aten.mul, torch.ops.aten.add]:
            return True
        return False
    
    def is_reassociable_op(self, node: Union[Node, float, int], target):
        if not isinstance(node, Node):
            return False
        if len(node.users) <= 1 and node.target == target:
            return node
        return False
    
    def reassociate_bb(self, graph):
        """
        Inspect all of the nodes in this graph, reasociating them as we go
        """
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

    
    def call(self, graph_module: GraphModule) -> PassResult:
        ## Preprocessing
        self.replace_neg_with_mul(graph_module)
        # Try to eliminate the sum node
        self.module = graph_module
        graph = graph_module.graph

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

        # Sort the nodes
        self.reassociate_bb(graph)
        # rank_map_ascending = dict(sorted(self.rank_map.items(), key=lambda item: item[1]))

    def requires(self, graph_module: GraphModule) -> None:
        pass

    def ensures(self, graph_module: GraphModule) -> None:
        pass


class ReductionElimination(PassBase):
    def __init__(self) -> None:
        super().__init__()
        self.reduction_node = None

    def backward_dfs(self, node: Node):
        if node in self.visited:
            return
        self.visited.append(node)
        for input in node.all_input_nodes:
            if is_reassociable_op(input, self.reduction_node):
                self.backward_dfs(input)
            else:
                self.input_nodes.append(input)

    def extract_subgraph(self, node: Node):
        """
        Extract the subgraph producing the current node
        """
        # self.extracted_nodes = []
        self.input_nodes = []
        self.visited = []
        self.reduction_node: Node = node
        self.backward_dfs(node)
        # Extract subgraph


    def call(self, graph_module: GraphModule) -> PassResult:
        graph: Graph = graph_module.graph

        sum_nodes = []

        for node in graph.nodes:
            if node.target in [torch.ops.aten.sum,]:
                sum_nodes.append(node)
        
        for node in sum_nodes:
            # Extract the subgraph producing the node
            self.extract_subgraph(node)
            output_nodes = [node]
            subgraph = _extract_graph_with_inputs_outputs(
                graph, self.input_nodes, output_nodes
            )
            submodule = GraphModule(graph_module, subgraph)

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
                continue

            # Copy the whole subgraph into the original graph module
            # val_map: Dict[Node, Node] mapping from nodes in g to nodes in 
            # self
            val_map = {}
            # match the input nodes:
            subgraph_placeholders = {n.name: n for n in submodule.graph.nodes if n.op == "placeholder"}
            subgraph_attrs = [n.target for n in submodule.graph.nodes if n.op == "get_attr"]

            for input in self.input_nodes:
                val_map[subgraph_placeholders[input.name]] = input
            
            # val_map[subgraph_output] = node
            with graph.inserting_before(node):
                copied_returning_nodes = graph.graph_copy(submodule.graph, val_map)
            
            node.replace_all_uses_with(copied_returning_nodes[0])

            # Copy the attributes
            for attr in subgraph_attrs:
                if not hasattr(graph_module, attr):
                    setattr(graph_module, attr, getattr(submodule, attr))
            
            graph.erase_node(node)
            
        legalize_graph(graph_module)
        graph.eliminate_dead_code()
        graph_module.recompile()


    def requires(self, graph_module: GraphModule) -> None:
        pass

    def ensures(self, graph_module: GraphModule) -> None:
        pass_fake_shape_infer(graph_module, graph_module.graph)


def pass_constant_propagation(module, graph):
    # tcp = TensorConstantProp()
    res = ReductionElimination()

    # module, modified = 
    res(module)