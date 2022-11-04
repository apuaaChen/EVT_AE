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
import pycutlass
from pycutlass import *
import cutlass
import torch.fx as fx
from treelib import Tree
from nodes import *
from functools import reduce
from passes.iterator import *
import operator
import philox


torch_2_cutlass_type = {
    torch.float32: cutlass.float32,
    torch.float16: cutlass.float16,
    torch.int64: cutlass.int64,
    torch.int32: cutlass.int32
}

def dfs(node, target):
    if node == target:
        return True
    else:
        if len(node.all_input_nodes) == 0:
            return False
        else:
            for input in node.all_input_nodes:
                if dfs(input, target): return True
            return False

################################################################################
# Epilogue nodes
################################################################################
class TensorOutputNodeDAG(TensorOutputNode):
    def __init__(self, element_accumulator, node) -> None:
        self.id = "output_" + node.name
        self.tag = self.id
        self.tag = "TensorOutput:" + self.tag
        self.type = "tensor"
        self.element_accumulator = element_accumulator

        # get the permutation of output
        valid_iters = {
            'b': None, 
            'm': None,
            'n': None
        }
        # get valid iterators
        dim_idx = 0
        for dim in node.meta['tensor'].dims:
            if len(dim) > 1:
                raise NotImplementedError()
            if dim[0].mod > 1:
                if valid_iters[dim[0].iter_var.name] is None:
                    valid_iters[dim[0].iter_var.name] = dim_idx
                    dim_idx += 1
        permute = []
        for key in valid_iters.keys():
            if valid_iters[key] is not None:
                permute.append(valid_iters[key])
        self.permute = permute
    
    def get_argument(self, visitor_args, kwargs):
        if len(self.permute) == 3:
            try:
                batch_size = kwargs["batch_size"]
            except:
                raise ValueError("batch size should be included in keyword args")

            # case 1: [0, 1, 2] -> [B, M, N]
            if self.permute == [0, 1, 2]:
                ldt = kwargs["problem_size"][1]
                batch_stride = kwargs["problem_size"][0] * kwargs["problem_size"][1]
            # case 2: [1, 0, 2] -> [M, B, N]
            elif self.permute == [1, 0, 2]:
                ldt = kwargs["problem_size"][1] * batch_size
                batch_stride = kwargs["problem_size"][1]
            else:
                raise NotImplementedError("Unsupported output permutation")
        else:
            ldt = kwargs["problem_size"][1]
            batch_stride = kwargs["problem_size"][0] * kwargs["problem_size"][1]

        self.argument = self.epilogue_node.argument_type(kwargs[self.id + "_ptr"], ldt, *visitor_args, batch_stride)

        

class RowReductionNodeDAG(RowReductionNode):
    def __init__(self, element_accumulator, element_reduction, element_reduction_accumulator, id, factor) -> None:
        super().__init__(element_accumulator, element_reduction, element_reduction_accumulator, id, factor)

class ColumnReductionNodeDAG(ColumnReductionNode):
    def __init__(self, element_accumulator, element_reduction, element_reduction_accumulator, id, factor) -> None:
        super().__init__(element_accumulator, element_reduction, element_reduction_accumulator, id, factor)

# operators
operators = {
    torch.ops.aten.add: "Add",
    torch.ops.aten.div: "Div",
    torch.ops.aten.sub: "Sub",
    torch.ops.aten.mul: "Mult",
    torch.ops.aten.neg: "Mult",
    torch.ops.aten.ne: "Ne",
    torch.ops.aten.gelu: "GeLU"
}


class UnaryNodeDAG(UnaryNode):
    cnt = 0
    def __init__(self, element_accumulator, element_compute, 
        elements_per_access, node, args, element_ptr=None) -> None:
        #
        if isinstance(node, BinOpNodeDAG):
            self.op = node.op
        elif isinstance(node, fx.Node):
            self.op = operators[node.target]
        else:
            raise TypeError
        self.tag = "Unary" + self.op + str(UnaryNode.cnt)
        self.id = self.op + str(UnaryNodeDAG.cnt)
        self.args = args
        self.element_ptr = element_ptr
        UnaryNodeDAG.cnt += 2

        self.type = "tensor"

        self.epilogue_op = getattr(pycutlass, self.op)(element_compute)

        # data types
        self.element_accumulator = element_accumulator
        self.element_compute = element_compute
        self.elements_per_access = elements_per_access
        

class OneHotNodeDAG(OneHotNode):
    def __init__(self, element_accumulator, element_compute, 
        elements_per_access, node) -> None:
        #
        self.op = "one_hot"
        self.tag = "OneHot" + str(UnaryNode.cnt)
        self.id = self.op + str(UnaryNode.cnt)
        UnaryNode.cnt += 1

        self.type = "tensor"

        # data types
        self.element_accumulator = element_accumulator
        self.element_compute = element_compute
        self.elements_per_access = elements_per_access
    
    def get_argument(self, visitor_args, kwargs):
        return super().get_argument(visitor_args, kwargs)

class BinOpNodeDAG(BinOpNode):
    cnt = 1
    def __init__(
        self, element_accumulator, element_compute, 
        elements_per_access, node) -> None:
        #
        self.op = operators[node.target]
        self.tag = "Binary" + self.op + str(BinOpNodeDAG.cnt)
        self.id = self.op + str(BinOpNodeDAG.cnt)
        self.args = None
        BinOpNodeDAG.cnt += 2

        self.type = "tensor"

        self.epilogue_op = getattr(pycutlass, "Vector"+self.op)(element_compute)

        # data types
        self.element_accumulator = element_accumulator
        self.element_compute = element_compute
        self.elements_per_access = elements_per_access
    
    def get_argument(self, visitor_args, kwargs):
        return super().get_argument(visitor_args, kwargs)

class DropoutForwardNodeDAG(DropoutForwardNode):
    cnt = 0
    def __init__(self, element_accumulator, element_compute, elements_per_access, node, anchor) -> None:
        self.op = "dropout_forward"
        self.tag = "DropoutForward" + str(DropoutForwardNodeDAG.cnt)
        self.type = "tensor"
        self.id = self.op + str(DropoutForwardNodeDAG.cnt)

        self.p = 1. - node.args[1]
        if anchor.target in [torch.ops.aten.mm, torch.ops.aten.bmm]:
            self.iterator_type = "cutlass::epilogue::threadblock::PredicatedTileIterator"
        elif anchor.target in [torch.ops.aten._softmax]:
            self.iterator_type = "cutlass::softmax::threadblock::RowTileIterator"
        
        # get mask ptr
        for user in node.users.keys():
            if user.target == operator.getitem and user.meta["tensor_meta"].dtype == torch.bool:
                self.mask_ptr = "output_" + user.name + "_ptr"

        # data types
        self.element_accumulator = element_accumulator
        self.element_compute = element_compute
        self.elements_per_access = elements_per_access
        DropoutForwardNodeDAG.cnt += 1
    
    def get_argument(self, visitor_args, kwargs):
        seed, offset = philox.philox_state(1024)
        self.argument = self.epilogue_node.argument_type(
            self.p, seed, offset, kwargs[self.mask_ptr],
            kwargs["problem_size"][1], 
            kwargs["problem_size"][0] * kwargs["problem_size"][1],
            *visitor_args
        )

class TensorInputNodeDAG(TensorInputNode):
    def __init__(self, element_accumulator, node, valid_iters=None) -> None:
        self.id = node.name
        self.tag = "TensorInput:" + self.id
        self.type = "tensor"
        self.element_accumulator = element_accumulator
        self.valid_iters = valid_iters
        self.element_input = torch_2_cutlass_type[node.meta['tensor_meta'].dtype]
    
    def get_argument(self, visitor_args, kwargs):
        try:
            factor = self.valid_iters['b'].factor
            modulo = self.valid_iters['b'].mod
            args = [
                kwargs[self.id + "_ptr"], 
                kwargs["problem_size"][1], 
                kwargs["problem_size"][0] * kwargs["problem_size"][1],
                factor, modulo]
            self.argument = self.epilogue_node.argument_type(*args)
        except:
            return super().get_argument(visitor_args, kwargs)

class RowBroadcastNodeDAG(RowBroadcastNode):
    def __init__(self, element_accumulator, element_fragment, node, element_input=None, valid_iters=None) -> None:
        self.id = node.name
        self.tag = "RowBroadcast:" + self.id
        self.type = "tensor"
        self.element_accumulator = element_accumulator
        self.element_fragment = element_fragment
        self.element_input = element_input
        self.valid_iters = valid_iters
    
    def get_argument(self, visitor_args, kwargs):
        try:
            factor = self.valid_iters['b'].factor
            modulo = self.valid_iters['b'].mod
            args = [
                kwargs[self.id + "_ptr"], kwargs["problem_size"][1],
                factor, modulo
            ]
            self.argument = self.epilogue_node.argument_type(*args)
        except:
            return super().get_argument(visitor_args, kwargs)

class ColumnBroadcastNodeDAG(ColumnBroadcastNode):
    def __init__(self, element_accumulator, element_fragment, node, element_input=None, valid_iters=None) -> None:
        self.id = node.name
        self.tag = "ColumnBroadcast:" + self.id
        self.type = "tensor"
        self.element_accumulator = element_accumulator
        self.element_fragment = element_fragment
        self.element_input = element_input
        self.valid_iters = valid_iters
    
    def get_argument(self, visitor_args, kwargs):
        try:
            factor = self.valid_iters['b'].factor
            modulo = self.valid_iters['b'].mod
            args = [
                kwargs[self.id + "_ptr"], kwargs["problem_size"][0],
                factor, modulo
            ]
            self.argument = self.epilogue_node.argument_type(*args)
        except:
            return super().get_argument(visitor_args, kwargs)

class ScalarInputNodeDAG(ScalarInputNode):
    def __init__(self, node) -> None:
        self.id = node.name
        self.tag = "Scalar:" + self.id
        self.type = "scalar"

        self.element_ptr = torch_2_cutlass_type[node.meta["tensor_meta"].dtype] 


class AccumulatorNodeDAG(AccumulatorNode):
    def __init__(self, element_accumulator, elements_per_access, node) -> None:
        self.id = node.name
        self.tag = "Accum:" + self.id
        self.type = "tensor"

        self.element_accumulator = element_accumulator
        self.elements_per_access = elements_per_access
    
    def get_argument(self, visitor_args, kwargs):
        return super().get_argument(visitor_args, kwargs)

################################################################################
# Epilogue parser function for pytorch DAG
################################################################################
class EpilogueASTDAG:
    """
    Helper class to create the Epilogue AST from a pytorch fx.Graph
    """
    def __init__(self, anchor, tile_description, element_accumulator,
        elements_per_access, element_compute, element_output) -> None:
        #

        self.tile_description = tile_description
        self.element_accumulator = element_accumulator
        self.elements_per_access = elements_per_access
        self.element_compute = element_compute
        self.element_output = element_output
        self.anchor = anchor
        
        self.shape = list(anchor.meta['tensor_meta'].shape)
        self.get_root()

        if self.anchor.target == torch.ops.aten.mm:
            print(self.root)

        self.outputs = []
        if anchor.target in [torch.ops.aten.bmm, torch.ops.aten.mm, torch.ops.aten._softmax]:
            self.ast_top_down_parsing_bmm(self.root, parse_output=True)

        self.stack = []
        self.reduction_source = {}

        self.input_args = {}

        # this list tracks the list of input nodes
        self.args = []

        # parse epilogue tree from DAG
        self.epilogue_tree = Tree()

        self.visit(self.root)

        self.returns = [node.name for node in self.outputs]

    #
    # Tree optimization pass
    #

    # pass 1: lower Binary to Unary
    def pass_binary_2_unary(self, tree, nid):
        node = tree.get_node(nid)
        if isinstance(node.data, BinOpNodeDAG):
            lhs_node = tree.get_node(node.successors(tree.identifier)[0])
            left_type = lhs_node.data.type
            rhs_node = tree.get_node(node.successors(tree.identifier)[1])
            right_type = rhs_node.data.type

            if left_type == "scalar" and right_type == "tensor":
                node.data = UnaryNodeDAG(
                    self.element_accumulator, self.element_compute,
                    self.elements_per_access,
                    node.data, [lhs_node.data.id,], lhs_node.data.element_ptr)
                node.tag = node.data.tag
                tree.remove_node(lhs_node.data.id)
                self.pass_binary_2_unary(tree, rhs_node.data.id)
            
            elif left_type == "tensor" and right_type == "scalar":
                node.data = UnaryNodeDAG(
                    self.element_accumulator, self.element_compute,
                    self.elements_per_access,
                    node.data, [rhs_node.data.id,], rhs_node.data.element_ptr)
                node.tag = node.data.tag
                tree.remove_node(rhs_node.data.id)
                self.pass_binary_2_unary(tree, lhs_node.data.id)
            
            else:
                self.pass_binary_2_unary(tree, lhs_node.data.id)
                self.pass_binary_2_unary(tree, rhs_node.data.id)
        else:
            for child in node.successors(tree.identifier):
                self.pass_binary_2_unary(tree, child)
    
    # pass 2: inject reduction nodes
    def pass_inject_reduction(self, tree, nid):
        node = tree.get_node(nid)
        if isinstance(node.data, TensorOutputNodeDAG):
            if node.data.id in self.reduction_source.keys():
                direction = self.reduction_source[node.data.id][0]
                target = self.reduction_source[node.data.id][-1]
                if direction == 'row':
                    reduction_node = RowReductionNodeDAG(
                        self.element_accumulator, self.element_output,
                        self.element_accumulator, target, self.tile_description.threadblock_shape[1])
                elif direction == "column":
                    reduction_node = ColumnReductionNodeDAG(
                        self.element_accumulator, self.element_output,
                        self.element_accumulator, target, self.tile_description.threadblock_shape[0])
                else:
                    raise ValueError(direction)
                child_nid = node.successors(tree.identifier)[0]
                # if this output node is injected only for reduction
                if node.data.id not in self.returns:
                    # get reduction config from disc
                    node.data = reduction_node
                    node.tag = reduction_node.tag
                    self.pass_inject_reduction(tree, child_nid)
                # if this output node is also a tensor output, inject reduction as its children
                else:
                    # get child node
                    tree.create_node(reduction_node.tag, reduction_node.id, data=reduction_node, parent=node.data.id)
                    tree.move_node(child_nid, reduction_node.id)
                    child = tree.get_node(child_nid)
                    for grand_child in child.successors(tree.identifier):
                        self.pass_inject_reduction(tree, grand_child)
            else:
                for child in node.successors(tree.identifier):
                    self.pass_inject_reduction(tree, child)
        else:
            for child in node.successors(tree.identifier):
                self.pass_inject_reduction(tree, child)

    def pass_inject_epilogue_op(self, tree, nid):
        node = tree.get_node(nid)
        visitors = []
        for child in node.successors(tree.identifier):
            visitors.append(self.pass_inject_epilogue_op(tree, child))
        
        node.data.get_epilogue_node(visitors)
        return node.data.epilogue_node
    
    def get_candidate_root_bmm(self, node):
        """
        This function performs DFS search for all the candidate root nodes for 
        the epilogue visitor tree
        """
        if node.op == "call_function":
            # get all user nodes
            user_nodes = list(node.users.keys())
            for usr in user_nodes:
                try:
                    node.meta['tensor'].get_node_tensor_bottom_up(usr)
                except:
                    if node.target == torch.ops.aten.permute:
                        if "epilogue_permute" not in node.meta.keys():
                            self.root_candidates[node.args[0]] = []
                        else:
                            self.root_candidates[node] = []
                    self.root_candidates[node] = []
                    return
                self.get_candidate_root_bmm(usr)

    
    def ast_top_down_parsing_bmm(self, node, parse_output=False):
        """
        This function parses the ast from the root in the top-down manner
        """
        # step 1: check if the node is a dependency of anchor
        if node != self.anchor and dfs(self.anchor, node):
            return [node,]
        if parse_output:
            if len(node.users) > 1 or node == self.root:
                # case for multiple output nodes. All its users are getitem
                if list(node.users.keys())[0].target == operator.getitem:
                    for user in node.users:
                        assert user.target == operator.getitem
                        if "in_ast" in user.meta.keys():
                            if len(user.users) > 1:
                                self.outputs.append(user)
                        else:
                            self.outputs.append(user)
                elif reduce(lambda x, y: x * y, self.shape) == reduce(lambda x, y: x * y, node.meta["tensor_meta"].shape):
                    if node not in self.outputs:
                        self.outputs.append(node)
        if node.op == "call_function":
            try:
                node.meta["tensor"].get_node_tensor_top_down(node)
                if node.target in [
                    torch.ops.aten.view, torch.ops.aten._unsafe_view, 
                    torch.ops.aten.unsqueeze, torch.ops.aten._to_copy, torch.ops.aten.clone]:
                    #
                    return self.ast_top_down_parsing_bmm(node.args[0], parse_output)
                elif node.target in [operator.getitem]:
                    if parse_output:
                        node.meta["in_ast"] = True
                    return self.ast_top_down_parsing_bmm(node.args[0], parse_output)
                elif node.target in [torch.ops.aten.add, torch.ops.aten.sub,
                    torch.ops.aten.mul, torch.ops.aten.div]:
                    #
                    lhs_nodes = self.ast_top_down_parsing_bmm(node.args[0], parse_output)
                    rhs_nodes = self.ast_top_down_parsing_bmm(node.args[1], parse_output)
                    return lhs_nodes + rhs_nodes + [node, ]
                elif node.target in [torch.ops.aten.neg, torch.ops.aten.sum, torch.ops.aten.native_dropout, torch.ops.aten.permute, torch.ops.aten.gelu]:
                    return self.ast_top_down_parsing_bmm(node.args[0], parse_output) + [node,]
                elif node.target in [torch.ops.aten.mm, torch.ops.aten._softmax, torch.ops.aten.one_hot, torch.ops.aten.ne, torch.ops.aten.bmm]:
                    return [node,]
                else:
                    return []
            except:
                return [node,]
            
        else:
            return []

    
    def get_fusion_cost(self, node):
        if node.op == "call_function":
            if node.target in [torch.ops.aten.add, torch.ops.aten.sub, 
                torch.ops.aten.mul, torch.ops.aten.div, torch.ops.aten.one_hot, torch.ops.aten.ne,
                torch.ops.aten.sum]:
                shape = get_shape(node)
                cost = 1
                for dim in shape:
                    cost *= dim
                return cost
            else:
                return 0
    
    def get_root(self):
        """
        The root is obtained as follows:
        1. start from the anchor node, we find all the possible root nodes
        2. for reduction nodes, if it shares the same parent with another root
           then it is removed
        3. from each root, we perform top-down parsing to identify all the nodes
           to be fused
        4. the reduced cost with each root is approximated with the size of the 
           function
        5. the root with the highest reduced cost is selected
        """
        self.root_candidates = {}

        # step 1: assign iterators to anchor
        if self.anchor.target == torch.ops.aten.bmm:
            iter_vars = [
                IterVar("b", extend=self.anchor.meta["tensor_meta"].shape[0]),
                IterVar("m", extend=self.anchor.meta["tensor_meta"].shape[1]),
                IterVar("n", extend=self.anchor.meta["tensor_meta"].shape[2])]
            self.anchor.meta["tensor"] = Tensor(iter_vars)
            self.get_candidate_root_bmm(self.anchor)
        elif self.anchor.target == torch.ops.aten.mm:
            iter_vars = [
                IterVar("m", extend=self.anchor.meta["tensor_meta"].shape[0]),
                IterVar("n", extend=self.anchor.meta["tensor_meta"].shape[1])
            ]
            self.anchor.meta["tensor"] = Tensor(iter_vars)
            self.get_candidate_root_bmm(self.anchor)
        elif self.anchor.target == torch.ops.aten._softmax:
            # The softmax can take an arbitrary dim 
            shape = self.anchor.meta["tensor_meta"].shape
            reduction_dim = self.anchor.args[1]
            if reduction_dim < 0:
                reduction_dim = len(shape) + reduction_dim
            independent_size = 1
            for idx, dim in enumerate(shape):
                if idx == reduction_dim: continue
                independent_size *= dim
            shape = (independent_size, shape[reduction_dim])
            iter_vars = [
                IterVar("m", extend=shape[0]),
                IterVar("n", extend=shape[1])
            ]
            self.anchor.meta["tensor"] = Tensor(iter_vars).view(self.anchor.meta["tensor_meta"].shape)
            self.get_candidate_root_bmm(self.anchor)
        else:
            raise NotImplementedError()

        # filter root candiates
        for root in self.root_candidates.keys():
            if root.target == torch.ops.aten.sum:
                for others in self.root_candidates.keys():
                    if others == root: continue
                    if root.args[0] in others.args:
                        self.root_candidates[others].append(root)
                        self.root_candidates[root] = None
        
        if self.anchor.target == torch.ops.aten.mm:
            print(self.root_candidates)

        if self.anchor.target in [torch.ops.aten.bmm, torch.ops.aten.mm, torch.ops.aten._softmax]:
            root_to_remove = []
            for root in self.root_candidates.keys():
                if self.root_candidates[root] is not None:
                    node_list = self.ast_top_down_parsing_bmm(root)
                    self.root_candidates[root] += node_list
                    if self.anchor not in self.root_candidates[root]:
                        self.root_candidates[root] = None
                        root_to_remove.append(root)
                else:
                    root_to_remove.append(root)
            for root in root_to_remove:
                self.root_candidates.pop(root)
        else:
            raise NotImplementedError
        
        # get the root with lowest cost
        max_cost = -1
        self.root = None
        for root in self.root_candidates.keys():
            cost = 0
            for node in self.root_candidates[root]:
                cost += self.get_fusion_cost(node)
            
            if cost > max_cost:
                max_cost = cost
                self.root = root
        # TODO: handle the case that cost of two graphs is the same

    def visit(self, node):
        is_input = False
        # TODO: the tensor input can be output of another node
        if node != self.anchor and dfs(self.anchor, node):
            # the node will be treated as a tensor input node
            is_input = True
        if self.anchor.target in [torch.ops.aten.bmm, torch.ops.aten.mm, torch.ops.aten._softmax]:
            if 'tensor' not in node.meta.keys():
                print(node)
                raise RuntimeError("The node to fuse doesn't have tensor attribute")
        if node in self.outputs:
            assert not is_input
            # visit the output node
            name_node = TensorOutputNodeDAG(self.element_accumulator, node)
            if len(self.stack) == 0:
                self.epilogue_tree.create_node(name_node.tag, name_node.id, data=name_node)
            else:
                self.epilogue_tree.create_node(name_node.tag, name_node.id, parent=self.stack[-1], data=name_node)
            self.stack.append(name_node.id)
        if node.op == "call_function" and not is_input:
            if node.target in [torch.ops.aten.view, torch.ops.aten._unsafe_view, torch.ops.aten.unsqueeze, torch.ops.aten._to_copy, operator.getitem, torch.ops.aten.clone, torch.ops.aten.permute]:
                self.visit(node.args[0])
            elif node.target in [torch.ops.aten.add, torch.ops.aten.mul, torch.ops.aten.sub, torch.ops.aten.div]:
                binop = BinOpNodeDAG(
                    self.element_accumulator, self.element_compute,
                    self.elements_per_access, node)
                self.epilogue_tree.create_node(
                    binop.tag, binop.id, data=binop,
                    parent=self.stack[-1])
                self.stack.append(binop.id)
                self.visit(node.args[0])
                self.visit(node.args[1])
                self.stack.pop()
            elif node.target in [torch.ops.aten.neg]:
                unaryop = UnaryNodeDAG(
                    self.element_accumulator, self.element_compute,
                    self.elements_per_access, node, [-1.]
                )
                self.epilogue_tree.create_node(
                    unaryop.tag, unaryop.id, data=unaryop,
                    parent=self.stack[-1]
                )
                self.stack.append(unaryop.id)
                self.visit(node.args[0])
                self.stack.pop()
            elif node.target in [torch.ops.aten.gelu]:
                unaryop = UnaryNodeDAG(
                    self.element_accumulator, self.element_compute,
                    self.elements_per_access, node, args=[]
                )
                self.epilogue_tree.create_node(
                    unaryop.tag, unaryop.id, data=unaryop,
                    parent=self.stack[-1]
                )
                self.stack.append(unaryop.id)
                self.visit(node.args[0])
                self.stack.pop()
            elif node.target in [torch.ops.aten.ne]:
                unaryop = UnaryNodeDAG(
                    self.element_accumulator, self.element_compute,
                    self.elements_per_access, node, [node.args[1]]
                )
                self.epilogue_tree.create_node(
                    unaryop.tag, unaryop.id, data=unaryop,
                    parent=self.stack[-1]
                )
                self.stack.append(unaryop.id)
                self.visit(node.args[0])
                self.stack.pop()
            elif node.target in [torch.ops.aten.one_hot]:
                one_hot_op = OneHotNodeDAG(
                    self.element_accumulator, self.element_compute,
                    self.elements_per_access, node
                )
                self.epilogue_tree.create_node(
                    one_hot_op.tag, one_hot_op.id, data=one_hot_op,
                    parent=self.stack[-1]
                )
                self.stack.append(one_hot_op.id)
                self.visit(node.args[0])
                self.stack.pop()
            elif node.target == self.anchor.target:
                name_node = AccumulatorNodeDAG(
                    self.element_accumulator, self.elements_per_access, node
                )
                self.epilogue_tree.create_node(
                    name_node.tag, name_node.id, 
                    data=name_node, parent=self.stack[-1])
            elif node.target in [torch.ops.aten.native_dropout]:
                dropout_node = DropoutForwardNodeDAG(
                    self.element_accumulator, self.element_compute, 
                    self.elements_per_access, node, self.anchor)
                self.epilogue_tree.create_node(
                    dropout_node.tag, dropout_node.id, data=dropout_node,
                    parent=self.stack[-1]
                )
                self.stack.append(dropout_node.id)
                self.visit(node.args[0])
                self.stack.pop()
            else:
                print("got not implemented node")
                print(node)
                print(node.target)
                raise NotImplementedError

        elif node.op in ["placeholder", "get_attr"] or is_input:
            if self.anchor.target in [torch.ops.aten.bmm, torch.ops.aten.mm, torch.ops.aten._softmax]:
                source_type = torch_2_cutlass_type[node.meta['tensor_meta'].dtype]

                node_tensor = node.meta['tensor']

                valid_iters = {
                    'b': None, 
                    'm': None,
                    'n': None
                }
                # get valid iterators
                for dim in node_tensor.dims:
                    if len(dim) > 1:
                        raise NotImplementedError()
                    if dim[0].mod > 1:
                        valid_iters[dim[0].iter_var.name] = dim[0]
                
                # case 1: scalar
                if valid_iters['b'] is None and valid_iters['m'] is None and valid_iters['n'] is None:
                    name_node = ScalarInputNodeDAG(node)
                # case 2: row broadcast
                elif valid_iters['m'] is None and valid_iters['n'] is not None:
                    name_node = RowBroadcastNodeDAG(
                        self.element_accumulator, self.element_compute, node, source_type, valid_iters
                    )
                # case 3: column broadcast
                elif valid_iters['n'] is None and valid_iters['m'] is not None:
                    name_node = ColumnBroadcastNodeDAG(
                        self.element_accumulator, source_type, node, source_type, valid_iters
                    )
                elif valid_iters['m'] is not None and valid_iters['n'] is not None:
                    name_node = TensorInputNodeDAG(self.element_accumulator, node, valid_iters)
                else:
                    raise NotImplementedError()
                
            if isinstance(name_node, TensorInputNodeDAG):
                self.input_args[name_node.id] = ["tensor",]
            elif isinstance(name_node, RowBroadcastNodeDAG):
                self.input_args[name_node.id] = ["row",]
            elif isinstance(name_node, ColumnBroadcastNodeDAG):
                self.input_args[name_node.id] = ["column",]
            elif isinstance(name_node, ScalarInputNodeDAG):
                self.input_args[name_node.id] = ["scalar",]
            else:
                raise NotImplementedError()
            try:
                self.epilogue_tree.create_node(
                    name_node.tag, name_node.id, 
                    data=name_node, parent=self.stack[-1])
            except:
                # TODO: the "_2" is not very general
                self.epilogue_tree.create_node(
                    name_node.tag, name_node.id + "_2", 
                    data=name_node, parent=self.stack[-1])
            self.args.append(node)

        if node in self.outputs:
            self.stack.pop()


    def get_arguments(self, tree, nid, kwargs):
        node = tree.get_node(nid)
        visitor_args = []
        for child in node.successors(tree.identifier):
            visitor_args.append(self.get_arguments(tree, child, kwargs))
        node.data.get_argument(visitor_args, kwargs)
        return node.data.argument

class EpilogueVisitTreeDAG(EpilogueVisitTree):
    # Epilogue visitor tree from DAG
    def __init__(
        self, elementwise_functor, tile_description, element_accumulator, 
        elements_per_access, element_compute, element_output) -> None:
        #
        super().__init__(
            elementwise_functor, tile_description, element_accumulator, 
            elements_per_access, element_compute, element_output)
        
    def initialize(self, node: fx.Node):
        # TODO: some EpilogueAST structure
        function = EpilogueASTDAG(
            node, self.tile_description, self.element_accumulator,
            self.elements_per_access, self.element_compute, self.element_output
        )

        tree = function.epilogue_tree
        self.tree = tree
        self.args = function.args
        self.root = function.root
        self.outputs = function.outputs
        
        function.pass_binary_2_unary(self.tree, self.tree.root)
        function.pass_inject_reduction(self.tree, self.tree.root)
        function.pass_inject_epilogue_op(self.tree, self.tree.root)

        visitor = self.tree.get_node(self.tree.root).data.epilogue_node
        self.visitor = visitor

        if function.anchor.target == torch.ops.aten.mm:
            self.tree.show()

        # create argument data type
        class _Argument(ctypes.Structure):
            _fields_ = [
                ("visitor_arg", visitor.argument_type)
            ]
            def __init__(self, **kwargs) -> None:
                # process input args
                _kwargs = {}
                for input_key in function.input_args.keys():
                    if input_key == "accum":
                        continue
                    if function.input_args[input_key][0] == "scalar":
                        continue
                    # tensor input
                    else:
                        setattr(self, input_key + "_ptr", int(TorchFrontend.argument(kwargs[input_key])))
                        _kwargs[input_key + "_ptr"] = getattr(self, input_key + "_ptr")

                # processing the return args
                for ret in function.returns:
                    setattr(self, "output_" + ret + "_ptr", int(TorchFrontend.argument(kwargs["output_" + ret])))
                    _kwargs["output_" + ret + "_ptr"] = getattr(self, "output_" + ret + "_ptr")

                _kwargs.update(kwargs)
                function.get_arguments(tree, tree.root, _kwargs)
                self.visitor_arg = tree.get_node(tree.root).data.argument
            
            def sync(self, stream_sync=True):
                if stream_sync:
                    err, = cudart.cudaDeviceSynchronize()
                    if err != cuda.CUresult.CUDA_SUCCESS:
                        raise RuntimeError("CUDA Error %s" % str(err))
                
                if err != cuda.CUresult.CUDA_SUCCESS:
                    raise RuntimeError("CUDA Error %s" % str(err))
                pass
        
        self.epilogue_type = _Argument
