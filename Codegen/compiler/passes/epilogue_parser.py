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
from passes.iterator_v3 import *
import operator
import philox


torch_2_cutlass_type = {
    torch.float32: cutlass.float32,
    torch.float16: cutlass.float16,
    torch.int64: cutlass.int64,
    torch.int32: cutlass.int32,
    torch.bool: cutlass.int8
}

def dfs(node, target):
    if node == target:
        return True
    else:
        try:
            if node.meta['topo_idx'] > target.meta['topo_idx']:
                return False
            if len(node.users) == 0:
                return False
            else:
                for user in node.users.keys():
                    if dfs(user, target): return True
                return False
        except:
            print("DFS Error!!!!")
            print(node)

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
        self.permute = node.meta['tensor'].get_permutation()
        # an attribute to track the layout of 
        if self.permute in [[0, 1, 2], [1, 0, 2], [0, 1]]:
            self.layout = cutlass.RowMajor
        elif self.permute in [[0, 2, 1], [2, 0, 1], [1, 0]]:
            self.layout = cutlass.ColumnMajor
        else:
            print(node)
            print(self.permute)
            raise NotImplementedError()
    
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
            # case 3: [0, 2, 1] -> [B, N, M]
            elif self.permute == [0, 2, 1]:
                ldt = kwargs["problem_size"][1]
                batch_stride = kwargs["problem_size"][0] * kwargs["problem_size"][1]
            # case 4: [2, 0, 1] -> [N, B, M]
            elif self.permute == [2, 0, 1]:
                ldt = kwargs["problem_size"][1] * batch_size
                batch_stride = kwargs["problem_size"][1]
            else:
                print(self.permute)
                raise NotImplementedError("Unsupported output permutation")
        else:
            ldt = kwargs["problem_size"][1]
            batch_stride = kwargs["problem_size"][0] * kwargs["problem_size"][1]

        self.argument = self.epilogue_node.argument_type(kwargs[self.id + "_ptr"], ldt, *visitor_args, batch_stride)

        

class RowReductionNodeDAG(RowReductionNode):
    def __init__(self, element_accumulator, element_reduction, element_reduction_accumulator, node, atomic=True) -> None:
        super().__init__(element_accumulator, element_reduction, element_reduction_accumulator, "output_" + node.name, 1., atomic)
        self.node = node
        non_reduction_dims = [iter_var for iter_var in node.meta["tensor"].get_iter_vars() if iter_var.extent > 1]
        if non_reduction_dims[0].name == "m":
            self.factor, self.mod = node.meta["tensor"].get_iterator_factor_mod('b', 'b')
        elif non_reduction_dims[0].name == "b.1+m":
            self.factor, self.mod = node.meta["tensor"].get_iterator_factor_mod('b', 'b.1')
        else:
            raise NotImplementedError
        print("factor: %d, mod: %d" % (self.factor, self.mod))
    
    def get_argument(self, visitor_args, kwargs):
        self.argument = self.epilogue_node.argument_type(kwargs[self.id + "_ptr"], *visitor_args, self.get_batch_stride(kwargs["problem_size"]), self.factor, self.mod)

class ColumnReductionNodeDAG(ColumnReductionNode):
    def __init__(self, element_accumulator, element_reduction, element_reduction_accumulator, node, atomic=True) -> None:
        super().__init__(element_accumulator, element_reduction, element_reduction_accumulator, "output_" + node.name, 1., atomic)
        self.node = node
        non_reduction_dims = [iter_var for iter_var in node.meta["tensor"].get_iter_vars() if iter_var.extent > 1]
        if non_reduction_dims[0].name == "n":
            self.factor, self.mod = node.meta["tensor"].get_iterator_factor_mod('b', 'b')
        elif non_reduction_dims[0].name == "b.1+n":
            self.factor, self.mod = node.meta["tensor"].get_iterator_factor_mod('b', 'b.1')
        else:
            raise NotImplementedError
        print("factor: %d, mod: %d" % (self.factor, self.mod))
    
    def get_argument(self, visitor_args, kwargs):
        self.argument = self.epilogue_node.argument_type(kwargs[self.id + '_ptr'], *visitor_args, self.get_batch_stride(kwargs["problem_size"]), self.factor, self.mod)

# operators
operators = {
    torch.ops.aten.add: "Add",
    torch.ops.aten.div: "Div",
    torch.ops.aten.sub: "Sub",
    torch.ops.aten.mul: "Mult",
    torch.ops.aten.neg: "Mult",
    torch.ops.aten.ne: "Ne",
    torch.ops.aten.gelu: "GeLU",
    torch.ops.aten.tanh: "tanh",
    torch.ops.aten.tanh_backward: "TanhBackward",
    torch.ops.aten.gelu_backward: "GeluBackward"
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
        elif anchor.target in [torch.ops.aten._softmax, torch.ops.aten._softmax_backward_data, torch.ops.aten.native_layer_norm, torch.ops.aten.native_layer_norm_backward]:
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
        # seed, offset = philox.philox_state(1024)
        seed = 0
        offset = 0
        self.argument = self.epilogue_node.argument_type(
            self.p, seed, offset, kwargs[self.mask_ptr],
            kwargs["problem_size"][1], 
            kwargs["problem_size"][0] * kwargs["problem_size"][1],
            *visitor_args
        )

class TensorInputNodeDAG(TensorInputNode):
    def __init__(self, element_accumulator, node) -> None:
        self.id = node.name
        self.tag = "TensorInput:" + self.id
        self.type = "tensor"
        self.element_accumulator = element_accumulator
        self.tensor = node.meta["tensor"]
        self.element_input = torch_2_cutlass_type[node.meta['tensor_meta'].dtype]
    
    def get_argument(self, visitor_args, kwargs):
        factor, modulo = self.tensor.get_batch_factor_mod()
        if factor is not None:
            args = [
                kwargs[self.id + "_ptr"], 
                kwargs["problem_size"][1], 
                kwargs["problem_size"][0] * kwargs["problem_size"][1],
                factor, modulo]
            self.argument = self.epilogue_node.argument_type(*args)
        else:
            return super().get_argument(visitor_args, kwargs)

class RowBroadcastNodeDAG(RowBroadcastNode):
    def __init__(self, element_accumulator, element_fragment, node, element_input=None) -> None:
        self.id = node.name
        self.tag = "RowBroadcast:" + self.id
        self.type = "tensor"
        self.element_accumulator = element_accumulator
        self.element_fragment = element_fragment
        self.element_input = element_input
        self.tensor = node.meta["tensor"]
    
    def get_argument(self, visitor_args, kwargs):
        factor, modulo = self.tensor.get_batch_factor_mod()
        if factor is not None:
            args = [
                kwargs[self.id + "_ptr"], kwargs["problem_size"][1],
                factor, modulo
            ]
            self.argument = self.epilogue_node.argument_type(*args)
        else:
            return super().get_argument(visitor_args, kwargs)

class ColumnBroadcastNodeDAG(ColumnBroadcastNode):
    def __init__(self, element_accumulator, element_fragment, node, element_input=None) -> None:
        self.id = node.name
        self.tag = "ColumnBroadcast:" + self.id
        self.type = "tensor"
        self.element_accumulator = element_accumulator
        self.element_fragment = element_fragment
        self.element_input = element_input
        self.tensor = node.meta["tensor"]
    
    def get_argument(self, visitor_args, kwargs):
        factor, modulo = self.tensor.get_batch_factor_mod()
        if factor is not None:
            args = [
                kwargs[self.id + "_ptr"], kwargs["problem_size"][0],
                factor, modulo
            ]
            self.argument = self.epilogue_node.argument_type(*args)
        else:
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
    def __init__(self, anchor, element_accumulator,
        elements_per_access, element_compute, element_output) -> None:
        #
        
        self.element_accumulator = element_accumulator
        self.elements_per_access = elements_per_access
        self.element_compute = element_compute
        self.element_output = element_output
        self.anchor = anchor
        
        self.shape = list(anchor.meta['tensor_meta'].shape)
        self.get_root()


        self.outputs = []
        # TODO: do we really need this?
        self.ast_top_down_parsing_bmm(self.root, parse_output=True)

        self.stack = [None, ]
        self.get_item_stack = []
        self.reduction_source = {}

        self.input_args = {}

        # this list tracks the list of input nodes
        self.args = []

        # parse epilogue tree from DAG
        self.epilogue_tree = Tree()

        self.output_layouts = []

        self.visit(self.root)

        # build relationship between output and kernel output
        self.kernel_outputs = [output for output in self.outputs]

        self.output_2_kernel_output = {}

        for output, k_output in zip(self.outputs, self.kernel_outputs):
            self.output_2_kernel_output[output] = k_output

        
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

    def pass_inject_epilogue_op(self, tree, nid):
        node = tree.get_node(nid)
        visitors = []
        for child in node.successors(tree.identifier):
            visitors.append(self.pass_inject_epilogue_op(tree, child))
        
        node.data.get_epilogue_node(visitors)
        return node.data.epilogue_node
    
    def get_fx_node_with_tree_node(self, node):
        name = node.data.id.strip("output_")
        for fx_node in self.kernel_outputs:
            if fx_node.name == name:
                return fx_node
        raise ValueError()
    
    def pass_output_node_fusion(self, tree, nid):
        node = tree.get_node(nid)
        if isinstance(node.data, TensorOutputNodeDAG):
            input_node = tree.get_node(node.successors(tree.identifier)[0])
            if isinstance(input_node.data, TensorOutputNodeDAG):
                tree.move_node(
                    input_node.successors(tree.identifier)[0],
                    nid
                )
                tree.remove_node(input_node.data.id)

                # remove the node from kernel outputs
                fx_output_node = self.get_fx_node_with_tree_node(input_node)
                fx_kernel_output_node = self.get_fx_node_with_tree_node(node)
                self.kernel_outputs.remove(fx_output_node)
                self.output_2_kernel_output[fx_output_node] = fx_kernel_output_node
    
    def get_candidate_root_bmm(self, node):
        """
        This function performs DFS search for all the candidate root nodes for 
        the epilogue visitor tree
        """
        if node.op == "call_function":
            # get all user nodes
            user_nodes = list(node.users.keys())
            # simple heuristic to stop dropout from tracing too deep
            if node.target == torch.ops.aten.native_dropout:
                node.meta['tensor'].get_node_tensor_bottom_up(user_nodes[1])
                user_nodes = [user_nodes[0], ]
            for usr in user_nodes:
                if node.target == torch.ops.aten.permute:
                    if "epilogue_permute" not in node.meta.keys():
                        self.root_candidates[node.args[0]] = []
                        continue
                try:
                    node.meta['tensor'].get_node_tensor_bottom_up(usr)
                except NotImplementedError:
                    self.root_candidates[node] = []
                    continue
                except:
                    print(usr)
                    assert 0
                if usr.target in [torch.ops.aten.sum]:
                    self.root_candidates[node] = [usr]
                    continue
                self.root_candidates[node] = []
                self.get_candidate_root_bmm(usr)

    
    def ast_top_down_parsing_bmm(self, node, parse_output=False):
        """
        This function parses the ast from the root in the top-down manner
        """
        # step 1: check if the node is a dependency of anchor
        if node.op == "call_function":
            if node != self.anchor and dfs(node, self.anchor):
                return [node,]
        else:
            return [node,]
        if parse_output:
            if len(node.users) > 1 or node == self.root:
                # case for multiple output nodes. All its users are getitem
                if list(node.users.keys())[0].target == operator.getitem:
                    for user in node.users:
                        assert user.target == operator.getitem
                        if "in_ast" in user.meta.keys():
                            if len(user.users) > 1:
                                if user not in self.outputs:
                                    self.outputs.append(user)
                        else:
                            if user not in self.outputs:
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
                    if "unfusible" not in node.meta.keys():
                        if parse_output:
                            node.meta["in_ast"] = True
                        return self.ast_top_down_parsing_bmm(node.args[0], parse_output)
                    else:
                        return self.ast_top_down_parsing_bmm(node.args[0], parse_output)
                        # return []
                elif node.target in [torch.ops.aten.add, torch.ops.aten.sub,
                    torch.ops.aten.mul, torch.ops.aten.div, torch.ops.aten.tanh_backward, torch.ops.aten.gelu_backward]:
                    #
                    if isinstance(node.args[0], fx.Node):
                        lhs_nodes = self.ast_top_down_parsing_bmm(node.args[0], parse_output)
                    else:
                        lhs_nodes = []
                    if isinstance(node.args[1], fx.Node):
                        rhs_nodes = self.ast_top_down_parsing_bmm(node.args[1], parse_output)
                    else:
                        rhs_nodes = []
                    return lhs_nodes + rhs_nodes + [node, ]
                elif node.target in [torch.ops.aten.neg, torch.ops.aten.sum, torch.ops.aten.native_dropout, torch.ops.aten.permute, torch.ops.aten.gelu, torch.ops.aten.tanh, torch.ops.aten.one_hot, torch.ops.aten.ne]:
                    return self.ast_top_down_parsing_bmm(node.args[0], parse_output) + [node,]
                elif node.target in [torch.ops.aten.mm, torch.ops.aten._softmax, torch.ops.aten.bmm, torch.ops.aten._softmax_backward_data, torch.ops.aten.native_layer_norm, torch.ops.aten.native_layer_norm_backward]:
                    return [node,]
                else:
                    return []
            except (NotImplementedError, AssertionError):
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
            iter_names = ['b', 'm', 'n']
            shape = list(self.anchor.meta["tensor_meta"].shape)
            self.anchor.meta["tensor"] = IterVarHyperGraph(shape, iter_names)
        elif self.anchor.target == torch.ops.aten.mm:
            iter_names = ['m', 'n']
            shape = list(self.anchor.meta["tensor_meta"].shape)
            self.anchor.meta["tensor"] = IterVarHyperGraph(shape, iter_names)
        elif self.anchor.target in [
            torch.ops.aten._softmax, torch.ops.aten._softmax_backward_data, 
            torch.ops.aten.native_layer_norm, torch.ops.aten.native_layer_norm_backward]:
            # The softmax can take an arbitrary dim 
            shape = self.anchor.meta["tensor_meta"].shape
            if self.anchor.target == torch.ops.aten._softmax:
                reduction_dim = self.anchor.args[1]
            elif self.anchor.target in [torch.ops.aten.native_layer_norm, torch.ops.aten.native_layer_norm_backward]:
                reduction_dim = -1
            else:
                reduction_dim = self.anchor.args[2]
            if reduction_dim < 0:
                reduction_dim = len(shape) + reduction_dim
            independent_size = 1
            for idx, dim in enumerate(shape):
                if idx == reduction_dim: continue
                independent_size *= dim
            shape = (independent_size, shape[reduction_dim])
            iter_names = ['m', 'n']
            graph = IterVarHyperGraph(shape, iter_names)
            graph.view(self.anchor.meta["tensor_meta"].shape)
            self.anchor.meta["tensor"] = graph
        else:
            raise NotImplementedError()
        self.get_candidate_root_bmm(self.anchor)

        # filter root candiates
        # for root in self.root_candidates.keys():
        #     if root.target == torch.ops.aten.sum:
        #         for others in self.root_candidates.keys():
        #             if others == root: continue
        #             if root.args[0] in others.args:
        #                 self.root_candidates[others].append(root)
        #                 self.root_candidates[root] = None
        
        if self.anchor.target in [torch.ops.aten.bmm, torch.ops.aten.mm, torch.ops.aten._softmax, torch.ops.aten._softmax_backward_data, torch.ops.aten.native_layer_norm, torch.ops.aten.native_layer_norm_backward]:
            root_to_remove = []
            for root in self.root_candidates.keys():
                if self.root_candidates[root] is not None:
                    node_list = self.ast_top_down_parsing_bmm(root)
                    self.root_candidates[root] += node_list
                    if self.anchor not in self.root_candidates[root]:
                        self.root_candidates[root] = None
                        root_to_remove.append(root)
                    for n in node_list:
                        if n.target in [torch.ops.aten.bmm, torch.ops.aten.mm, torch.ops.aten._softmax, torch.ops.aten._softmax_backward_data, torch.ops.aten.native_layer_norm, torch.ops.aten.native_layer_norm_backward]:
                            if n.meta['topo_idx'] > self.anchor.meta['topo_idx']:
                                if root not in root_to_remove:
                                    root_to_remove.append(root)
                        elif n.target == torch.ops.aten.permute and self.anchor.target in [torch.ops.aten._softmax, torch.ops.aten._softmax_backward_data, torch.ops.aten.native_layer_norm, torch.ops.aten.native_layer_norm_backward]:
                            if root not in root_to_remove:
                                root_to_remove.append(root)
                else:
                    root_to_remove.append(root)
            for root in root_to_remove:
                self.root_candidates.pop(root)
        else:
            raise NotImplementedError
        
        print("Root Candidates:")
        print(self.root_candidates)## A new filter node to ensure fusion is in topological order
        
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
            elif cost == max_cost and root.meta['topo_idx'] > self.root.meta['topo_idx']:
                max_cost = cost
                self.root = root
        print("Root:")
        print(self.root)
        # TODO: handle the case that cost of two graphs is the same

    def visit(self, node):
        
        reduction_node = None
        ### inject reduction
        for user_node in list(node.users.keys()):
            # step 1: get the reduction user node
            if user_node.target in [torch.ops.aten.sum]:
                # step 2: get the non-reduction direction
                assert "tensor" in list(user_node.meta.keys())
                non_reduction_dims = [iter_var for iter_var in user_node.meta["tensor"].get_iter_vars() if iter_var.extent > 1]
                reduction_type = None
                # currently we only support row reduction and column reduction
                if len(non_reduction_dims) == 1:
                    if non_reduction_dims[0].name == "m":
                        reduction_type = "row_reduction"
                    elif non_reduction_dims[0].name == "n":
                        reduction_type = "column_reduction"
                    elif non_reduction_dims[0].name == "b.1+m":
                        reduction_type = "row_reduction"
                    elif non_reduction_dims[0].name == "b.1+n":
                        reduction_type = "column_reduction"
                
                if reduction_type is not None:
                    # case 1: for GEMM kernels
                    if self.anchor.target in [torch.ops.aten.mm, torch.ops.aten.bmm]:
                        # create reduction node
                        if reduction_type == "row_reduction":
                            reduction_node = RowReductionNodeDAG(
                                self.element_accumulator, self.element_output,
                                self.element_accumulator, user_node
                            )
                        elif reduction_type == "column_reduction":
                            reduction_node = ColumnReductionNodeDAG(
                                self.element_accumulator, self.element_output,
                                self.element_accumulator, user_node
                            )
                        else:
                            raise NotImplementedError()
                        self.outputs.append(user_node)
        
        if reduction_node is not None:
            self.epilogue_tree.create_node(reduction_node.tag, reduction_node.id, parent=self.stack[-1], data=reduction_node)
            self.stack.append(reduction_node.id)
        
        is_input = False
        # TODO: the tensor input can be output of another node
        if node.op == "call_function":
            if node != self.anchor and dfs(node, self.anchor):
                # the node will be treated as a tensor input node
                is_input = True
            elif node.target in [operator.getitem]:
                if "unfusible" in node.meta.keys():
                    is_input = True
        else:
            is_input = True
        if self.anchor.target in [torch.ops.aten.bmm, torch.ops.aten.mm, torch.ops.aten._softmax, torch.ops.aten._softmax_backward_data]:
            if 'tensor' not in node.meta.keys():
                print(node)
                raise RuntimeError("The node to fuse doesn't have tensor attribute")
        is_output = False

        if node in self.outputs:
            if not is_input:
            # assert not is_input
                if reduction_node is None or len(node.users.keys()) != 1:
                    # visit the output node
                    name_node = TensorOutputNodeDAG(self.element_accumulator, node)
                    if name_node.layout not in self.output_layouts:
                        self.output_layouts.append(name_node.layout)
                    self.epilogue_tree.create_node(name_node.tag, name_node.id, parent=self.stack[-1], data=name_node)
                    self.stack.append(name_node.id)
                    is_output = True
            if not is_output:
                self.outputs.remove(node)

        if node.op == "call_function" and not is_input:
            if node.target in [torch.ops.aten.view, torch.ops.aten._unsafe_view, torch.ops.aten.unsqueeze, torch.ops.aten._to_copy, torch.ops.aten.clone, torch.ops.aten.permute]:
                self.visit(node.args[0])
            elif node.target in [operator.getitem]:
                if "unfusible" in node.meta.keys():
                    is_input = True
                else:
                    # if there is multiple output
                    self.get_item_stack.append(node)
                    self.visit(node.args[0])
                    self.get_item_stack.pop(-1)
            elif node.target in [torch.ops.aten.add, torch.ops.aten.mul, torch.ops.aten.sub, torch.ops.aten.div, torch.ops.aten.tanh_backward, torch.ops.aten.gelu_backward]:
                # check number of nonconstant node
                if len(node.all_input_nodes) == 1:
                    args = []
                    for arg in node.args:
                        if arg not in node.all_input_nodes:
                            args.append(arg)
                    binop = UnaryNodeDAG(
                        self.element_accumulator, self.element_compute,
                        self.elements_per_access, node, args
                    )
                else:
                    binop = BinOpNodeDAG(
                        self.element_accumulator, self.element_compute,
                        self.elements_per_access, node)
                self.epilogue_tree.create_node(
                    binop.tag, binop.id, data=binop,
                    parent=self.stack[-1])
                self.stack.append(binop.id)
                for arg in node.all_input_nodes:
                    self.visit(arg)
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
            elif node.target in [torch.ops.aten.gelu, torch.ops.aten.tanh]:
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
            elif node == self.anchor:
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
                # raise NotImplementedError
                # TODO: currently we treat all unknown nodes as input nodes
                is_input = True

        if node.op in ["placeholder", "get_attr"] or is_input:
            if self.anchor.target in [
                torch.ops.aten.bmm, torch.ops.aten.mm, 
                torch.ops.aten._softmax, torch.ops.aten._softmax_backward_data,
                torch.ops.aten.native_layer_norm, torch.ops.aten.native_layer_norm_backward]:
                if node.target in [torch.ops.aten.native_dropout]:
                    node = self.get_item_stack[-1]
                source_type = torch_2_cutlass_type[node.meta['tensor_meta'].dtype]

                node_tensor = node.meta['tensor']

                tensor_type = node_tensor.get_tensor_type()
                # case 1: scalar
                if tensor_type == "scalar":
                    name_node = ScalarInputNodeDAG(node)
                # case 2: row broadcast
                elif tensor_type == "row":
                    name_node = RowBroadcastNodeDAG(
                        self.element_accumulator, self.element_compute, node, source_type
                    )
                # case 3: column broadcast
                elif tensor_type == "column":
                    name_node = ColumnBroadcastNodeDAG(
                        self.element_accumulator, source_type, node, source_type
                    )
                # case 4: tensor
                elif tensor_type == "tensor":
                    name_node = TensorInputNodeDAG(self.element_accumulator, node)
                else:
                    raise NotImplementedError()
                
                print(tensor_type)
            else:
                raise NotImplementedError("Anchor type %s is not supported" % str(self.anchor.target))
                
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

        if is_output:
            self.stack.pop()
        
        if reduction_node is not None:
            self.stack.pop()
        


    def get_arguments(self, tree, nid, kwargs):
        node = tree.get_node(nid)
        visitor_args = []
        for child in node.successors(tree.identifier):
            visitor_args.append(self.get_arguments(tree, child, kwargs))
        node.data.get_argument(visitor_args, kwargs)
        return node.data.argument

class EpilogueVisitTreeDAG:
    KernelTemplate = """
${visitor}

using ${operation_name}_EpilogueVisitor = cutlass::epilogue::threadblock::EpilogueVisitorGeneric<${visitor_name}>;
""" 
    # Epilogue visitor tree from DAG
    def __init__(
        self, elementwise_functor, element_accumulator, 
        elements_per_access, element_compute, element_output) -> None:
        #
        # data types
        self.element_accumulator = element_accumulator
        self.elements_per_access = elements_per_access
        self.element_compute = element_compute
        self.element_output = element_output
        # TODO: deprecate this
        self.elementwise_functor = elementwise_functor

    def initialize(self, node: fx.Node):
        # TODO: some EpilogueAST structure
        function = EpilogueASTDAG(
            node, self.element_accumulator,
            self.elements_per_access, self.element_compute, self.element_output
        )

        tree = function.epilogue_tree
        self.tree = tree
        self.args = function.args
        self.root = function.root
        self.function = function
        self.outputs = function.outputs
        self.kernel_outputs = function.kernel_outputs
        self.output_2_kernel_output = function.output_2_kernel_output
        try:
            assert len(function.output_layouts) == 1
        except AssertionError:
            print("Outputs can only have one kind of layers, get %d" % len(function.output_layouts))
            exit()
        self.output_layout = function.output_layouts[0]

        # check whether output is simple TensorOutput
        if node.target == torch.ops.aten.mm:
            root_node = self.tree.get_node(self.tree.root)
            if isinstance(root_node.data, TensorOutputNodeDAG):
                child_node = self.tree.get_node(root_node.successors(self.tree.identifier)[0])
                if isinstance(child_node.data, AccumulatorNodeDAG):
                    return True
        return False
    
    def optimize(self, tile_description):
        self.function.tile_description = tile_description
        # tree = self.tree
        function = self.function

        self.tree.show()

        function.pass_binary_2_unary(self.tree, self.tree.root)
        # function.pass_inject_reduction(self.tree, self.tree.root)
        function.pass_output_node_fusion(self.tree, self.tree.root)
        function.pass_inject_epilogue_op(self.tree, self.tree.root)

        visitor = self.tree.get_node(self.tree.root).data.epilogue_node
        self.visitor = visitor
        self.kernel_outputs = function.kernel_outputs
        self.output_2_kernel_output = function.output_2_kernel_output
        # if function.anchor.target == torch.ops.aten._softmax:
        self.tree.show()
        tree = self.tree
        
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
                        if isinstance(kwargs[input_key], tuple):
                            print(input_key)
                            tree.show()

                        setattr(self, input_key + "_ptr", int(TorchFrontend.argument(kwargs[input_key])))
                        _kwargs[input_key + "_ptr"] = getattr(self, input_key + "_ptr")

                # processing the return args
                for ret in function.kernel_outputs:
                    setattr(self, "output_" + ret.name + "_ptr", int(TorchFrontend.argument(kwargs["output_" + ret.name])))
                    _kwargs["output_" + ret.name + "_ptr"] = getattr(self, "output_" + ret.name + "_ptr")

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
    
    def emit(self, operation):
        values = {
            'visitor': self.visitor.emit(operation),
            'operation_name': operation.procedural_name(),
            'visitor_name': self.visitor.instance_name
        }
        return SubstituteTemplate(self.KernelTemplate, values)
