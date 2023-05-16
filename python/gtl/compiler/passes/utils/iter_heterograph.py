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
# This file contains the base class of the Iterator Hypergraph used in GTL
################################################################################
import networkx as nx
import matplotlib.pyplot as plt


################################################################################
# Iterator Variable Node
################################################################################
class IterVarBase:
    def __init__(self, name, extent) -> None:
        self.name = name
        self.extent = extent
        self.valid = True
    
    def __str__(self):
        return "IterVar: %s(%d)" % (self.name, self.extent)

################################################################################
# Iterator Mutators
################################################################################
class SplitNodeBase:
    def __init__(self, iter_var, shape):
        self.name = "split " + iter_var.name
        self.shape = shape
    
    def __str__(self):
        return "Split: " + str(self.shape)

class MergeNodeBase:
    def __init__(self, iter_var_list) -> None:
        self.name = "merge ("
        for iter_var in iter_var_list:
            self.name += iter_var.name + ","
        self.name += ")"

class DebroadcastNodeBase:
    def __init__(self, iter_var) -> None:
        self.name = "debroadcast" + iter_var.name


class IterHeteroGraphBase:
    def __init__(self, shape) -> None:
        """
        Args:
            shape: the tensor shape
        """
        # type name can be overwritten
        self.IterVarType = IterVarBase
        self.SpliteNodeType = SplitNodeBase
        self.MergeNodeType = MergeNodeBase
        self.DebroadcastNodeType = DebroadcastNodeBase

        # The heterograph
        self.graph = nx.DiGraph()
        
        # Insert root iterVar
        numel = 1
        for dim in shape:
            numel *= dim
        root = IterVarBase(name="root", extent=numel)
        self.graph.add_node(root.name, attr={'data': root})
        # Split the iterators according the new shape
        self.split(root, shape)
    
    def show(self):
        nx.draw(self.graph, with_labels=True)
        plt.savefig("./iter_var_graph.png")
    
    ############################################################################
    # Helper functions
    ############################################################################
    def get_edge_type(self, edge):
        """
        Getting the type of the input edge
        returns:
            "order" for order edges
            None for mutator edges
        """
        edge_attrs =  nx.get_edge_attributes(self.graph, 'type')
        try:
            return edge_attrs[edge]
        except:
            return None
    
    def infer_split(self, shape, iter_vars):
        """
        Infer how to split the iterator variables given the new shape recursively
        """
        if len(shape) == 0 and len(iter_vars) == 0:
            return
        # this is done recursively by only process the last iter var at each time
        dim = shape[-1]
        iter_var = iter_vars[-1]
        # exact match
        if iter_var.extent == dim:
            # move to the next iter
            return self.infer_split(shape[:-1], iter_vars[:-1])
        # needs split
        if iter_var.extent > dim:
            assert iter_var.extent % dim == 0
            node_0, node_1 = self.split(iter_var, shape=(iter_var.extent // dim, dim))
            return self.infer_split(shape[:-1], iter_vars[:-1] + [node_0,])
        # needs merge
        elif iter_var.extent < dim:
            assert dim % iter_var.extent == 0
            return self.infer_split(shape[:-1] + [dim // iter_var.extent,], iter_vars[:-1])
        
        raise NotImplementedError()
    
    def infer_merge(self, shape, iter_vars):
        if len(shape) == 0 and len(iter_vars) == 0:
            return
        
        for dim in reversed(shape):
            merge_list = []
            while dim != 1:
                merge_list.append(iter_vars[-1])
                dim /= iter_vars[-1].extent
                iter_vars.pop(-1)
            
            if len(merge_list) > 1:
                self.merge(merge_list)


    
        
    def get_order_edge(self, node):
        """
        Getting the outgoing order edge of the input node
        """
        # for node with out degree > 0
        if self.graph.out_degree(node.name) > 0:
            edges = list(nx.edges(self.graph, node.name))
            for edge in edges:
                edge_type = self.get_edge_type(edge)
                # only care about order edge here
                if edge_type == "order":
                    return edge
        return None

    def get_iter_vars(self):
        nodes = list(nx.topological_sort(self.graph))
        iter_vars = []
        for node in reversed(nodes):
            node_data = self.graph.nodes[node]['attr']['data']
            if self.graph.out_degree(node) == 0:
                if node_data.valid:
                    iter_vars.append(node_data)
            elif self.graph.out_degree(node) == 1:
                edge = list(nx.edges(self.graph, node))[0]
                edge_type = self.get_edge_type(edge)
                if edge_type == "order":
                    if node_data.valid:
                        iter_vars.append(node_data)


        return iter_vars
    
    def print_iter_vars(self):
        iter_vars = self.get_iter_vars()
        iter_var_names = [iter_var.name for iter_var in iter_vars]
        print(iter_var_names)
    
    def get_shape(self):
        iter_vars = self.get_iter_vars()
        shape = []
        for var in iter_vars:
            shape.append(var.extent)
        return shape


    ############################################################################
    # Basic multation functions
    ############################################################################
    def split(self, iter_var, shape, names=None):
        """
        Split an iterator variable
        """
        # insert the split node
        split = self.SpliteNodeType(iter_var, shape)
        self.graph.add_node(split.name, attr={'data': split})
        # add mutator edge between iter_var -> the split node
        self.graph.add_edge(*(iter_var.name, split.name))

        # create name for new iterators if not given
        assert names is None or len(shape) == len(names)
        if names is None:
            names = []
            for idx in range(len(shape)):
                names.append(iter_var.name + ".%d" % (idx))
        
        # create the new iterator variables
        nodes = [] # buffer containing the new iterator nodes
        previous_node = None
        for dim, name in zip(shape, names):
            # create iterator variable
            node = self.IterVarType(name, dim)
            self.graph.add_node(node.name, attr={'data': node})
            # add mutator edge between split node -> new iter var
            self.graph.add_edge(*(split.name, node.name))
            # add order edge m.0 <- m.1
            if previous_node is not None:
                self.graph.add_edge(*(node.name, previous_node.name), type="order")
            previous_node = node
            nodes.append(node)
        
        # propagate the order edge
        # get the out going order edge : x <- m 
        order_edge = self.get_order_edge(iter_var)
        # if the order edge exists
        if order_edge is not None:
            # remove the original order edge
            self.graph.remove_edge(*order_edge)
            # add new order edge x <- m.0
            new_order_edge = (nodes[0].name, order_edge[1])
            self.graph.add_edge(*new_order_edge, type="order")

        return nodes
    
    def merge(self, iter_vars):
        """
        Merge a list of iterator variables
        """
        # insert the merge node
        merge = self.MergeNodeType(iter_vars)
        self.graph.add_node(merge.name, attr={'data': merge})

        # get the name & extent of the merged iterators
        merged_iter_var_node = ""
        extent = 1
        for iter_var in iter_vars:
            merged_iter_var_node = iter_var.name + "+" + merged_iter_var_node
            extent *= iter_var.extent

        # create the merged iterator variable
        merged_node = self.IterVarType(name=merged_iter_var_node[:-1], extent=extent)
        self.graph.add_node(merged_node.name, attr={'data': merged_node})
        # add edge merge_node -> merged iter var
        self.graph.add_edge(*(merge.name, merged_node.name))
        # add edges iter_var -> merge_node
        for iter_var in iter_vars:
            self.graph.add_edge(*(iter_var.name, merge.name))
        
        # update the order edge
        # the input iter vars are connected by a order edges as a chain
        # iter_vars[0] -> iter_vars[1] ... -> iter_vars[-1] -> x
        order_edge = self.get_order_edge(iter_vars[-1])
        if order_edge is not None:
            self.graph.remove_edge(*order_edge)
            # create new order edge: merged iter_var -> x
            new_order_edge = (merged_node.name, order_edge[1])
            self.graph.add_edge(*new_order_edge, type="order")
    
    def _debroadcast(self, iter_var):
        """
        Shunk a dimension to 1, usually caused by reduction or backward analysis
        """
        # inject the debroadcast mutator node
        debroadcast_op = self.DebroadcastNodeType(iter_var)
        self.graph.add_node(debroadcast_op.name, attr={'data': debroadcast_op})
        self.graph.add_edge(iter_var.name, debroadcast_op.name)

        # create the new iter var with extent 1
        debroadcast_node = self.IterVarType(name="db " + iter_var.name, extent=1)
        self.graph.add_node(debroadcast_node.name, attr={'data': debroadcast_node})
        self.graph.add_edge(debroadcast_op.name, debroadcast_node.name)

        # update the order edge
        # x -> iter_var => x -> debroadcasted iter var
        order_edge = self.get_order_edge(iter_var)
        if order_edge is not None:
            self.graph.remove_edge(*order_edge)
            new_order_edge = (debroadcast_node.name, order_edge[1])
            self.graph.add_edge(*new_order_edge, type="order")
        
    ############################################################################
    # High-level mutation functions
    # They will be decomposed to a set of basic mutation functions
    ############################################################################
    def view(self, shape):
        """
        For torch.ops.aten.view
        """
        # preprocessing shape
        # The input shape sometimes contains -1
        if -1 in shape:
            # get current shape & num element
            current_shape = self.get_shape()
            numel = 1
            for dim in current_shape: numel *= dim
            neg_idx = None
            # get the "-1" index and its actual size
            for idx, dim in enumerate(shape):
                if dim != -1:
                    numel /= dim
                else:
                    neg_idx = idx
            # update the shape
            shape[neg_idx] = numel
        
        # get the iterator variables
        iter_var_list = self.get_iter_vars()
        self.infer_split(shape, iter_var_list)
        # simplification
        self.pass_flat_order_edges()

        # infer the merge ops
        iter_var_list = self.get_iter_vars()
        self.infer_merge(shape, iter_var_list)
        # simplification
        self.pass_flat_order_edges()
    
    def squeeze(self, squeeze_idx):
        """
        For torch.ops.aten.squeeze
        """
        iter_var_list = self.get_iter_vars()
        squeezed_iter_var = iter_var_list[squeeze_idx]
        assert squeezed_iter_var.extent == 1
        # invalide the iter var as it is squeezed
        self.graph.nodes[squeezed_iter_var.name]["attr"]["data"].valid = False
    
    ############################################################################
    # Current Progress

    
    def debroadcast(self, shape):
        iter_vars = self.get_iter_vars()
        # we follow the pytorch broadcasting sementics
        # https://pytorch.org/docs/stable/notes/broadcasting.html
        # Two tensors are "broadcastable" if the following rules hold
        # * Each tensor has at least one dimension
        # * When iterating over the dimension sizes, starting at the trailing 
        #   dimension, the dimension sizes must either be equal, one of them is 
        #   1, or one of them does not exist
        iter_names = []
        for dim, iter_var in zip(reversed(shape), reversed(iter_vars)):
            if dim == 1:
                self._debroadcast(iter_var)
            if dim in [1, iter_var.extent]:
                iter_names.append(iter_var.name)
        for iter_var in iter_vars:
            if iter_var.name not in iter_names:
                self.graph.nodes[iter_var.name]["attr"]["data"].valid = False

        self.pass_constant_reduction()
    
    def permute(self, permute_idx):
        # get all iter vars
        iter_var_list = self.get_iter_vars()
        # remove all edges
        self.remove_order_edges()
        permuted_iter_vars = [iter_var_list[idx] for idx in permute_idx]
        # add new edges
        self.add_order_edges(permuted_iter_vars)
    
    def get_index(self, indices):
        if not isinstance(indices, list):
            indices = [indices,]
        iter_var_list = self.get_iter_vars()
        for idx, var in enumerate(iter_var_list):
            if idx not in indices:
                self.graph.nodes[var.name]["attr"]["data"].valid = False
    
    def reduce(self, indices):
        iter_vars = self.get_iter_vars()
        for idx in indices:
            self._debroadcast(iter_vars[idx])

    
    ### Graph optimization passes
    def remove_order_edges(self):
        edges = self.graph.edges()
        order_edges = []
        for edge in edges:
            edge_type = self.get_edge_type(edge)
            if edge_type == "order":
                order_edges.append(edge)
        for edge in order_edges:
            self.graph.remove_edge(*edge)
    
    def add_order_edges(self, iter_vars):
        for i in range(len(iter_vars) - 1):
            self.graph.add_edge(*(iter_vars[i+1].name, iter_vars[i].name), type="order")


    def pass_flat_order_edges(self):
        # get all iter vars
        iter_var_list = self.get_iter_vars()
        # remove all edges
        self.remove_order_edges()
        # add new edges
        self.add_order_edges(iter_var_list)
        
    
    def pass_fuse_split_merge(self):
        merge_nodes = []
        for node in self.graph.nodes():
            node_data = self.graph.nodes[node]["attr"]["data"]
            if isinstance(node_data, MergeNode):
                merge_nodes.append(node_data)
        
        for merge_node in merge_nodes:
            node_merged = list(self.graph.predecessors(merge_node.name))
            predecessors = {}
            for node in node_merged:
                inputs = list(self.graph.predecessors(node))
                if len(inputs) == 1:
                    input_data = self.graph.nodes[inputs[0]]["attr"]["data"]
                    if isinstance(input_data, SplitNode):
                        if input_data.name not in predecessors.keys():
                            predecessors[input_data.name] = [self.graph.nodes[node]["attr"]["data"],]
                        else:
                            predecessors[input_data.name] += [self.graph.nodes[node]["attr"]["data"],]
            for pred in predecessors.keys():
                iter_vars = predecessors[pred]
                merged_iter_var_node = "fuse "
                extent = 1
                for iter_var in iter_vars:
                    merged_iter_var_node += iter_var.name + "+"
                    extent *= iter_var.extent
                merged_iter_var = IterVar(merged_iter_var_node[:-1], extent)
                self.graph.add_node(merged_iter_var.name, attr={'data': merged_iter_var})
                self.graph.add_edge(pred, merged_iter_var.name)
                self.graph.add_edge(merged_iter_var.name, merge_node.name)
                for iter_var in iter_vars:
                    self.graph.remove_node(iter_var.name) 
    
    def replace_all_uses_with(self, node_replaced, node):
        successors = list(self.graph.successors(node_replaced.name))
        precessors = list(self.graph.predecessors(node_replaced.name))
        self.graph.remove_node(node_replaced.name)
        for suc in successors:
            suc_node = self.graph.nodes[suc]["attr"]["data"]
            if isinstance(suc_node, IterVar):
                if node.name != suc:
                    self.graph.add_edge(node.name, suc, type="order")
            else:
                self.graph.add_edge(node.name, suc)
        for pre in precessors:
            pre_node = self.graph.nodes[pre]["attr"]["data"]
            if isinstance(pre_node, IterVar):
                if pre != node.name:
                    self.graph.add_edge(pre, node.name, type="order")
            else:
                self.graph.add_edge(pre, node.name)
        
    def get_node_data(self, node):
        return self.graph.nodes[node]["attr"]["data"]
    
    def pass_constant_reduction(self):
        # remove constant split nodes
        split_nodes = []
        for node in self.graph.nodes():
            node_data = self.graph.nodes[node]["attr"]["data"]
            if isinstance(node_data, SplitNode):
                split_nodes.append(node_data)
        
        for split_node in split_nodes:
            if len(list(self.graph.successors(split_node.name))) == 1:
                input_node = self.graph.nodes[list(self.graph.predecessors(split_node.name))[0]]["attr"]["data"]
                output_node = self.graph.nodes[list(self.graph.successors(split_node.name))[0]]["attr"]["data"]
                self.graph.remove_node(split_node.name)
                self.replace_all_uses_with(output_node, input_node)
        
        # remove constant merge nodes
        merge_nodes = []
        for node in self.graph.nodes():
            node_data = self.graph.nodes[node]["attr"]["data"]
            if isinstance(node_data, MergeNode):
                merge_nodes.append(node_data)
        for merge_node in merge_nodes:
            if len(list(self.graph.predecessors(merge_node.name))) == 1:
                input_node = self.graph.nodes[list(self.graph.predecessors(merge_node.name))[0]]["attr"]["data"]
                output_node = self.graph.nodes[list(self.graph.successors(merge_node.name))[0]]["attr"]["data"]
                self.graph.remove_node(merge_node.name)
                self.replace_all_uses_with(output_node, input_node)
    
    ### helper functions for EAST

    def get_permutation(self):
        # get the output permutation
        # our idea is concatenating the name list
        iter_vars = self.get_iter_vars()
        name = ""
        for var in iter_vars:
            if var.extent > 1:
                name += var.name
        
        valid_iters = {
            'b': None, 
            'm': None,
            'n': None
        }

        dim_idx = 0
        for s in name:
            if s in ['b', 'm', 'n']:
                valid_iters[s] = dim_idx
                dim_idx += 1
        
        permute = []
        for key in valid_iters.keys():
            if valid_iters[key] is not None:
                permute.append(valid_iters[key])
        return list(np.argsort(permute))
    
    def get_tensor_type_nchw(self):
        iter_vars = self.get_iter_vars()

        valid_iters = {
            "m.0.0": None,
            "n": None,
            "m.0.1": None,
            "m.1": None
        }

        try:
            for iter_var in iter_vars:
                if iter_var.extent > 1:
                    if "m.0.0" in iter_var.name:
                        valid_iters['m.0.0'] = iter_var
                    elif "n" in iter_var.name:
                        valid_iters['n'] = iter_var
                    elif "m.0.1" in iter_var.name:
                        valid_iters['m.0.1'] = iter_var
                    elif "m.1" in iter_var.name:
                        valid_iters['m.1'] = iter_var
                    else:
                        self.print_iter_vars()
                        raise NotImplementedError()
            # get tensor type
            if valid_iters['m.0.0'] is None and valid_iters['n'] is None and valid_iters['m.0.1'] is None and valid_iters['m.1'] is None:
                type = "scalar"
            elif valid_iters['m.0.0'] is None and valid_iters['m.0.1'] is None and valid_iters['m.1'] is None and valid_iters['n'] is not None:
                type = "row"
            elif valid_iters['m.0.0'] is not None and valid_iters['m.0.1'] is not None and valid_iters['m.1'] is not None and valid_iters['n'] is None:
                type = "column"
            elif valid_iters['m.0.0'] is not None and valid_iters['m.0.1'] is not None and valid_iters['m.1'] is not None and valid_iters['n'] is not None:
                type = "tensor"
            else:
                raise NotImplementedError()

            return type
        except NotImplementedError:
            # TODO: this is not always true
            return "tensor"
        

    def get_tensor_type(self):
        iter_vars = self.get_iter_vars()
        # get m & n

        valid_iters = {
            "b": None,
            "m": None,
            "n": None
        }
        try:
            for iter_var in iter_vars:
                if iter_var.extent > 1:
                    if "m" in iter_var.name:
                        valid_iters['m'] = iter_var
                    elif "n" in iter_var.name:
                        valid_iters['n'] = iter_var
                    elif "b" in iter_var.name:
                        valid_iters['b'] = iter_var
                    else:
                        self.print_iter_vars()
                        raise NotImplementedError()
            
            # get tensor type
            if valid_iters['b'] is None and valid_iters['m'] is None and valid_iters['n'] is None:
                type = "scalar"
            elif valid_iters['m'] is None and valid_iters['n'] is not None:
                type = "row"
            elif valid_iters['n'] is None and valid_iters['m'] is not None:
                type = "column"
            elif valid_iters['m'] is not None and valid_iters['n'] is not None:
                type = "tensor"
            else:
                raise NotImplementedError()

            return type
        except NotImplementedError:
            # TODO: this is not always true
            return "tensor"
    
    def get_iterator_factor_mod(self, source_iter_name, target_iter_name):
        try:
            target_iter = self.get_node_data(target_iter_name)
            # we need to first remove the order edges
            self.remove_order_edges()
            # get the path between 'b' and the itervar 'b*'
            # we interpret the factor in this way
            path = nx.shortest_path(self.graph, source_iter_name, target_iter_name)
            factor = 1
            for idx, node in enumerate(path):
                node_data = self.get_node_data(node)
                if isinstance(node_data, SplitNode):
                    shape = node_data.shape
                    # get the factor
                    chosen = path[idx + 1]
                    # get the successors of split node
                    successors = list(self.graph.successors(node))
                    successor_idx = successors.index(chosen)
                    for i in range(successor_idx+1, len(shape), 1):
                        factor *= shape[i]
                elif isinstance(node_data, IterVar):
                    continue
                else:
                    raise NotImplementedError()
            
            # get the modulo
            mod = target_iter.extent
        
            return 1./factor, mod
        except:
            return 1, 1


    def get_batch_factor_mod(self):
        iter_vars = self.get_iter_vars()
        # get m & n

        valid_iters = {
            "b": [],
            "m": [],
            "n": []
        }

        for iter_var in iter_vars:
            if iter_var.extent > 1:
                if "b" in iter_var.name:
                    valid_iters['b'].append(iter_var)
        # assert len(valid_iters['b']) <= 1
        try:
            valid_iters['b'] = valid_iters['b'][0]
        except:
            valid_iters['b'] = None

        # get batch factor and modulo
        if valid_iters['b'] is not None:
            # we need to first remove the order edges
            self.remove_order_edges()
            # get the path between 'b' and the itervar 'b*'
            # we interpret the factor in this way
            path = nx.shortest_path(self.graph, 'b', valid_iters['b'].name)
            factor = 1
            for idx, node in enumerate(path):
                node_data = self.get_node_data(node)
                if isinstance(node_data, SplitNode):
                    shape = node_data.shape
                    # get the factor
                    chosen = path[idx + 1]
                    # get the successors of split node
                    successors = list(self.graph.successors(node))
                    successor_idx = successors.index(chosen)
                    for i in range(successor_idx+1, len(shape), 1):
                        factor *= shape[i]
                elif isinstance(node_data, IterVar):
                    continue
                else:
                    raise NotImplementedError()
            
            # get the modulo
            mod = valid_iters['b'].extent
        
            return 1./factor, mod
        else:
            return None, None

    ## parser functions
    def get_node_tensor_bottom_up(self, node):
        shape = self.get_shape()
        new_graph = deepcopy(self)
        if node.target in [torch.ops.aten.view, torch.ops.aten._unsafe_view]:
            new_graph.view(node.args[1])
            node.meta["tensor"] = new_graph
        elif node.target in [torch.ops.aten.add, torch.ops.aten.sub, torch.ops.aten.mul, torch.ops.aten.div, torch.ops.aten.tanh_backward, torch.ops.aten.gelu_backward]:
            for arg in node.args:
                if isinstance(arg, fx.Node):
                    if not 'tensor' in arg.meta.keys():
                        new_arg_graph = deepcopy(self)
                        new_arg_graph.debroadcast(list(arg.meta["tensor_meta"].shape))
                        arg.meta['tensor'] = new_arg_graph
            node.meta["tensor"] = new_graph
        elif node.target in [torch.ops.aten.neg, torch.ops.aten.native_dropout, torch.ops.aten.clone, torch.ops.aten.gelu, torch.ops.aten.tanh, torch.ops.aten.relu, torch.ops.aten.sigmoid]:
            node.meta["tensor"] = new_graph
        elif node.target in [operator.getitem]:
            if len(node.meta["tensor_meta"].shape) != len(shape):
                raise NotImplementedError()
            for dim, node_dim in zip(shape, node.meta["tensor_meta"].shape):
                if dim != node_dim:
                    raise NotImplementedError()
            node.meta["tensor"] = new_graph
        elif node.target in [torch.ops.aten.permute]:
            new_graph.permute(node.args[1])
            node.meta["tensor"] = new_graph
        elif node.target in [torch.ops.aten.sum]:
            new_graph.reduce(node.args[1])
            node.meta["tensor"] = new_graph
        else:
            raise NotImplementedError("unsupported operator")
    
    def get_node_tensor_top_down(self, node):
        new_graph = deepcopy(self)
        # if the node has 'tensor' in meta data
        if node.target in [torch.ops.aten.add, torch.ops.aten.sub, torch.ops.aten.mul, torch.ops.aten.div, torch.ops.aten.tanh_backward, torch.ops.aten.gelu_backward]:
            for input in node.all_input_nodes:
                if 'tensor' in input.meta.keys(): continue
                new_arg_graph = deepcopy(self)
                new_arg_graph.debroadcast(list(input.meta['tensor_meta'].shape))
                input.meta['tensor'] = new_arg_graph
        elif node.target in [torch.ops.aten.view, torch.ops.aten._unsafe_view]:
            input = node.args[0]
            if 'tensor' in input.meta.keys(): return
            # new_graph.view(shape=list(input.meta['tensor_meta'].shape))
            input.meta['tensor'] = new_graph
        elif node.target in [torch.ops.aten._to_copy, torch.ops.aten.neg, torch.ops.aten.native_dropout, operator.getitem, torch.ops.aten.ne, torch.ops.aten.clone, torch.ops.aten.gelu, torch.ops.aten.tanh, torch.ops.aten.relu, torch.ops.aten.sigmoid]:
            input = node.args[0]
            if 'tensor' in input.meta.keys(): return
            input.meta['tensor'] = new_graph
        elif node.target in [torch.ops.aten.permute]:
            input = node.args[0]
            if 'tensor' in input.meta.keys(): return
            permute = node.args[1]
            new_graph.permute(list(np.argsort(permute)))
            input.meta['tensor'] = new_graph
        elif node.target in [torch.ops.aten.unsqueeze]:
            input = node.args[0]
            if 'tensor' in input.meta.keys(): return
            new_graph.squeeze(node.args[1])
            input.meta['tensor'] = new_graph
        elif node.target in [torch.ops.aten.one_hot]:
            input = node.args[0]
            if 'tensor' in input.meta.keys(): return
            new_graph.get_index(0)
            input.meta['tensor'] = new_graph
        elif node.target in [torch.ops.aten.mm, torch.ops.aten.bmm, torch.ops.aten._softmax, torch.ops.aten._softmax_backward_data, torch.ops.aten.native_layer_norm, torch.ops.aten.native_layer_norm_backward, torch.ops.aten.convolution, torch.nn.grad.conv2d_input, torch.nn.grad.conv2d_weight]:
            raise NotImplementedError()
        else:
            raise NotImplementedError()
        
        

