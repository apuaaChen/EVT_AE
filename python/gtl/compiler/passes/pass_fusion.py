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
"""
This file contains the partitining algorithm inspired by the paper 
"A branch-and-bound algorithm for the acyclic partitioning problem"
https://www.sciencedirect.com/science/article/pii/S0305054813002190
"""
import sqlite3
from cutlass import CACHE_FILE
import json
import networkx as nx
import torch
from torch.fx.graph_module import GraphModule
from torch.fx.node import Node
from torch.fx.graph import Graph
from torch.fx.passes.utils.fuser_utils import validate_partition
import operator
import pydot
from typing import Any, Optional, Union
import math
import numpy as np
import scipy
from tqdm import tqdm
import seaborn as sns
import matplotlib.pyplot as plt
from concurrent.futures import ThreadPoolExecutor
from torch.fx.passes.infra.pass_base import PassBase, PassResult
from torch.fx.passes.shape_prop import TensorMetadata
from torch.fx.passes.tools_common import legalize_graph
from gtl.compiler.passes.pass_constant_propagation import PermuteAbv

from gtl.compiler.passes.evt_fuser import EVTFuser


# The operators fusible
FUSIBLE_TGT = [
    torch.ops.aten.clone,
    torch.ops.aten.view,
    torch.ops.aten.add,
    torch.ops.aten.permute,
    torch.ops.aten.div,
    torch.ops.aten.native_dropout,
    operator.getitem,
    torch.ops.aten.unsqueeze,
    torch.ops.aten.squeeze,
    torch.ops.aten.add,
    torch.ops.aten.rsub,
    torch.ops.aten.mul,
    torch.ops.aten.gelu,
    torch.ops.aten.tanh,
    torch.ops.aten.tanh_backward,
    torch.ops.aten.sum,
    torch.ops.aten.gelu_backward,
    torch.ops.aten.native_dropout_backward,
    torch.ops.aten.ne,
    torch.ops.aten.one_hot,
    torch.rand_like,
    torch.ops.aten.ge
]

################################################################################
# Artifact Manager
################################################################################
class ArtifactManager:
    """
    This class manages the cached partitioning results solved by previous ilp solver
    """
    def __init__(self) -> None:
        connection = sqlite3.connect(CACHE_FILE)
        cursor = connection.cursor()
        # Create the table if it does not exist
        sqlite_create_partition_table_query = """
        CREATE TABLE IF NOT EXISTS partitions(hash TEXT NOT NULL,
                                              p_idx INTEGER NOT NULL,
                                              n_idx INTEGER NOT NULL,
                                              node TEXT,
                                              PRIMARY KEY (hash, p_idx, n_idx))
        """
        sqlite_create_graph_table_query = """
        CREATE TABLE IF NOT EXISTS subgraphs(hash TEXT NOT NULL UNIQUE PRIMARY KEY,
                                             subgraph TEXT NOT NULL)
        """
        cursor.execute(sqlite_create_partition_table_query)
        cursor.execute(sqlite_create_graph_table_query)
        connection.commit()
        cursor.close()
    
    def insert_partition(self, hash, partitions, graph):
        connection = sqlite3.connect(CACHE_FILE)
        cursor = connection.cursor()
        sqlite_insert_partition_query = """ INSERT OR IGNORE INTO partitions (hash, p_idx, n_idx, node) VALUES (?, ?, ?, ?)"""
        for p_idx, partition in enumerate(partitions):
            for n_idx, node in enumerate(partition):
                data_tuple = (hash, p_idx, n_idx, node.name)
                cursor.execute(sqlite_insert_partition_query, data_tuple)
        
        sqlite_insert_subgraph_query = """ INSERT OR IGNORE INTO subgraphs (hash, subgraph) VALUES (?, ?)"""
        for node in graph.nodes:
            del graph.nodes[node]["meta"]
        graph_data_json = json.dumps(nx.node_link_data(graph,))
        data_tuple = (hash, graph_data_json)
        cursor.execute(sqlite_insert_subgraph_query, data_tuple)
        connection.commit()
        cursor.close()
    
    def fetch_partition(self, hash):
        connection = sqlite3.connect(CACHE_FILE)
        cursor = connection.cursor()
        sqlite_fetch_partition_query = """SELECT * from partitions where hash = ?"""
        cursor.execute(sqlite_fetch_partition_query, (hash,))
        record = cursor.fetchall()
        if len(record) == 0:
            return None
        partitions = []
        for row in record:
            _, _, n_idx, node = row
            if n_idx == 0:
                partitions.append([node,])
            else:
                partitions[-1].append(node)
        
        sqlite_fetch_graph_query = """SELECT * from subgraphs where hash = ?"""
        cursor.execute(sqlite_fetch_graph_query, (hash,))
        record = cursor.fetchall()
        assert len(record) == 1
        _, graph_data_json = record[0]
        graph_data = json.loads(graph_data_json)
        subgraph = nx.node_link_graph(graph_data)
        cursor.close()

        return (partitions, subgraph)

################################################################################
# SubGraph Drawer
################################################################################
class SubGraphDrawer:
    """
    Visualize the apartition graph for debug
    """
    def __init__(
            self, graph: nx.DiGraph, 
            name: str,
            node_list: list
            ) -> None:
        self._name = name
        self.node_list = node_list
        self._dot_graphs = {}
        self._dot_graphs[name] = self._to_dot(graph, name)
    
    def _get_node_style(self, node):
        template = {
            "shape": "record",
            "fillcolor": "#CAFFE3",
            "style": '"filled,rounded"',
            "fontcolor": "#000000",
        }

        if node not in self.node_list:
            template["fillcolor"] = "white"
            template["fontcolor"] = "grey"
        else:
            template["fillcolor"] = "PowderBlue"
        return template
    
    def _get_node_label(self, node):
        label = "{" + f"name={node}" + "}"
        return label
    
    def _to_dot(
        self,
        graph: nx.DiGraph,
        name: str
    ):
        dot_graph = pydot.Dot(name, randir="TB")
        for node in graph.nodes:
            style = self._get_node_style(node)
            label = self._get_node_label(node)
            dot_node = pydot.Node(
                node, label=label, **style
            )
            dot_graph.add_node(dot_node)

        # Add edges
        for src, dst in graph.edges:
            dot_graph.add_edge(pydot.Edge(src, dst))

        return dot_graph


################################################################################
# ILP Solver
################################################################################
class ILPSolver:
    def __init__(self, component: 'list[str]', 
                 graph: nx.DiGraph, node_topo: 'list[str]',
                 cache: dict) -> None:
        """
        Args:
            component: the 
        """
        self.cache = ArtifactManager()
        self.solution_cache = cache
        self.graph = graph
        # Create the undirected copy of the compute graph
        self.graph_undirected = nx.Graph()
        self.graph_undirected.add_nodes_from(self.graph.nodes)
        self.graph_undirected.add_edges_from(self.graph.edges)
        self.node_topo = node_topo

        # Step 1: get the subgraph to work on
        self.get_subgraph(component)
    
    def __call__(self):
        hash = self.hash()
        cached_result = self.cache.fetch_partition(hash)
        if cached_result is not None:
            return self.fetch_cached_solution(cached_result)
        # if hash in self.solution_cache:
        #     return self.fetch_cached_solution(hash)
        else:
            return self.solve(hash)

    #
    # Helper functions
    #

    def is_fusible(self, node: Union[str, Node]):
        """
        Helper function checking if the given node is a fusible operator
        """
        if isinstance(node, str):
            node = self.graph.nodes[node]["meta"]
        if node.op == "placeholder":
            raise RuntimeError("Placeholder should not be included in IR")
        elif node.op == "call_function":
            # Get non-suffix name
            if node.target in FUSIBLE_TGT:
                return True
            else:
                return False

    def get_node_meta(self, node: str) -> Node:
        """
        Get the metadata of the node
        """
        return self.graph.nodes[node]["meta"]
    
    def topo_idx(self, node: str):
        """
        Get the index of a node in the graph under the topological order
        """
        return self.node_topo.index(node)

    def get_volumn(self, src: str, dst: str):
        """
        Get the volumn of an edge. Used to assign the edge weight
        """
        if "tensor_meta" not in self.get_node_meta(src).meta.keys():
            return None
        src_metadata = self.get_node_meta(src).meta["tensor_meta"]
        if not isinstance(src_metadata, TensorMetadata):
            dst_node = self.get_node_meta(dst)
            assert dst_node.target == operator.getitem
            src_metadata = src_metadata[dst_node.args[1]]
        
        shape = src_metadata.shape
        volumn = 1
        for d in shape:
            volumn *= d
        return volumn
    
    def get_ancestors_and_descendants(self, component: 'list[str]'):
        """
        Get the one-hop ancestors and descendants of a node
        """
        node_set = set(component)
        ancestor_set = set()
        descendant_set = set()
        for node in node_set:
            for ancesor, _ in self.graph.in_edges(node):
                ancestor_set.add(ancesor)
            for _, descendant in self.graph.out_edges(node):
                descendant_set.add(descendant)
        ancestor_set = ancestor_set.difference(node_set)
        descendant_set = descendant_set.difference(node_set)
        neighbour_set = ancestor_set.union(descendant_set)

        return node_set, ancestor_set, descendant_set, neighbour_set
    
    def get_subgraph(self, component: 'list[str]'):
        """
        Get the subgraph to work on
        """
        node_set, ancestor_set, descendant_set, neighbour_set = \
            self.get_ancestors_and_descendants(component)
        
        nodes_to_include = node_set.union(node_set, neighbour_set)
        self.subgraph: nx.DiGraph = self.graph.subgraph(nodes_to_include).copy()

        ########################################################################
        # Step 1: Connect the one-hop neighbors if there is a path between them
        # Step 1.1: create the connectivity graph
        connectivity_graph = self.graph_undirected.copy()
        connectivity_graph.remove_nodes_from(node_set)

        # Step 1.2: add edges between neighbors
        for n1 in neighbour_set:
            n1_idx = self.topo_idx(n1)
            for n2 in neighbour_set:
                n2_idx = self.topo_idx(n2)
                if n2_idx > n1_idx:
                    if nx.has_path(connectivity_graph, n1, n2):
                        self.subgraph.add_edge(n1, n2)
        
        ########################################################################
        # Step 2: Label node and edges
        for node in self.subgraph.nodes:
            if node in node_set:
                self.subgraph.nodes[node]["label"] = str(
                    self.subgraph.nodes[node]["meta"].target)
            else:
                self.subgraph.nodes[node]["label"] = "neighbor"
        edge_weights = set()
        for src, tgt in self.subgraph.edges:
            if src in node_set and tgt in node_set:
                volume = self.get_volumn(src, tgt)
                self.subgraph[src][tgt]["weight"] = volume
                edge_weights.add(volume)
            else:
                self.subgraph[src][tgt]["label"] = "0"
        mcf = math.gcd(*edge_weights)
        if mcf == 0:
            mcf = 1  # Special case when set is empty

        for src, tgt in self.subgraph.edges:
            if src in node_set and tgt in node_set:
                self.subgraph[src][tgt]["label"] = str(
                    self.subgraph[src][tgt]["weight"] / mcf)
        
        # Register node lists
        self.node_list = list(node_set)
        self.ancestor_list = list(ancestor_set)
        self.descendant_list = list(descendant_set)

        self.all_node_list = self.node_list + self.ancestor_list + self.descendant_list

        # Initialize counters
        self.nc = len(self.node_list)
        self.na = len(self.ancestor_list)
        self.nd = len(self.descendant_list)
        self.n = self.subgraph.number_of_nodes()

        self.num_z = int(self.nc * self.nc * (self.nc - 1) / 2)
        self.num_x = self.nc * self.nc
        self.num_y = self.nc * self.nc + self.nc * (self.na + self.nd)
        self.num_p = self.n

        self.num_all = self.num_z + self.num_x + self.num_y + self.num_p
    
    #
    # Cache helper functions
    #

    def hash(self):
        """
        Compute the graph isomorphic hash value
        """
        assert hasattr(self, "subgraph")
        return nx.weisfeiler_lehman_graph_hash(
            self.subgraph, edge_attr="label", node_attr="label")

    def fetch_cached_solution(self, result):
        """
        Fetch the previously cached result
        """
        cached_partitions, cached_subgraph = result
        nm = nx.algorithms.isomorphism.categorical_node_match("label", "neighbor")
        em = nx.algorithms.isomorphism.categorical_edge_match("label", "0")
        digm = nx.algorithms.isomorphism.DiGraphMatcher(
            cached_subgraph, self.subgraph, node_match=nm, edge_match=em)
        assert digm.is_isomorphic()
        partitions = []
        for cached_parition in cached_partitions:
            partition = []
            for node in cached_parition:
                partition.append(
                    self.graph.nodes[digm.mapping[node]]["meta"])
            if validate_partition(partition):
                partitions.append(partition)
            else:
                breakpoint()
                raise RuntimeError(f"Invalid partition: {partition}")
        return partitions
    
    #
    # Convenient functions calculating the index to z, x, y, and pi
    #

    def z(self, i, j, s):
        assert i < self.nc and i >= 0
        assert j < self.nc and j >= 0
        assert s < self.nc and s >= 0
        assert i < j
        return int(
            (self.nc * (self.nc - 1) / 2) * s + 
            (2 * self.nc - i - 1) * i / 2 + 
            j-i-1)
    
    def x(self, i, s):
        assert i < self.nc and i >= 0
        assert s < self.nc and s >= 0
        return self.num_z + s * self.nc + i
    
    def y(self, s, t):
        assert s < self.n
        assert t < self.n
        if s < self.nc and t < self.nc:
            return self.num_z + self.num_x + s * (self.nc + self.nd) + t
        elif s < self.nc and t >= self.nc + self.na:
            return self.num_z + self.num_x + s  * (self.nc + self.nd) + t - self.na
        elif s >= self.nc and s < self.nc + self.na and t < self.nc:
            return (self.num_z + self.num_x + self.nc * (self.nc + self.nd) 
                    + (s - self.nc) * self.nc + t)
        else:
            return None
    
    def pi(self, s):
        assert s < self.n and s >= 0
        return self.num_z + self.num_x + self.num_y + s
    
    def node_idx(self, node):
        return self.all_node_list.index(node)
    
    #
    # Get the optimization objective function
    #

    def get_objective_fn(self):
        """
        Get the objective function of the ILP
        """
        edge_weights = []
        for i in range(self.nc):
            for j in range(i+1, self.nc):
                node_i = self.node_list[i]
                node_j = self.node_list[j]
                weight = 0
                if self.subgraph.has_edge(node_i, node_j):
                    weight += self.subgraph[node_i][node_j]["weight"]
                if self.subgraph.has_edge(node_j, node_i):
                    weight += self.subgraph[node_j][node_i]["weight"]
                edge_weights.append(weight)
        c = -np.concatenate([
            np.array(edge_weights * self.nc), 
            np.zeros(shape=self.num_y+self.num_x + self.num_p)], axis=0)
        return c

    #
    # Constraints
    #
    def constraint_sum_x(self):
        """
        Constraint (2.2): \sum_s x_si = 1, for all 1 < i < nc
        """
        A = np.zeros(shape=(self.nc, self.num_all))
        for i in range(self.nc):
            for s in range(self.nc):
                A[i][self.x(i, s)] = 1
        b = np.ones(shape=A.shape[0])
        return scipy.optimize.LinearConstraint(A, lb=b, ub=b)

    def constraint_z_sij_le_x_si(self):
        """
        Constraint (2.4): z_sij <= x_si, forall 1 <= i < j < n0, 1 <=s<= n0
        """
        A = np.zeros(shape=(self.num_z, self.num_all))
        cnt = 0
        ij_cnt = 0
        for i in range(self.nc):
            for j in range(i+1, self.nc):
                for s in range(self.nc):
                    A[cnt][s * int(self.num_z / self.nc) + ij_cnt] = 1
                    A[cnt][self.x(i, s)] = -1
                    cnt += 1
                ij_cnt += 1
        ub = np.zeros(shape=A.shape[0])
        return scipy.optimize.LinearConstraint(A, ub=ub)
    
    def constraint_z_sij_le_x_sj(self):
        """
        Constraint (2.5): z_sij <= x_sj, forall 1 <= i < j < n0, 1 <=s<= n0
        """
        A = np.zeros(shape=(self.num_z, self.num_all))
        cnt = 0
        ij_cnt = 0
        for i in range(self.nc):
            for j in range(i+1, self.nc):
                for s in range(self.nc):
                    A[cnt][s * int(self.num_z / self.nc) + ij_cnt] = 1
                    A[cnt][self.x(j, s)] = -1
                    cnt += 1
                ij_cnt += 1
        ub = np.zeros(shape=A.shape[0])
        return scipy.optimize.LinearConstraint(A, ub=ub)

    def constraint_x_y(self):
        """
        Constraint (2.6): x_si + x_tj - 1 <= y_st, forall vi,vj in A, 1 <=s !=t <= n
        """
        A = []
        ub = []
        # i < nc, j < nc
        for src, dst in self.subgraph.edges():
            i = self.node_idx(src)
            j = self.node_idx(dst)
            if src in self.node_list and dst in self.node_list:
                A_tmp = np.zeros(shape=(self.nc*(self.nc-1), self.num_all))
                cnt = 0
                # for s, t >= nc, x_si = x_tj = 0
                for s in range(self.nc):
                    for t in range(self.nc):
                        if s != t:
                            A_tmp[cnt][self.x(i, s)] = 1
                            A_tmp[cnt][self.x(j, t)] = 1
                            A_tmp[cnt][self.y(s, t)] = -1
                            cnt += 1
                A.append(A_tmp)
                ub.append(np.ones(shape=A_tmp.shape[0]))
            # i < nc, nc+na<=j < nc+na+nd
            elif src in self.node_list and dst in self.descendant_list:
                # For x_jt = 1 when j=t, otherwise 0
                # So we only consider t = j
                A_tmp = np.zeros(shape=(self.nc, self.num_all))
                # for s > nc, x_si = 0
                # for t != j, x_tj = 0
                t = j
                for s in range(self.nc):
                    A_tmp[s][self.x(i, s)] = 1
                    # x_tj = 0
                    A_tmp[s][self.y(s, t)] = -1
                A.append(A_tmp)
                ub.append(np.zeros(shape=(A_tmp.shape[0],)))
            # nc<=i < nc + na, j < nc
            elif src in self.ancestor_list and dst in self.node_list:
                # For x_is = 1 when i=s, otherwise 0
                # So we only consider s = i
                A_tmp = np.zeros(shape=(self.nc, self.num_all))
                s = i
                for t in range(self.nc):
                    A_tmp[t][self.x(j, t)] = 1
                    A_tmp[t][self.y(s, t)] = -1
                A.append(A_tmp)
                ub.append(np.zeros(shape=(A_tmp.shape[0],)))

        if len(A) > 0:
            A = np.concatenate(A, axis=0)
            ub = np.concatenate(ub, axis=0)
            return scipy.optimize.LinearConstraint(A, ub=ub)
        else:
            return None

    def constraint_py(self):
        """
        Constraint (2.7): pi_s - pi_t + n*y_st <= n-1, forall 1 <= s != t <= n
        """
        A = np.zeros(shape=(self.n * (self.n-1), self.num_all))
        ub = np.full(shape=A.shape[0], fill_value=self.n-1)
        cnt = 0
        for s in range(self.n):
            for t in range(self.n):
                if s == t: continue
                A[cnt][self.pi(s)] = 1
                A[cnt][self.pi(t)] = -1
                y_idx = self.y(s, t)
                if y_idx is not None:
                    A[cnt][y_idx] = self.n
                # Other cases
                elif s >= self.nc and t >= self.nc:
                    node_s = self.all_node_list[s]
                    node_t = self.all_node_list[t]

                    if self.subgraph.has_edge(node_s, node_t):
                        # If has edge s->t, then y_st = 1
                        ub[cnt] = -1
                cnt += 1
        
        return scipy.optimize.LinearConstraint(A, ub=ub)
    
    def constraint_x_anti_sym(self):
        """
        Constraint (2.8): \sum_i x_si <= \sum_i x_s-1 i
        """
        A = np.zeros(shape=(self.nc-1, self.num_all))
        cnt = 0
        for s in range(1, self.nc):
            for i in range(self.nc):
                A[cnt][self.x(i, s)] = 1
                A[cnt][self.x(i, s-1)] = -1
            cnt += 1
        
        ub = np.zeros(shape=A.shape[0])
        return scipy.optimize.LinearConstraint(A, ub=ub)
    
    def constraint_nucleis_heuristics(self):
        """
        Constraint (2.13): nuclei heuristic
        Because each partition in the epilogue fusion can only contain a single
        unfusable node, so we use them as the condensation nuclei to reduce
        search space and eliminate symmetry
        """
        nucleis = []
        for node in self.node_list:
            if not self.is_fusible(node):
                nucleis.append(node)
        
        A = np.zeros(shape=(len(nucleis), self.num_all))
        for s, node in enumerate(nucleis):
            i = self.node_list.index(node)
            # x_si = 1
            A[s][self.x(i,s)] = 1
        
        b = np.ones(shape=A.shape[0])
        return scipy.optimize.LinearConstraint(A, ub=b, lb=b)
    
    def constraint_view_heuristics(self):
        """
        The view nodes are always fused with their preceeding nodes
        """
        A = []
        for node in self.node_list:
            if "view" not in node and "unsqueeze" not in node:#and "getitem" not in node:
                continue
            in_edges = list(self.subgraph.in_edges(node))
            assert len(in_edges) <= 1
            for src, _ in self.subgraph.in_edges(node):
                if src not in self.node_list:
                    continue
                # Special case for sum
                if "sum" in src:
                    continue
                i = self.node_list.index(src)
                j = self.node_list.index(node)
                At = np.zeros(shape=(1, self.num_all))
                for s in range(self.nc):
                    At[0][self.z(min(i,j), max(i,j), s)] = 1
                A.append(At)
        if len(A) > 0:
            A = np.concatenate(A, axis=0)
            b = np.ones(shape=A.shape[0])
            return scipy.optimize.LinearConstraint(A, ub=b, lb=b)
        return None

    def constraint_eliminate_mainloop_fusion(self):
        """
        This add additional constraint that prohibits mainloop fusions
        while enforce t/permute->mm/bmm fusion
        """
        A = []
        ub = []
        lb = []
        for node in self.node_list:
            # For unfusible nodes
            if not self.is_fusible(node):
                in_edges = list(self.subgraph.in_edges(node))
                node_tgt = self.get_node_meta(node).target
                for input, _ in in_edges:
                    if input not in self.node_list:
                        continue
                    i = self.node_list.index(input)
                    j = self.node_list.index(node)
                    input_tgt = self.get_node_meta(input).target
                    # Special case for t/permute->mm/bmm
                    if node_tgt in [torch.ops.aten.mm, torch.ops.aten.bmm, torch.ops.aten.addmm] and input_tgt in [torch.ops.aten.t, torch.ops.aten.transpose, torch.ops.aten.permute]:
                        At = np.zeros(shape=(1, self.num_all))
                        for s in range(self.nc):
                            At[0][self.z(min(i,j), max(i,j), s)] = 1
                        A.append(At)
                        ub.append(np.ones(shape=At.shape[0]))
                        lb.append(np.ones(shape=At.shape[0]))
                        # Find the input of i
                        in_edges_input = list(self.subgraph.in_edges(input))
                        assert len(in_edges_input) <= 1
                        for grand_input, _ in in_edges_input:
                            if grand_input not in self.node_list:
                                continue
                            k = self.node_list.index(grand_input)
                            At = np.zeros(shape=(self.nc, self.num_all))
                            for s in range(self.nc):
                                At[s][self.x(i, s)] = 1
                                At[s][self.x(k, s)] = 1
                            A.append(At)
                            ub.append(np.ones(shape=At.shape[0]))
                            lb.append(np.full(shape=At.shape[0], fill_value=-np.inf))
                    else:
                        At = np.zeros(shape=(self.nc, self.num_all))
                        for s in range(self.nc):
                            At[s][self.x(i, s)] = 1
                            At[s][self.x(j, s)] = 1
                        A.append(At)
                        ub.append(np.ones(shape=At.shape[0]))
                        lb.append(np.full(shape=At.shape[0], fill_value=-np.inf))
        if len(A) > 0:
            A = np.concatenate(A, axis=0)
            ub = np.concatenate(ub, axis=0)
            lb = np.concatenate(lb, axis=0)

            return scipy.optimize.LinearConstraint(A, ub=ub, lb=lb)
        return None
    
    def constraint_sum_is_leaf(self):
        """
        The sum nodes cannot be fused with its children
        """
        A = []
        for node in self.node_list:
            if "sum" in node:
                out_edges = list(self.subgraph.out_edges(node))
                for _, output in out_edges:
                    if output not in self.node_list:
                        continue
                    i = self.node_list.index(node)
                    j = self.node_list.index(output)

                    At = np.zeros(shape=(self.nc, self.num_all))
                    for s in range(self.nc):
                        At[s][self.x(i, s)] = 1
                        At[s][self.x(j, s)] = 1
                    A.append(At)

        if len(A) > 0:
            A = np.concatenate(A, axis=0)
            b = np.ones(shape=A.shape[0])

            return scipy.optimize.LinearConstraint(A, ub=b)
        return None
    
    #
    # Integrality and bound
    #

    def get_integrality(self):
        return np.ones(shape=self.num_all)
    
    def get_bound(self):
        lb = np.concatenate([
            np.zeros(shape=self.num_z + self.num_x + self.num_y),
            np.full(shape=self.num_p, fill_value=-np.inf)
        ], axis=0)

        ub = np.concatenate([
            np.ones(shape=self.num_z + self.num_x + self.num_y),
            np.full(shape=self.num_p, fill_value=np.inf)
        ], axis=0)

        return scipy.optimize.Bounds(lb=lb, ub=ub)

    #
    # Solver
    #

    def replace_all_uses_with(self, node1, node2):
        """
        Replace all uses of node1 with node2
        """
        edges_to_remove = []
        for _, user in self.subgraph.out_edges(node1):
            self.subgraph.add_edge(node2, user)
            edges_to_remove.append((node1, user))
        self.subgraph.remove_edges_from(edges_to_remove)
        self.subgraph.remove_node(node1)

    def solve(self, hash):        
        c = self.get_objective_fn()

        constraints = [
            getattr(self, builder)() for builder in dir(self) 
            if callable(getattr(self, builder)) 
            and builder.startswith("constraint")]
        
        constraints = [c for c in constraints if c is not None]

        integrality = self.get_integrality()
        bounds = self.get_bound()
        
        res = scipy.optimize.milp(
            c=c, constraints=constraints, bounds=bounds, 
            integrality=integrality, options={"disp":True, "time_limit": 500})
        if res.x is None:
            drawer = SubGraphDrawer(self.subgraph, "error", self.node_list)
            graph = drawer._dot_graphs["error"]
            graph.write_svg(f"./error.svg")
            breakpoint()
            return
        result = res.x[self.num_z: self.num_z + self.num_x]

        nodes = [self.graph.nodes[node]["meta"] for node in self.node_list]

        partitions = []
        for s in range(self.nc):
            partition = []
            for i in range(self.nc):
                if result[s * self.nc + i] > 0.5:
                    partition.append(nodes[i])
            if len(partition) > 0:
                # Verify the partition
                if validate_partition(partition):
                    partitions.append(partition)
                else:
                    breakpoint()
                    raise RuntimeError(f"Invalid partition: {partition}")
        
        # self.solution_cache[hash] = (partitions, self.subgraph)
        self.cache.insert_partition(hash, partitions, self.subgraph)
        return partitions


################################################################################
# Compute Graph IR
################################################################################
class ComputeGraphIR:
    def __init__(self, graph: Graph) -> None:
        self._graph = nx.DiGraph()
        # We use the _graph_undirected to track the binary fusible relationship
        # Between nodes
        self.partition_graph = nx.Graph()

        for node in graph.nodes:
            # Insert the node
            if node.op == "call_function":
                self.add_node(node, self._graph)
                self.add_node(node, self.partition_graph)
                node_fusible = self.is_fusible(node)
                # Insert the edge:
                for input in node.all_input_nodes:
                    if input.op == "call_function":
                        self.add_edge(input, node, self._graph)
                        if node_fusible:
                            # We cut the path -> unfusible nodes
                            self.add_edge(input, node, self.partition_graph)
                        else:
                            # special cases for mainloop fusion
                            # Although most unfusible nodes only support epilogue
                            # fusion, there are some have limited mainloop fusion
                            # capability. One special case we have here is the
                            # gemm/convolution kernels support mainloop fusion
                            # with transposes and permutations
                            if input.target in [torch.ops.aten.permute]:
                                if node.target in [torch.ops.aten.mm, torch.ops.aten.bmm, torch.ops.aten.addmm]:
                                    self.add_edge(input, node, self.partition_graph)
                                    # Also cut the edge between input and its parent
                                    parent_of_inputs = input.all_input_nodes
                                    assert len(input.users) == 1
                                    assert len(parent_of_inputs) == 1
                                    if self.partition_graph.has_edge(
                                        parent_of_inputs[0].name, input.name):
                                        self.partition_graph.remove_edge(
                                            parent_of_inputs[0].name, input.name
                                        )
        
        self.node_topo = list(nx.topological_sort(self._graph))
        self.solution_cache = {}
    #
    # IR manipulator
    #
    def add_node(self, node: Node, graph: nx.Graph):
        """
        Add a node to dag ir
        """
        if self.has_node(node, graph):
            raise SyntaxError(f"Variable '{node.name}' cannot be defined twice.")
        graph.add_node(node.name, meta=node)
    
    def get_node_meta(self, node: str) -> Node:
        """
        Get the metadata of the node
        """
        return self._graph.nodes[node]["meta"]
    
    def add_edge(self, src: Node, dst: Node, graph: nx.Graph):
        if not self.has_node(src, graph):
            raise SyntaxError(f"Variable '{src.name}' is undefined.")
        if not self.has_node(dst, graph):
            raise SyntaxError(f"Variable '{dst.name}' is undefined.")
        graph.add_edge(src.name, dst.name)
    
    #
    # Helper functions for getting attrs
    #
    def has_node(self, node: Node, graph: nx.Graph) -> bool:
        """
        Check if the node is in the graph
        """
        return graph.has_node(node.name)
    
    #
    # High-level APIs
    #
    def shortest_length_matrix(self):
        self.length_dict_directed = dict(nx.all_pairs_shortest_path_length(self._graph))

    def is_fusible(self, node: Union[str, Node]):
        if isinstance(node, str):
            node = self._graph.nodes[node]["meta"]
        if node.op == "placeholder":
            raise RuntimeError("Placeholder should not be included in IR")
        elif node.op == "call_function":
            # Get non-suffix name
            if node.target in FUSIBLE_TGT:
                return True
            else:
                return False


    def initialize_same_partition_matrix(self):
        if not hasattr(self, "length_dict"):
            self.shortest_length_matrix()
        
        nodes = list(self._graph.nodes())

        # TODO: This greatly reduces the fusible space
        # Let's carefully exam this and also the succeeding conditions
        
        # Remove edges based on constraints
        # Acyclic Constraint: 
        #   If node 2 is unfusable, and node1 can reach
        #   node2, then node2 and all nodes reachable from node2 are not
        #   in the same partition with node 1
        # This constraint helps cut the connection between forward and backward
        # graph
        for node1 in tqdm(nodes):
            unfusible_nodes = set()
            for node2 in self.length_dict_directed[node1].keys():
                # Reduce the complexity
                if node2 in unfusible_nodes:
                    continue
                if not self.is_fusible(node2) and node2 != node1:
                    # Special case for t/permute->mm
                    node1_tgt = self.get_node_meta(node1).target
                    node2_tgt = self.get_node_meta(node2).target
                    if node1_tgt == torch.ops.aten.permute and node2_tgt in [torch.ops.aten.mm, torch.ops.aten.bmm, torch.ops.aten.addmm]:
                        continue
                    unfusible_nodes = unfusible_nodes.union(
                        self.length_dict_directed[node2].keys())
            
            # Remove the edges
            node1_neighbors = nx.all_neighbors(self.partition_graph, node1)
            unfusible_nodes = unfusible_nodes.intersection(node1_neighbors)
            for node3 in unfusible_nodes:
                self.partition_graph.remove_edge(node1, node3)

        connected_components = list(nx.connected_components(self.partition_graph))

        # Support multi-thread solving. Although it doesn't bring much benefits
        # results = []
        # for component in tqdm(connected_components):
        #     solver = ILPSolver(component, self._graph, self.node_topo, self.solution_cache)
        #     results.append(solver())
        with ThreadPoolExecutor(1) as executor:
            with tqdm(total=len(connected_components)) as pbar:
                def solve_component(component):
                    solver = ILPSolver(component, self._graph, self.node_topo, self.solution_cache)
                    result = solver()
                    pbar.update(1)
                    return result
                results = list(executor.map(solve_component, connected_components))
        
        return results
    
    def visualize_adj_matrix(self, matrix: np.matrix, name):
        sns.set()
        plt.figure(figsize=(8, 6))
        heatmap = sns.heatmap(matrix, annot=False)
        plt.title("name")
        heatmap.get_figure().savefig(f"./{name}.png")
        plt.close()


################################################################################
# Pass 1: duplication before partition
################################################################################

class DuplicationBeforePartition(PassBase):

    def call(self, graph_module: GraphModule) -> PassResult | None:
        graph: Graph = graph_module.graph
        for node in graph_module.graph.nodes:
            # This pass assumes that all the transposes are unified to permute
            if node.target == torch.ops.aten.permute:
                users = list(node.users)
                heavy_users = []
                for user in users:
                    if user.target in [torch.ops.aten.mm, torch.ops.aten.bmm, torch.ops.aten.addmm]:
                        heavy_users.append(user)
                if len(users) > 1 and len(heavy_users) > 0:
                    for user in heavy_users:
                        graph.inserting_before(user)
                        duplicated_node = graph.call_function(node.target,
                                                            args=node.args)
                        duplicated_node.meta = {}
                        duplicated_node.meta["tensor_meta"] = node.meta["tensor_meta"]._replace()
                        user.replace_input_with(node, duplicated_node)
    
    def ensures(self, graph_module: GraphModule) -> None:
        graph_module.graph.lint()

################################################################################
# Pass 2: partitioning pass
################################################################################
class Partitioning(PassBase):
    def requires(self, graph_module: GraphModule) -> None:
        duplicate_pass = DuplicationBeforePartition()
        duplicate_pass(graph_module)

    def call(self, graph_module: GraphModule) -> PassResult | None:
        compute_graph_ir = ComputeGraphIR(graph_module.graph)
        compute_graph_ir.shortest_length_matrix()
        partitions = compute_graph_ir.initialize_same_partition_matrix()

        for idx, partition in enumerate(partitions):
            print(f"==============={idx}==================")
            print(partition)
            if idx >= 26: break
            if idx in [2,16,18,19,24]: 
                print("skipped")
                continue
            for par in partition:
                if len(par) > 1:
                    fuser = EVTFuser()
                    fuser.trace(graph_module, par)

        return super().call(graph_module)

    def enforce_contiguous_format(self, graph: Graph):
        """
        Insert "clone" nodes to incomming edges of fused nodes if they are not
        contiguous.

        Intuition: The fused kernels assume that all the incoming tensors are
        under contiguous layout. This function enforces this assumption.
        """
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
        
        # Insert the clone nodes
        for node in graph.nodes:
            if "evt" in node.meta:
                for input in node.all_input_nodes:
                    abv = self.workspace[input]
                    if not abv.is_contiguous():
                        # Insert the clone node
                        graph.inserting_after(input)
                        clone_node = graph.call_function(
                            torch.ops.aten.clone, args=(input,), 
                            kwargs={"memory_format": torch.contiguous_format})
                        
                        clone_node.meta = {}
                        clone_node.meta["tensor_meta"] = input.meta["tensor_meta"]._replace()
                        node.replace_input_with(input, clone_node)

    def ensures(self, graph_module: GraphModule) -> None:
        # Inject clone nodes
        self.enforce_contiguous_format(graph_module.graph)

        # Cleanup
        legalize_graph(graph_module)
        graph_module.graph.eliminate_dead_code()
        graph_module.recompile()


def pass_fusion(module, graph):
    partitioner = Partitioning()
    partitioner(module)
