import torch
from torch.fx.graph_module import GraphModule
from torch.fx.graph import Graph
from torch.fx.node import Node
from torch.fx._symbolic_trace import symbolic_trace
from typing import Union, Callable, List, Dict, Set
from torch.fx.subgraph_rewriter import Match, ReplacedPatterns, _replace_submodules
from torch.fx.passes.utils.matcher_utils import SubgraphMatcher, InternalMatch, logger
from collections import defaultdict
from torch.fx.passes.tools_common import NodeList, NodeSet
import copy

def validate_partition(partition: NodeList) -> bool:
    # verify the partition does't form a dependency cycle in the original graph
    # returns True for valid partition, False for invalid

    partition_set = set(partition)

    outputs: NodeList = list()
    for node in partition_set:
        for user_node in node.users:
            if user_node not in partition_set:
                # external user node, need to expose as an output
                outputs.append(user_node)

    # Perform BFS on the partition outputs.
    # If it reaches a node within the partition, then it found a cycle.
    # This function takes the ownership of `root_nodes` and may modify it.
    def bfs_find_cycle(root_nodes: NodeList) -> bool:
        # Set used to exclude nodes that have already been visited.
        # If a node has been visited, that node and all its children have
        # been checked for cycles.
        visited: NodeSet = set()

        # Start with `root_nodes` and traverse through (toward child nodes)
        # their connected sub-graph. Nodes in `visited` won't be added
        # to `queue` again.
        queue: NodeList = root_nodes
        while queue:
            current = queue.pop()
            visited.add(current)
            if current in partition_set:
                # Started from partition's `output` nodes, and reached
                # another node in partition. Cycle!
                return True
            for user_node in current.users:
                if user_node in visited:
                    continue
                queue.append(user_node)
        # `root_nodes` don't cause cycle.
        return False

    # Use all output nodes as roots to traverse
    # the graph to check cycles.
    if bfs_find_cycle(outputs):
        return False

    return True

class GTLSubgraphMatcher(SubgraphMatcher):
    def __init__(self, pattern: Graph, 
                 match_output: bool = False, 
                 match_placeholder: bool = False, 
                 remove_overlapping_matches: bool = True) -> None:
        super().__init__(
            pattern, match_output, match_placeholder, remove_overlapping_matches)
    
    def match(self, graph: Graph) -> List[InternalMatch]:

        # find candidate nodes to match with pattern anchors
        match_candidates: Dict[Node, List[Node]] = defaultdict(list)
        for pattern_anchor in self.pattern_anchors:
            for node in graph.nodes:
                if self._nodes_are_equal(pattern_anchor, node):
                    match_candidates[pattern_anchor].append(node)
        match_candidates_list = list(match_candidates.items())

        logger.info(f"Initial match_candidates_list: {match_candidates_list}\n")

        matches: List[InternalMatch] = []

        def backtracking(anchor_index, match):
            if anchor_index == len(match_candidates_list):
                match.placeholder_nodes = [match.nodes_map[pn] for pn in self.pattern_placeholder_nodes]
                match.returning_nodes = [match.nodes_map[pn] for pn in self.pattern_returning_nodes]
                matches.append(match)

                logger.info(f"Found a match: {match}\n")
                return

            pattern_anchor, candidate_nodes = match_candidates_list[anchor_index]
            saved_match = copy.copy(match)

            for node in candidate_nodes:
                logger.info(f"Trying to match anchor {pattern_anchor} to {node}")

                match_found = self._match_nodes(pattern_anchor, node, match)
                if match_found:
                    # match next anchor
                    backtracking(anchor_index + 1, match)
                else:
                    logger.info(f"Failed to match anchor {pattern_anchor} to {node}\n")

                # revert to saved_match before matching with current anchor
                match = copy.copy(saved_match)

        match = InternalMatch(anchors=self.pattern_anchors)
        if match_candidates_list:
            backtracking(0, match)

        # filter out the matches where the subgraph is not fully_contained
        before = len(matches)
        matches = [match for match in matches if self._is_contained(match.nodes_map)]
        after = len(matches)
        if before != after:
            logger.info(f"Filtered out {before - after} matches because they are not fully contained")

        # filter out the matches that that forms a cycle if the subgraph is fused
        valid_matches = []
        for match in matches:
            matched_compute_nodes = \
                [gn for pn, gn in match.nodes_map.items() if pn.op not in {"placeholder", "output"}]
            if validate_partition(matched_compute_nodes):
                valid_matches.append(match)
        if len(valid_matches) != len(matches):
            logger.info(f"Filtered out {len(matches) - len(valid_matches)} matches because \
                          matched subgraph would form a cycle if fused")

        if self.remove_overlapping_matches:
            before = len(valid_matches)
            matches = self._remove_overlapping_matches(valid_matches)
            after = len(matches)
            if before != after:
                logger.info(f"Filtered out {before - after} matches because matched subgraphs are overlapping")

        logger.info(f"Matches returned: {matches}")

        return matches


def replace_pattern(
    gm: GraphModule,
    pattern: Union[Callable, GraphModule],
    replacement: Union[Callable, GraphModule]
) -> List[Match]:
    match_and_replacements = _replace_pattern(gm, pattern, replacement)
    return [Match(anchor=m.anchor, nodes_map=m.nodes_map) for m in match_and_replacements]

def _replace_pattern(
    gm: GraphModule,
    pattern: Union[Callable, GraphModule],
    replacement: Union[Callable, GraphModule],
    match_filters: List[Callable[["InternalMatch", Graph, Graph], bool]] = None,  # type: ignore[name-defined]
) -> List[ReplacedPatterns]:

    if match_filters is None:
        match_filters = []

    # Get the graphs for `gm`, `pattern`, `replacement`
    original_graph: Graph = gm.graph

    if isinstance(pattern, GraphModule):
        pattern_graph = pattern.graph
    else:
        pattern_graph = symbolic_trace(pattern).graph

    if isinstance(replacement, GraphModule):
        replacement_graph = replacement.graph
    else:
        replacement_graph = symbolic_trace(replacement).graph

    matcher = GTLSubgraphMatcher(pattern_graph, match_output=False, match_placeholder=False,
                              remove_overlapping_matches=True)
    _matches: List[InternalMatch] = matcher.match(original_graph)

    # Filter out matches that don't match the filter
    _matches = [
        m for m in _matches
        if all(match_filter(m, original_graph, pattern_graph)
               for match_filter in match_filters)
    ]

    replacement_placeholders = [n for n in replacement_graph.nodes if n.op == "placeholder"]

    # As we progressively replace nodes, we'll need to keep track of how the match results should change
    match_changed_node: Dict[Node, Node] = {}

    match_and_replacements = []
    for match in _matches:

        # Build connecting between replacement graph's input and original graph input producer node

        # Initialize `val_map` with mappings from placeholder nodes in
        # `replacement` to their corresponding node in `original_graph`
        assert len(match.placeholder_nodes) == len(replacement_placeholders)
        val_map: Dict[Node, Node] = {}
        for rn, gn in zip(replacement_placeholders, match.placeholder_nodes):
            if isinstance(gn, Node):
                val_map[rn] = match_changed_node.get(gn, gn)
            else:
                val_map[rn] = gn

        # Copy the replacement graph over
        user_nodes: Set[Node] = set()
        for n in match.returning_nodes:
            for user in n.users:
                user_nodes.add(user)
        assert user_nodes, "The returning_nodes should have at least one user node"

        if len(user_nodes) == 1:
            first_user_node = list(user_nodes)[0]
        else:
            # If there are multiple user nodes, we need to find the first user node
            # in the current execution order of the `original_graph`
            for n in original_graph.nodes:
                if n in user_nodes:
                    first_user_node = n
                    break

        with original_graph.inserting_before(first_user_node):
            copied_returning_nodes = original_graph.graph_copy(replacement_graph, val_map)

        if isinstance(copied_returning_nodes, Node):
            copied_returning_nodes = (copied_returning_nodes, )

        # Get a list of nodes that have been replaced into the graph
        replacement_nodes = []

        def get_replacement_nodes(curr_node: Node):
            nonlocal replacement_nodes
            for arg in curr_node.args:
                if isinstance(arg, Node):
                    if arg not in val_map.values():
                        get_replacement_nodes(arg)
            replacement_nodes.append(curr_node)

        for ret_node in copied_returning_nodes:
            get_replacement_nodes(ret_node)

        # Hook the output Node of the replacement subgraph in to the
        # original Graph at the correct location
        assert len(match.returning_nodes) == len(copied_returning_nodes)
        for gn, copied_node in zip(match.returning_nodes, copied_returning_nodes):
            gn.replace_all_uses_with(copied_node)
            match_changed_node[gn] = copied_node
        # Remove the original nodes
        for node in reversed(pattern_graph.nodes):
            if node.op != "placeholder" and node.op != "output":
                gn = match.nodes_map[node]
                gm.graph.erase_node(gn)

        match_and_replacements.append(
            ReplacedPatterns(
                anchor=match.anchors[0],
                nodes_map=match.nodes_map,
                replacements=replacement_nodes
            )
        )

    # Update the passed-in GraphModule to reflect the new state of
    # `original_graph`
    gm.recompile()

    # If `replacement` was an nn.Module, we'll need to make sure that
    # all the submodules have been copied over correctly
    if isinstance(replacement, torch.nn.Module):
        _replace_submodules(gm, replacement)

    return match_and_replacements