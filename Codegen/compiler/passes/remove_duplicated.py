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
from nodes import *


################################################################################
# Graph-level pass to merge duplicated nodes
################################################################################

def pass_remove_duplicated_node(module, graph):
    # step 1: update topological index
    for idx, node in enumerate(graph.nodes):
        node.meta["topo_idx"] = idx
    modified = True
    while(modified):
        modified = False
        for u in graph.nodes:
            for input in u.all_input_nodes:
                user_list = list(input.users.keys())
                for v in user_list:
                    if u.meta["topo_idx"] >= v.meta["topo_idx"]: continue
                    if node_equal(u, v):
                        v.replace_all_uses_with(u)
                        graph.erase_node(v)
                        modified = True


    graph.eliminate_dead_code()
    graph.lint()