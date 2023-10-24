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
from gtl.compiler.nodes import *

################################################################################
# Graph-level pass to replase tranpose node on 3D tensors with permutation
# This simplifies the kernel fusion interface
################################################################################
def pass_trans_2_permute(module, graph):
    for node in graph.nodes:
        if node.target == torch.ops.aten.transpose:
            shape = list(node.meta['tensor_meta'].shape)
            if len(shape) >= 3:
                dim = [i for i in range(len(shape))]
                dim[node.args[1]] = node.args[2]
                dim[node.args[2]] = node.args[1]

                permute_node = inject_permute(node, graph, node.args[0], dims=dim)
                node.replace_all_uses_with(permute_node)
                graph.erase_node(node)



                
