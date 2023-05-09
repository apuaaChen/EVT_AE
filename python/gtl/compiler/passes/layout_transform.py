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


################################################################################
# Graph-level pass to preprocess layer norm for layout transformation
################################################################################
def pass_nchw_to_nhwc(module, graph):
    for node in graph.nodes:
        if node.target == torch.ops.aten.mean:
            if len(node.meta["tensor_meta"].shape) == 4:
                node_arg = list(node.args)
                if node_arg[1] == [-1, -2]:
                    node_arg[1] = [-2, -3]
                
                node.args = tuple(node_arg)
        
        elif node.target in [torch.ops.aten.view, torch.ops.aten.expand]:
            if len(node.args[1]) == 4:
                new_shape = [node.args[1][i] for i in [0, 2, 3, 1]]
                node.args = (node.args[0], new_shape)