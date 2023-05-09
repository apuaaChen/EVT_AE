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
# Temporary fix for the permute->view error
################################################################################

def pass_permute_view_fix(module, graph):
    for node in graph.nodes:
        if node.target == torch.ops.aten.permute:
            # create clone_ node
            graph.inserting_after(node)
            clone_node = graph.call_function(torch.ops.aten.clone, args=(node.args[0],), kwargs={"memory_format": torch.contiguous_format})
            node.replace_all_uses_with(clone_node)
            clone_node.replace_input_with(node.args[0], node)
        # if node.target == torch.ops.aten.view:
        #     node.target = tensor_contiguous_view
