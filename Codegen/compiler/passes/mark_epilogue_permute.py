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
# Mark the permutations to be fused into epilogues
################################################################################
def pass_mark_epilogue_permutations(module, graph):
    for node in graph.nodes:
        if node.target == torch.ops.aten.permute:
            if len(node.users) == 1:
                if list(node.users.keys())[0].target == torch.ops.aten.clone:
                    node.meta["epilogue_permute"] = True
                    print(node)