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
from gtl.compiler.nodes import *
from torch.utils._python_dispatch import _pop_mode_temporarily


################################################################################
# Graph-level pass to replase loss node with a static Tensor 1.0, 
# Eliminates the original loss with dead code elimination
################################################################################
def pass_loss_elimination(module, graph):
    for node in graph.nodes:
        if node.op == "output":
            # loss is at input_node[0]
            loss_node = node.all_input_nodes[0]
            with _pop_mode_temporarily():
                fake_loss_node = inject_get_attr(
                    loss_node, module, graph, 
                    torch.Tensor([1.0,]).to("cuda").requires_grad_().to(torch.float16), 
                    "_fake_loss_0")

            loss_node.replace_all_uses_with(fake_loss_node)
    graph.eliminate_dead_code()
    graph.lint()