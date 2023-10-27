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
from torch.fx.passes import graph_drawer


################################################################################
# Graph-level pass to print fx graph to disk as svg file
################################################################################
def pass_print_graph(module, file):
    """
    Print the graph in {file}
    """
    g = graph_drawer.FxGraphDrawer(module, "dynamic_classifier")
    with open(file, "wb") as f:
        f.write(g.get_dot_graph().create_svg())
    # raw_dot = g.get_dot_graph().to_string()
    # g.get_dot_graph().write_raw("./test.dot")