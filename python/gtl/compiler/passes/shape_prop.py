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
from torch.fx.passes.shape_prop import *
import torch


################################################################################
# Update metadata info of newly inserted nodes
################################################################################
def pass_shape_prop(module, graph):
    # get inputs
    inputs = []
    for node in graph.nodes:
        if node.op == "place_holder":
            shape = node.meta['tensor_meta'].shape
            dtype = node.meta['tensor_meta'].dtype
            inputs.append(torch.empty(size=shape, dtype=dtype, device="cuda"))
    
    # shape prop
    ShapeProp(module).propagate(*inputs)