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
from torch.fx.passes.shape_prop import TensorMetadata

################################################################################
# Graph-level pass to update data type of module attributes
################################################################################

def pass_update_attributes(module, graph):
    for node in graph.nodes:
        if node.op == "get_attr":
            attr = node.target
            # tensor = getattr(module, attr).to(torch.float16).to("cuda")
            tensor = getattr(module, attr).to("cuda")
            setattr(module, attr, tensor)
            node.meta = {}
            node.meta['tensor_meta'] = TensorMetadata(
                shape=tensor.shape, dtype=tensor.dtype, requires_grad=False, 
                stride=(1,), memory_format=torch.contiguous_format, 
                is_quantized=False, qparams={})
