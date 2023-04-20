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


import torch.fx as fx
import torch
from torch._functorch.partitioners import default_partition
import logging


def partition_func(
        joint_module: fx.GraphModule, joint_inputs, joint_compiler, **kwargs):
    #
    joint_compiler(joint_module)
    return default_partition(joint_module, joint_inputs, **kwargs)

def compiler_fn(fx_module: torch.fx.GraphModule, _):
    logging.debug("============Optimized Source Code============")
    logging.debug(fx_module.code)
    return fx_module