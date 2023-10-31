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
# Unit test for evt fuser on mm operator
import torch
import unittest
from gtl.compiler.passes import pass_fusion
from gtl.helper import UnitTestBase


class EVTFuserMM(UnitTestBase):
    
    # Helper function for launching test
    def util_test_evt_fuser(self, cls, inputs):
        self.util_test(cls, inputs, [pass_fusion], criteria={"atol": 5e-1}, profile=True)
    
    def test_mm_partition1(self):
        # Model
        class MMPartition1(torch.nn.Module):
            def __init__(self) -> None:
                super().__init__()
            
            def forward(self, primals_1, view_11, primals_2):
                permute_24 = torch.ops.aten.permute(primals_1, [1, 0])
                mm_37 = torch.ops.aten.mm(view_11, permute_24)
                add_21 = torch.ops.aten.add(mm_37, primals_2)
                view_12 = torch.ops.aten.view(add_21, [512, 2, 4096])
                gelu = torch.ops.aten.gelu(view_12, approximate="tanh")
                view_13 = torch.ops.aten.view(gelu, [1024, 4096])
                return view_13, view_12
        
        # Inputs
        primals_1 = torch.randn((4096, 1024), dtype=torch.float16, device="cuda")
        view_11 = torch.randn((1024, 1024), dtype=torch.float16, device="cuda")
        primals_2 = torch.randn((4096,), dtype=torch.float16, device="cuda")

        self.util_test_evt_fuser(MMPartition1, [primals_1, view_11, primals_2])

if __name__ == '__main__':
    unittest.main()
