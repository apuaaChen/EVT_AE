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
# Unit test for evt fuser on bmm operator
import torch
import unittest
from gtl.compiler.passes import pass_fusion
from gtl.helper import UnitTestBase


class EVTFuserBMM(UnitTestBase):

    # Helper function for launching test
    def util_test_evt_fuser(self, cls, inputs):
        self.util_test(cls, inputs, [pass_fusion], criteria={"atol": 2e-1})
    
    def test_bmm_partition1(self):
        # Model
        class BMMPartition1(torch.nn.Module):
            def forward(self, view_5, view_7):
                permute_127 = torch.ops.aten.permute(view_5, [1, 0, 2])
                bmm_1 = torch.ops.aten.bmm(view_7, permute_127)
                permute_132 = torch.ops.aten.permute(bmm_1, [1, 0, 2])
                clone_3 = torch.ops.aten.clone(
                    permute_132, memory_format=torch.contiguous_format)
                view_8 = torch.ops.aten.view(clone_3, [512, 2, 1024])
                view_9 = torch.ops.aten.view(view_8, [1024, 1024])
                return view_9
        
        # Inputs
        view_5 = torch.randn((512, 32, 64), dtype=torch.float16, device="cuda")
        view_7 = torch.randn((32, 512, 512), dtype=torch.float16, device="cuda")

        self.util_test_evt_fuser(BMMPartition1, [view_5, view_7])
    
    def test_bmm_partition2(self):
        # Model
        class BMMPartition2(torch.nn.Module):
            def forward(self, view_3, view_83):
                permute_128 = torch.ops.aten.permute(view_3, [1, 2, 0])
                bmm_10 = torch.ops.aten.bmm(permute_128, view_83)
                permute_177 = torch.ops.aten.permute(bmm_10, [2, 0, 1])
                view_84 = torch.ops.aten.view(permute_177, [512, 2, 1024])
                # sum_15 = torch.ops.aten.sum(view_84, [0, 1], True)
                clone_11 = torch.ops.aten.clone(
                    view_84, memory_format=torch.contiguous_format)
                view_101 = torch.ops.aten.view(clone_11, [1024, 1024])
                return view_101#, sum_15
        
        # Inputs
        view_3 = torch.randn((512, 32, 64), dtype=torch.float16, device="cuda")
        view_83 = torch.randn((32, 512, 512), dtype=torch.float16, device="cuda")

        self.util_test_evt_fuser(BMMPartition2, [view_3, view_83])




if __name__ == '__main__':
    unittest.main()
