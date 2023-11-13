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
# Unit test for evt fuser on dropout operator
import torch
import unittest
from gtl.compiler.passes import pass_fusion
from gtl.helper import UnitTestBase


class EVTFuserSoftmaxBackward(UnitTestBase):
    # Helper function for launching test
    def util_test_evt_fuser(self, cls, inputs):
        self.util_test(
            cls, inputs, [pass_fusion], 
            criteria={"atol": 3e-1}, profile=False)
    
    def test_softmaxbackward(self):
        # Model
        class SoftmaxBackward(torch.nn.Module):
            def forward(self, grad, input):
                _softmax = torch.ops.aten._softmax_backward_data(grad, input, -1, torch.float16)
                return _softmax
        
        # Inputs
        input = torch.randn((4, 16, 512, 512), dtype=torch.float16, device="cuda")
        grad = torch.randn((4, 16, 512, 512), dtype=torch.float16, device="cuda")

        self.util_test_evt_fuser(SoftmaxBackward, [grad, input])

if __name__ == '__main__':
    unittest.main()