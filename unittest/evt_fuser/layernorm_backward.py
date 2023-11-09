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
import operator


class EVTFuserLayerNormBackward(UnitTestBase):
    # Helper function for launching test
    def util_test_evt_fuser(self, cls, inputs):
        self.util_test(
            cls, inputs, [pass_fusion], 
            criteria={"atol": 3e-2}, profile=False)
    
    def test_layer_norm_backward_pattern1(self):
        # Model
        class LayerNormBackward(torch.nn.Module):
            def forward(self, grad, input, mean, invstd, gamma, beta):
                native_layer_norm = torch.ops.aten.native_layer_norm_backward(
                    grad, input, [1024], mean, invstd, gamma, beta, [True, False, False]
                )
                getitem = operator.getitem(native_layer_norm, 0)
                return getitem
        
        # Inputs
        grad = torch.randn((512, 4, 1024), dtype=torch.float16, device="cuda")
        input = torch.randn((512, 4, 1024), dtype=torch.float16, device="cuda")
        mean = torch.mean(input, dim=2, keepdim=True, dtype=torch.float32)
        invstd = 1./torch.std(input, dim=2, keepdim=True)
        invstd = invstd.to(torch.float32)
        gamma = torch.randn((1024,), dtype=torch.float16, device="cuda")
        beta = torch.randn_like(gamma)

        self.util_test_evt_fuser(LayerNormBackward, [grad, input, mean, invstd, gamma, beta])


if __name__ == '__main__':
    unittest.main()