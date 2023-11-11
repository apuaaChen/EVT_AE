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
            criteria={"atol": 3e-2}, verify=False, profile=True)
    
    def test_layer_norm_backward_pattern1(self):
        # Model
        class LayerNormBackward(torch.nn.Module):
            def forward(self, grad, input, mean, invstd, gamma, beta, ge_1):
                native_layer_norm = torch.ops.aten.native_layer_norm_backward(
                    grad, input, [1024], mean, invstd, gamma, beta, [True, False, False]
                )
                getitem = operator.getitem(native_layer_norm, 0)
                mul_42 = torch.ops.aten.mul(getitem, ge_1)
                mul_43 = torch.ops.aten.mul(mul_42, 1.0)
                view_282 = torch.ops.aten.view(mul_43, [16384, 1024])
                sum_13 = torch.ops.aten.sum(view_282, [0], True)
                return getitem, view_282, sum_13
        
        # Inputs
        grad = torch.randn((512, 32, 1024), dtype=torch.float16, device="cuda")
        input = torch.randn((512, 32, 1024), dtype=torch.float16, device="cuda")
        mean = torch.mean(input, dim=2, keepdim=True, dtype=torch.float32)
        invstd = 1./torch.std(input, dim=2, keepdim=True)
        invstd = invstd.to(torch.float32)
        gamma = torch.randn((1024,), dtype=torch.float16, device="cuda")
        beta = torch.randn_like(gamma)
        ge_1 = torch.ops.aten.ge(torch.rand_like(input), 0.5)

        self.util_test_evt_fuser(LayerNormBackward, [grad, input, mean, invstd, gamma, beta, ge_1])


if __name__ == '__main__':
    unittest.main()