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
# Unit test for evt fuser on conv2d operator
import torch
import unittest
from gtl.compiler.passes import pass_fusion
from gtl.helper import UnitTestBase
from gtl.compiler.passes.pass_decomposition import convolution_forward_channel_last, batch_norm_stat, pass_decomposition
from gtl.compiler.passes.pass_permute_propagation import pass_permute_propagation

class EVTFuserMM(UnitTestBase):
    # Helper function for launching test
    def util_test_evt_fuser(self, cls, inputs):
        self.util_test(
            cls, inputs, 
            [pass_decomposition, pass_permute_propagation, pass_fusion], 
            criteria={"atol": 2e-1}, profile=False)
    
    def test_conv_fprop_partition1(self):
        # Model
        class Conv2dFpropPartition1(torch.nn.Module):
            def forward(self, input, weight, gamma, beta):
                conv_output = torch.ops.aten.convolution(
                    input, weight, None, [2, 2], [3, 3], [1, 1], False, [0, 0], 1)
                output, mean, rstd = torch.ops.aten._native_batch_norm_legit.no_stats(
                    conv_output, gamma, beta, True, 0.1, 1e-5)
                return output, mean, rstd
        
        # Inputs
        input = torch.randn((128,4,224,224), dtype=torch.float16, device="cuda")
        weight = torch.randn((64,4,7,7), dtype=torch.float16, device="cuda")
        gamma = torch.randn((64,), dtype=torch.float16, device="cuda")
        beta = torch.randn((64,), dtype=torch.float16, device="cuda")

        self.util_test_evt_fuser(Conv2dFpropPartition1, [input, weight, gamma, beta])

if __name__ == '__main__':
    # with VizTracer(output_file="./host.html") as tracer:
    unittest.main()
