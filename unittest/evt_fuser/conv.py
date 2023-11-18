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
import operator

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
    
    def test_conv_dgrad_partition1(self):
        # Model
        class Conv2dDgradPartition1(torch.nn.Module):
            def forward(self, grad_y, input, weight, bn_input_, saved_mean, saved_rstd, gamma):
                conv_output = torch.ops.aten.convolution_backward(
                    grad_y, input, weight, [0], [1,1], [1,1], [1,1], False, [0,0], 1, [True, True, False]
                )
                grad_input = operator.getitem(conv_output, 0)
                grad_weight = operator.getitem(conv_output, 1)
                bn_input = torch.ops.aten.permute(bn_input_, [0, 3, 1, 2])
                grad_x, grad_gamma, grad_beta = torch.ops.aten.native_batch_norm_backward(
                    grad_input, bn_input, gamma, None, None, saved_mean, saved_rstd, True, 1e-5, [True, True, True]
                )
                return grad_x, grad_gamma, grad_beta, grad_weight
        
        # Inputs
        grad_y = torch.randn((128, 64, 56, 56), dtype=torch.float16, device="cuda")
        input = torch.randn((128, 64, 56, 56), dtype=torch.float16, device="cuda")
        weight = torch.randn((64, 64, 3, 3), dtype=torch.float16, device="cuda")
        bn_input = torch.randn((128, 56, 56, 64), dtype=torch.float16, device="cuda") * 2 + 1
        saved_mean = torch.ops.aten.mean(bn_input, [0, 1, 2], dtype=torch.float32, keepdim=False)
        saved_var = torch.ops.aten.mean(bn_input * bn_input, [0, 1, 2], dtype=torch.float32, keepdim=False) - saved_mean * saved_mean
        gamma = torch.randn((64,), dtype=torch.float16, device="cuda")
        saved_rstd = torch.ops.aten.rsqrt(saved_var)
        # gamma = torch.randn((64,), dtype=torch.float16, device="cuda")
        # beta = torch.randn((64,), dtype=torch.float16, device="cuda")

        self.util_test_evt_fuser(Conv2dDgradPartition1, [grad_y, input, weight, bn_input, saved_mean, saved_rstd, gamma])
    
    def test_conv_dgrad_strided(self):
        # Model
        class Conv2dDgradStrided(torch.nn.Module):
            def forward(self, grad_y, input, weight):
                conv_output = torch.ops.aten.convolution_backward(
                    grad_y, input, weight, [0], [2,2], [1,1], [1,1], False, [0,0], 1, [True, True, False]
                )
                grad_input = operator.getitem(conv_output, 0)
                grad_weight = operator.getitem(conv_output, 1)
                grad_input_permuted = torch.ops.aten.permute(grad_input, [0, 2, 3, 1])
                return grad_input_permuted, grad_weight
        
        # Inputs
        grad_y = torch.randn((128, 256, 14, 14), dtype=torch.float16, device="cuda")
        input = torch.randn((128, 128, 28, 28), dtype=torch.float16, device="cuda")
        weight = torch.randn((256, 128, 3, 3), dtype=torch.float16, device="cuda")

        self.util_test_evt_fuser(Conv2dDgradStrided, [grad_y, input, weight])

if __name__ == '__main__':
    # with VizTracer(output_file="./host.html") as tracer:
    unittest.main()
