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
# Unit test for pass_decomposition
import torch
import unittest
from gtl.compiler.passes import pass_decomposition
from gtl.helper import UnitTestBase


class Decomposition(UnitTestBase):

    # Helper function for launching test
    def util_test_decomposition(self, cls, inputs):
        self.util_test(cls, inputs, [pass_decomposition,])

    # decomposition of _log_softmax = aten.log(aten.softmax)
    def test_log_softmax_forward(self):
        # Model
        class LogSoftmaxForward(torch.nn.Module):
            def __init__(self) -> None:
                super().__init__()
            
            def forward(self, input):
                return torch.ops.aten._log_softmax(input, 1, False)
        
        # inputs
        x = torch.randn((512, 1024), dtype=torch.float16, device="cuda")

        self.util_test_decomposition(LogSoftmaxForward, [x,])
    
    def test_nll_loss_backward(self):
        # Model
        class NllLossBackward(torch.nn.Module):
            def __init__(self) -> None:
                super().__init__()
            
            def forward(
                    self, tangents, x, target):
                #
                log_softmax = torch.ops.aten._log_softmax(x, 1, False)
                nll_loss_forward = torch.ops.aten.nll_loss_forward(log_softmax, target, None, 2, -1)
                loss = nll_loss_forward[0]
                total_weight = nll_loss_forward[1]
                nll_loss_backward = torch.ops.aten.nll_loss_backward(
                    tangents, log_softmax, target, None, 
                    2, -1, total_weight)
                _log_softmax_backward_data = torch.ops.aten._log_softmax_backward_data(nll_loss_backward, log_softmax, 1, torch.float16)
                return loss, _log_softmax_backward_data
        
        # Inputs
        tangents = torch.randn((1,), dtype=torch.float16, device="cuda")
        x = torch.randn((512, 64), dtype=torch.float16, device='cuda')
        target = torch.randint(low=0, high=63, size=(512, ), dtype=torch.int64, device="cuda")

        self.util_test_decomposition(NllLossBackward, [tangents, x, target])
    
    def test_rsub(self):
        # Model
        class Rsub(torch.nn.Module):
            def __init__(self) -> None:
                super().__init__()
            
            def forward(self, x, y):
                return torch.ops.aten.rsub(x, y)
        
        # Inputs
        x = torch.randn((512, 1024), dtype=torch.float16, device='cuda')
        y = torch.randn((512, 1024), dtype=torch.float16, device='cuda')

        self.util_test_decomposition(Rsub, [x, y])
    
    def test_native_dropout_backward(self):
        # Model
        class NativeDropoutBackward(torch.nn.Module):
            def __init__(self) -> None:
                super().__init__()
            
            def forward(self, grad_y, mask):
                return torch.ops.aten.native_dropout_backward(grad_y, mask, 2.)
        
        # Inputs
        grad_y = torch.randn((512, 16, 1024), dtype=torch.float16, device='cuda')
        mask = torch.rand((512, 16, 1024), device='cuda') < 0.5

        self.util_test_decomposition(NativeDropoutBackward, [grad_y, mask])
    
    def test_threshold_backward(self):
        # Model
        class ThresholdBackward(torch.nn.Module):
            def __init__(self) -> None:
                super().__init__()
            
            def forward(self, grad_y, threshold_output):
                return torch.ops.aten.threshold_backward(grad_y, threshold_output, 0)
        
        # Inputs
        grad_y = torch.randn((512, 1024), dtype=torch.float16, device='cuda')
        threshold_output = torch.ops.aten.relu(torch.randn_like(grad_y))

        self.util_test_decomposition(ThresholdBackward, [grad_y, threshold_output])
        
    def test_addmm(self):
        # Model
        class Addmm(torch.nn.Module):
            def __init__(self) -> None:
                super().__init__()
            
            def forward(self, bias, lhs, rhs):
                return torch.ops.aten.addmm(bias, lhs, rhs)
        
        # Inputs
        lhs = torch.randn((16, 64), dtype=torch.float16, device='cuda')
        rhs = torch.randn((64, 16), dtype=torch.float16, device='cuda')
        bias = torch.randn((16,), dtype=torch.float16, device='cuda')

        self.util_test_decomposition(Addmm, [bias, lhs, rhs])
    
    def test_log_softmax_backward_data(self):
        # Model
        class LogSoftmaxBackwardData(torch.nn.Module):
            def __init__(self) -> None:
                super().__init__()
            
            def forward(self, grad_y, softmax):
                log_softmax = torch.ops.aten.log(softmax)
                return log_softmax, torch.ops.aten._log_softmax_backward_data(
                    grad_y, log_softmax, -1, torch.float32)
        
        # Inputs
        grad_y = torch.randn((2, 16, 24), dtype=torch.float32, device='cuda')
        x = torch.randn_like(grad_y)
        softmax = torch.ops.aten._softmax(x, -1, False)

        self.util_test_decomposition(LogSoftmaxBackwardData, [grad_y, softmax])
    
    def test_native_layer_norm(self):
        # Model
        class NativeLayerNorm(torch.nn.Module):
            def __init__(self) -> None:
                super().__init__()
            
            def forward(self, input, gamma, beta, interm):
                output, mean, rstd = torch.ops.aten.native_layer_norm(
                    input, [1024,], gamma, beta, 1e-12
                )

                grad_out = interm

                grad_input, grad_gamma, grad_beta = torch.ops.aten.native_layer_norm_backward(
                    grad_out, input, [1024,], mean, rstd, gamma, beta, 
                    [True, True, True]
                )
                return output, grad_input, grad_gamma, grad_beta
        
        # Inputs
        interm = torch.randn((512, 16, 1024), dtype=torch.float32, device="cuda")
        input = torch.randn_like(interm)
        gamma = torch.randn((1024,), dtype=torch.float32, device="cuda")
        beta = torch.randn_like(gamma)

        self.util_test_decomposition(NativeLayerNorm, [input, gamma, beta, interm])


if __name__ == '__main__':
    unittest.main()
