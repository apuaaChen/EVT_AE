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
# Unit test for the uturn optimization
import torch
import unittest
from gtl.helper import UnitTestBase
from gtl.compiler.passes import pass_decomposition, pass_constant_propagation, pass_fusion

def pass_dead_code_elimination(model, graph):
    graph.eliminate_dead_code()

class Uturn(UnitTestBase):

    # Helper function for launching test
    def util_test_uturn(self, cls, inputs):
        self.util_test(cls, inputs, [
            pass_decomposition, pass_dead_code_elimination,
            pass_constant_propagation, pass_fusion])

    def test_uturn(self):
        # Model
        class UturnFpBp(torch.nn.Module):
            def __init__(self) -> None:
                super().__init__()
            
            def forward(
                    self, tangents, x, target):
                #
                log_softmax = torch.ops.aten._log_softmax(x, 1, False)
                nll_loss_forward = torch.ops.aten.nll_loss_forward(log_softmax, target, None, 2, -1)
                total_weight = nll_loss_forward[1]
                nll_loss_backward = torch.ops.aten.nll_loss_backward(
                    tangents, log_softmax, target, None, 
                    2, -1, total_weight)
                _log_softmax_backward_data = torch.ops.aten._log_softmax_backward_data(nll_loss_backward, log_softmax, 1, torch.float16)
                return _log_softmax_backward_data
        
        # Inputs
        tangents = torch.randn((1,), dtype=torch.float16, device="cuda")
        x = torch.randn((512, 64), dtype=torch.float16, device='cuda')
        target = torch.randint(low=0, high=63, size=(512, ), dtype=torch.int64, device="cuda")

        self.util_test_uturn(UturnFpBp, [tangents, x, target])
    
if __name__ == '__main__':
    unittest.main()
