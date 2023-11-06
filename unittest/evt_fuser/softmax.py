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


class EVTFuserSoftmax(UnitTestBase):
    # Helper function for launching test
    def util_test_evt_fuser(self, cls, inputs):
        self.util_test(
            cls, inputs, [pass_fusion], 
            criteria={"atol": 3e-3}, profile=False)
    
    def test_softmax_aux_store(self):
        # Model
        class Softmax(torch.nn.Module):
            def forward(self, input):
                _softmax = torch.ops.aten._softmax(input, -1, False)
                view = torch.ops.aten.view(_softmax, [16384, 512])
                return view
        
        # Inputs
        input = torch.randn((16384, 512), dtype=torch.float16, device="cuda")

        self.util_test_evt_fuser(Softmax, [input,])
    
    def test_softmax_aux_load(self):
        # Model
        class Softmax(torch.nn.Module):
            def forward(self, input, aux):
                _softmax = torch.ops.aten._softmax(input, -1, False)
                out = torch.ops.aten.add(_softmax, aux)
                return out
        
        # Inputs
        input = torch.randn((16384, 512), dtype=torch.float16, device="cuda")
        aux = torch.randn((16384, 512), dtype=torch.float16, device="cuda")

        self.util_test_evt_fuser(Softmax, [input, aux])
    
    def test_softmax_row_broadcast(self):
        # Model
        class Softmax(torch.nn.Module):
            def forward(self, input, row):
                _softmax = torch.ops.aten._softmax(input, -1, False)
                out = torch.ops.aten.add(_softmax, row)
                return out
        
        # Inputs
        input = torch.randn((16384, 512), dtype=torch.float16, device="cuda")
        row = torch.randn((512), dtype=torch.float16, device="cuda")

        self.util_test_evt_fuser(Softmax, [input, row])
    
    def test_softmax_column_broadcast(self):
        # Model
        class Softmax(torch.nn.Module):
            def forward(self, input, col):
                _softmax = torch.ops.aten._softmax(input, -1, False)
                out = torch.ops.aten.add(_softmax, col)
                return out
        
        # Inputs
        input = torch.randn((16384, 512), dtype=torch.float16, device="cuda")
        column = torch.randn((16384,1), dtype=torch.float16, device="cuda")

        self.util_test_evt_fuser(Softmax, [input, column])
    
    def test_softmax_scalar_broadcast(self):
        # Model
        class Softmax(torch.nn.Module):
            def forward(self, input):
                _softmax = torch.ops.aten._softmax(input, -1, False)
                out = torch.ops.aten.mul(_softmax, 2.)
                return out
        
        # Inputs
        input = torch.randn((16384, 512), dtype=torch.float16, device="cuda")

        self.util_test_evt_fuser(Softmax, [input,])
    
    def test_softmax_column_reduce(self):
        # Model
        class Softmax(torch.nn.Module):
            def forward(self, input):
                _softmax = torch.ops.aten._softmax(input, -1, False)
                out = torch.ops.aten.mul(_softmax, 2.)
                sum = torch.ops.aten.sum(out, [1], True)
                return out, sum
        
        # Inputs
        input = torch.randn((16384, 512), dtype=torch.float16, device="cuda")

        self.util_test_evt_fuser(Softmax, [input,])
    
    def test_softmax_column_reduce(self):
        # Model
        class Softmax(torch.nn.Module):
            def forward(self, input):
                _softmax = torch.ops.aten._softmax(input, -1, False)
                out = torch.ops.aten.mul(_softmax, 2.)
                sum = torch.ops.aten.sum(out, [0], True)
                return out, sum

        # Inputs
        input = torch.randn((16, 512), dtype=torch.float16, device="cuda")

        self.util_test_evt_fuser(Softmax, [input,])

    
if __name__ == '__main__':
    unittest.main()