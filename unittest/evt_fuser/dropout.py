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
from gtl.compiler.passes import pass_fusion, pass_decomposition
from gtl.helper import UnitTestBase
from torch.fx import symbolic_trace
from torch.fx.passes.shape_prop import ShapeProp
import operator


class EVTFuserDropout(UnitTestBase):
    # Helper function for launching test
    def util_test_evt_fuser(self, cls, inputs):
        self.util_test(
            cls, inputs, [pass_decomposition, pass_fusion], 
            criteria={"atol": 2e-1}, profile=False)
    
    def test_dropout(self):
        # Model
        class Dropout1(torch.nn.Module):
            def forward(self, view_9, primals_16, primals_17, permute_104):
                permute_115 = torch.ops.aten.permute(primals_16, [1, 0])
                mm_36 = torch.ops.aten.mm(view_9, permute_115)
                add_20 = torch.ops.aten.add(mm_36, primals_17)
                view_10 = torch.ops.aten.view(add_20, [512, 2, 1024])
                dropout = torch.ops.aten.native_dropout(view_10, 0.7, True)
                getitem_2 = operator.getitem(dropout, 0)
                getitem_3 = operator.getitem(dropout, 1)
                add_4 = torch.ops.aten.add(getitem_2, permute_104)
                return add_4, getitem_3
        
        # Inputs
        view_9 = torch.randn((1024, 1024), dtype=torch.float16, device="cuda")
        primals_16 = torch.randn((1024, 1024), dtype=torch.float16, device="cuda")
        primals_17 = torch.randn((1024,), dtype=torch.float16, device="cuda")
        permute_104 = torch.randn((512, 2, 1024), dtype=torch.float16, device="cuda")

        inputs = [view_9, primals_16, primals_17, permute_104]

        model = Dropout1()
        symbolic_traced: torch.fx.GraphModule = symbolic_trace(model)
        ShapeProp(symbolic_traced).propagate(*inputs)

        for p in [pass_decomposition, pass_fusion]:
            p(symbolic_traced, symbolic_traced.graph)
        symbolic_traced.recompile()

        out, mask = symbolic_traced(*inputs)

        # Compute reference output
        permute_115 = torch.ops.aten.permute(primals_16, [1, 0])
        mm_36 = torch.ops.aten.mm(view_9, permute_115)
        add_20 = torch.ops.aten.add(mm_36, primals_17)
        view_10 = torch.ops.aten.view(add_20, [512, 2, 1024])
        dropout = view_10 * mask / 0.3
        ref = torch.ops.aten.add(dropout, permute_104)

        self.assertTrue(torch.allclose(out, ref, atol=5e-1))
        self.assertTrue(torch.mean(mask.to(torch.float32)) > 0.3 - 1e-3)
        self.assertTrue(torch.mean(mask.to(torch.float32)) < 0.3 + 1e-3)
    
if __name__ == '__main__':
    unittest.main()
