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


class EVTFuserLayerNorm(UnitTestBase):
    # Helper function for launching test
    def util_test_evt_fuser(self, cls, inputs):
        self.util_test(
            cls, inputs, [pass_fusion], 
            criteria={"atol": 3e-2}, profile=False)
    
    def test_layer_norm_pattern1(self):
        # Model
        class LayerNorm(torch.nn.Module):
            def forward(self, add_4, primals_18, primals_19):
                native_layer_norm_5 = torch.ops.aten.native_layer_norm(
                    add_4, [1024], None, None, 1e-12
                )
                getitem_46 = operator.getitem(native_layer_norm_5, 0)
                getitem_47 = operator.getitem(native_layer_norm_5, 1)
                getitem_48 = operator.getitem(native_layer_norm_5, 2)
                mul_2 = torch.ops.aten.mul(getitem_46, primals_18)
                add_32 = torch.ops.aten.add(mul_2, primals_19)
                view_11 = torch.ops.aten.view(add_32, [1024,1024])
                return getitem_47, getitem_48, add_32, view_11
        
        # Inputs
        add_4 = torch.randn((512, 2, 1024), dtype=torch.float16, device="cuda") + 1.
        primals_18 = torch.randn((1024,), dtype=torch.float16, device="cuda")
        primals_19 = torch.randn((1024,), dtype=torch.float16, device="cuda")


        self.util_test_evt_fuser(LayerNorm, [add_4, primals_18, primals_19])
    
    def test_layer_norm_pattern2(self):
        # Model
        class LayerNorm(torch.nn.Module):
            def forward(self, add_8, primals_36, primals_37):
                native_layer_norm_8 = torch.ops.aten.native_layer_norm(
                    add_8, [1024], None, None, 1e-12
                )
                getitem_55 = operator.getitem(native_layer_norm_8, 0)
                getitem_56 = operator.getitem(native_layer_norm_8, 1)
                getitem_57 = operator.getitem(native_layer_norm_8, 2)
                mul_5 = torch.ops.aten.mul(getitem_55, primals_36)
                add_35 = torch.ops.aten.add(mul_5, primals_37)
                permute_141 = torch.ops.aten.permute(add_35, [1, 0, 2])
                clone_5 = torch.ops.aten.clone(permute_141, memory_format=torch.contiguous_format)
                view_33 = torch.ops.aten.view(clone_5, [1024,1024])
                return getitem_56, getitem_57, clone_5, view_33
        
        # Inputs
        add_8 = torch.randn((512, 2, 1024), dtype=torch.float16, device="cuda") + 1.
        primals_36 = torch.randn((1024,), dtype=torch.float16, device="cuda")
        primals_37 = torch.randn((1024,), dtype=torch.float16, device="cuda")


        self.util_test_evt_fuser(LayerNorm, [add_8, primals_36, primals_37])


if __name__ == '__main__':
    unittest.main()