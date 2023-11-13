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
# Unit test for evt fuser on mm operator
import torch
import unittest
from gtl.compiler.passes import pass_fusion
from gtl.helper import UnitTestBase
from viztracer import VizTracer


class EVTFuserMM(UnitTestBase):
    
    # Helper function for launching test
    def util_test_evt_fuser(self, cls, inputs):
        self.util_test(cls, inputs, [pass_fusion], criteria={"atol": 2e-1}, profile=False)
    
    def test_mm_partition1(self):
        # Model
        class MMPartition1(torch.nn.Module):
            def __init__(self) -> None:
                super().__init__()
            
            def forward(self, primals_1, view_11, primals_2):
                permute_24 = torch.ops.aten.permute(primals_1, [1, 0])
                mm_37 = torch.ops.aten.mm(view_11, permute_24)
                add_21 = torch.ops.aten.add(mm_37, primals_2)
                view_12 = torch.ops.aten.view(add_21, [512, 2, 4096])
                gelu = torch.ops.aten.gelu(view_12, approximate="tanh")
                view_13 = torch.ops.aten.view(gelu, [1024, 4096])
                return view_13, view_12
        
        # Inputs
        primals_1 = torch.randn((4096, 1024), dtype=torch.float16, device="cuda")
        view_11 = torch.randn((1024, 1024), dtype=torch.float16, device="cuda")
        primals_2 = torch.randn((4096,), dtype=torch.float16, device="cuda")

        self.util_test_evt_fuser(MMPartition1, [primals_1, view_11, primals_2])
    
    def test_mm_partition2(self):
        # Model
        class MMPartition2(torch.nn.Module):
            def __init__(self) -> None:
                super().__init__()
            
            def forward(self, select, primals_5, primals_6):
                permute_32 = torch.ops.aten.permute(primals_5, [1, 0])
                mm_45 = torch.ops.aten.mm(select, permute_32)
                add_29 = torch.ops.aten.add(mm_45, primals_6)
                tanh = torch.ops.aten.tanh(add_29)
                return tanh
        
        # Inputs
        primals_5 = torch.randn((1024, 1024), dtype=torch.float16, device="cuda") * 0.018
        select = torch.randn((4, 1024), dtype=torch.float16, device="cuda")
        primals_6 = torch.randn((1024,), dtype=torch.float16, device="cuda") * 0.018

        self.util_test_evt_fuser(MMPartition2, [select, primals_5, primals_6])
    
    def test_mm_partition3(self):
        # Model
        class MMPartition3(torch.nn.Module):
            def __init__(self) -> None:
                super().__init__()
            
            def forward(self, view_93, primals_10, primals_11):
                permute_112 = torch.ops.aten.permute(primals_10, [1, 0])
                mm = torch.ops.aten.mm(view_93, permute_112)
                view = torch.ops.aten.view(mm, [512, 2, 1024])
                add = torch.ops.aten.add(view, primals_11)
                view_3 = torch.ops.aten.view(add, [512, 32, 64])
                return view_3
        
        # Inputs
        primals_10 = torch.randn((1024, 1024), dtype=torch.float16, device="cuda")
        view_93 = torch.randn((1024, 1024), dtype=torch.float16, device="cuda")
        primals_11 = torch.randn((1024,), dtype=torch.float16, device="cuda")

        self.util_test_evt_fuser(MMPartition3, [view_93, primals_10, primals_11])
    
    def test_mm_partition4(self):
        # Model
        class MMPartition4(torch.nn.Module):
            def __init__(self) -> None:
                super().__init__()
            
            def forward(self, tanh, primals_41, primals_42):
                permute_123 = torch.ops.aten.permute(primals_41, [1, 0])
                mm_47 = torch.ops.aten.mm(tanh, permute_123)
                add = torch.ops.aten.add(mm_47, primals_42)
                view_39 = torch.ops.aten.view(add, [2, 2])
                return view_39
        
        # Inputs
        primals_41 = torch.randn((2, 1024), dtype=torch.float16, device="cuda")
        tanh = torch.randn((2, 1024), dtype=torch.float16, device="cuda")
        primals_42 = torch.randn((2,), dtype=torch.float16, device="cuda")

        self.util_test_evt_fuser(MMPartition4, [tanh, primals_41, primals_42])
    
    def test_mm_partition5(self):
        # Model
        class MMPartition5(torch.nn.Module):
            def forward(self, A, B, tanh):
                mm = torch.ops.aten.mm(A, B)
                out = torch.ops.aten.tanh_backward(mm, tanh)
                sum = torch.ops.aten.sum(out, [0], True)
                return out, sum
        
        # Inputs
        A = torch.randn((4, 2), dtype=torch.float16, device="cuda")
        B = torch.randn((2, 1024), dtype=torch.float16, device="cuda")
        tanh = torch.ops.aten.tanh(torch.randn((4, 1024), dtype=torch.float16, device="cuda"))

        self.util_test_evt_fuser(MMPartition5, [A, B, tanh])

    def test_mm_partition6(self):
        # Model
        class MMPartition6(torch.nn.Module):
            def forward(self, view_295, primals_12, add_17, view_299):
                mm_33 = torch.ops.aten.mm(view_295, primals_12)
                view_298 = torch.ops.aten.view(mm_33, [512, 4, 1024])
                add_18 = torch.ops.aten.add(view_298, add_17)
                add_19 = torch.ops.aten.add(add_18, view_299)
                permute_183 = torch.ops.aten.permute(add_19, [1,0,2])
                return permute_183
        
        # Inputs
        view_295 = torch.randn((2048, 1024), dtype=torch.float16, device="cuda")
        primals_12 = torch.randn((1024, 1024), dtype=torch.float16, device="cuda")
        add_17 = torch.randn((512, 4, 1024), dtype=torch.float16, device="cuda")
        view_299 = torch.randn((512, 4, 1024), dtype=torch.float16, device="cuda")

        self.util_test_evt_fuser(MMPartition6, [view_295, primals_12, add_17, view_299])
    
    def test_mm_partition7(self):
        # Model
        class MMPartition7(torch.nn.Module):
            def forward(self, view_295, primals_12, add_17, view_299):
                mm_33 = torch.ops.aten.mm(view_295, primals_12)
                view_298 = torch.ops.aten.view(mm_33, [512, 4, 1024])
                add_18 = torch.ops.aten.add(view_298, add_17)
                add_19 = torch.ops.aten.add(add_18, view_299)
                view_183 = torch.ops.aten.view(add_19, [2048, 1024])
                return view_298, add_18, add_19, view_183
        
        # Inputs
        view_295 = torch.randn((2048, 1024), dtype=torch.float16, device="cuda")
        primals_12 = torch.randn((1024, 1024), dtype=torch.float16, device="cuda")
        add_17 = torch.randn((512, 4, 1024), dtype=torch.float16, device="cuda")
        view_299 = torch.randn((512, 4, 1024), dtype=torch.float16, device="cuda")

        self.util_test_evt_fuser(MMPartition7, [view_295, primals_12, add_17, view_299])

    def test_mm_partition9(self):
        # Model
        class MMPartition9(torch.nn.Module):
            def __init__(self) -> None:
                super().__init__()
            
            def forward(self, select, primals_5, primals_6):
                permute_32 = torch.ops.aten.permute(primals_5, [1, 0])
                mm_45 = torch.ops.aten.mm(select, permute_32)
                add_29 = torch.ops.aten.add(mm_45, primals_6)
                tanh = torch.ops.aten.tanh(add_29)
                add_30 = torch.ops.aten.add(tanh, primals_6)
                return add_30
        
        # Inputs
        primals_5 = torch.randn((1024, 1024), dtype=torch.float16, device="cuda") * 0.018
        select = torch.randn((4, 1024), dtype=torch.float16, device="cuda")
        primals_6 = torch.randn((1024,), dtype=torch.float16, device="cuda") * 0.018

        self.util_test_evt_fuser(MMPartition9, [select, primals_5, primals_6])

    def test_mm_partition10(self):
        # Model
        class MMPartition10(torch.nn.Module):
            def forward(self, view_237, primals_34):
                mm_12 = torch.ops.aten.mm(view_237, primals_34)
                sum_6 = torch.ops.aten.sum(mm_12, [0], True)
                return mm_12, sum_6
        
        # Input
        view_237 = torch.randn((16384,1024), dtype=torch.float16, device="cuda")
        primals_34 = torch.randn((1024, 4096), dtype=torch.float16, device="cuda")

        self.util_test_evt_fuser(MMPartition10, [view_237, primals_34])
    
    def test_mm_partition11(self):
        # Model
        class MMPartition11(torch.nn.Module):
            def forward(self, view_206, view_160):
                permute_142 = torch.ops.aten.permute(view_206, [1,0])
                mm_9 = torch.ops.aten.mm(permute_142, view_160)
                return mm_9

        # Inputs
        view_206 = torch.randn((33792, 768), dtype=torch.float16, device="cuda")
        view_160 = torch.randn((33792, 768), dtype=torch.float16, device="cuda")

        self.util_test_evt_fuser(MMPartition11, [view_206, view_160])




if __name__ == '__main__':
    # with VizTracer(output_file="./host.html") as tracer:
    unittest.main()
