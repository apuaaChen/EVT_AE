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
# Unit test for evt fuser on spmm operator
import torch
import unittest
from gtl.compiler.passes import pass_fusion, pass_decomposition
from gtl.helper import UnitTestBase
from model_zoo.gcn import example_inputs as gcn_input


feature, _, csr, _, _, _ = gcn_input("ogbn-mag")
num_nodes = feature.size(0)

def stride():
    return (0,0)

def is_contiguous(*args, **kwargs):
    return True

setattr(csr, "stride", stride)
setattr(csr, "is_contiguous", is_contiguous)

class EVTFuserSpmm(UnitTestBase):

    # Helper function for launching test
    def util_test_evt_fuser(self, cls, inputs):
        self.util_test(cls, inputs, [pass_decomposition, pass_fusion], criteria={"atol": 2e-1}, profile=False)
    
    def test_spmm_partition1(self):
        # Model
        class SpMMPartition1(torch.nn.Module):
            def __init__(self, *args, **kwargs) -> None:
                super().__init__(*args, **kwargs)
                self.csr = csr
            def forward(self, embedding):
                spmm = torch.ops.aten.mm(csr, embedding)
                return spmm
        
        # Inputs
        embedding = torch.randn((num_nodes, 64), dtype=torch.float16, device="cuda")

        self.util_test_evt_fuser(SpMMPartition1, [embedding])


if __name__ == '__main__':
    unittest.main()
