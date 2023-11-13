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
# Overloaded argument type for BMM to support permutation

from cutlass.backend.gemm_operation import GemmArguments2x, ArgumentBase, GemmOperationUniversal
from cutlass.backend.evt.ir.tensor import Tensor
from cutlass import GemmUniversalMode, LayoutType


class BmmArguments2x(GemmArguments2x):
    def __init__(self, operation, problem_size, A, B, C, D, output_op, **kwargs):
        self.operation = operation
        self.layout_A = operation.A.layout
        self.layout_B = operation.B.layout
        self.layout_C = operation.C.layout

        self.element_A = operation.A.element
        self.element_B = operation.B.element
        self.element_C = operation.C.element

        ArgumentBase.__init__(self, A, B, C, D)

        fake_A = Tensor(tensor=A)
        fake_B = Tensor(tensor=B)

        if "permute_A" in kwargs:
            fake_A.permute(kwargs["permute_A"])
        
        if "permute_B" in kwargs:
            fake_B.permute(kwargs["permute_B"])
        
        self.problem_size = problem_size
        self.lda = self.get_leading_dim(fake_A, self.layout_A)
        self.ldb = self.get_leading_dim(fake_B, self.layout_B)
        self.ldc = fake_A.shape[1]
        self.ldd = self.ldc
        self.output_op = output_op
        self.gemm_mode = GemmUniversalMode.Batched
        self.batch_count = fake_A.shape[0]
        self.batched_stride_A = fake_A.stride[0]
        self.batched_stride_B = fake_B.stride[0]
        self.batched_stride_C = fake_A.shape[1] * fake_B.shape[2]
        self.batched_stride_D = self.batched_stride_C

        if isinstance(self.operation, GemmOperationUniversal):
            self.initialize()
    
    def get_leading_dim(self, fake_tensor, layout):
        if layout == LayoutType.RowMajor:
            return fake_tensor.stride[1]
        else:
            return fake_tensor.stride[2]