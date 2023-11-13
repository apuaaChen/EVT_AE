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
# Autotuner for BMM EVT
from gtl.compiler.autotuner.tuner_base import AutoTunerBase
from gtl.compiler.autotuner.gemm_tuner import MMTuner
from cutlass.shape import GemmCoord
import torch
from cutlass.utils.datatypes import torch_type
from cutlass import DataTypeTag, LayoutTag, LayoutType
import numpy as np
from gtl.ops.bmm import BmmArguments2x
from cutlass import epilogue


class BMMTuner(MMTuner):
    VALID_SWIZZLES = ["Identity1", "Identity2", "Identity4", "Identity8"]
    VALID_EPILOGUE_STAGE = [1, 2]
    table_name = "best_config_bmm"
    config_columns = MMTuner.config_columns

    def __init__(self, plan, batch_count, problem_size: GemmCoord, permute_A, permute_B, epilogue_visitor):
        super(MMTuner, self).__init__(epilogue_visitor, torch.ops.aten.bmm)
        self.plan = plan
        self.ps = problem_size
        self.batch_count = batch_count
        self.problem_size = problem_size

        self.dtype_A = torch_type(self.plan._element_a)
        self.layout_A = self.plan._layout_a
        if permute_A in [[1,2,0],[2,1,0]]:
            permute_A = [0, 1, 2]
        self.permute_A = permute_A
        inv_permute_A = np.argsort(permute_A)
        self.shape_A = [batch_count, self.ps.m, self.ps.k]
        self.shape_A = [self.shape_A[i] for i in inv_permute_A]
        if self.layout_A == LayoutType.RowMajor:
            self.align_A = self.get_alignment(self.ps.k)
        else:
            self.align_A = self.get_alignment(self.ps.m)

        self.dtype_B = torch_type(self.plan._element_b)
        self.layout_B = self.plan._layout_b
        if permute_B in [[1,2,0],[2,1,0]]:
            permute_B = [0,1,2]
        self.permute_B = permute_B
        inv_permute_B = np.argsort(permute_B)
        self.shape_B = [batch_count, self.ps.k, self.ps.n]
        self.shape_B = [self.shape_B[i] for i in inv_permute_B]
        if self.layout_B == LayoutType.RowMajor:
            self.align_B = self.get_alignment(self.ps.n)
        else:
            self.align_B = self.get_alignment(self.ps.k)

        self.dtype_C = torch_type(self.plan._element_c)
        self.shape_C = (batch_count, self.ps.m, self.ps.n)
        self.align_C = self.get_alignment(self.ps.n)

        self.valid_tds = []
        for td in self.plan.tile_descriptions():
            if td not in self.valid_tds:
                self.valid_tds.append(td)
        
    @property
    def key_no_epilogue(self):
        """
        The key is {batch_count}_{m}_{n}_{k}_{elem_a}_{elem_b}_{layout_a}_{layout_b}_{batch_count}
        """
        return f"{self.batch_count}_{self.ps.m}_{self.ps.n}_{self.ps.k}_{DataTypeTag[self.plan._element_a]}_{DataTypeTag[self.plan._element_b]}_{LayoutTag[self.layout_A]}_{LayoutTag[self.layout_B]}"
    
    def profile_with_config(self, config, with_epilogue, *args, **kwargs):
        td, swizzle, epi_stage = config
        if with_epilogue:
            self.epilogue_visitor.epilogue_stages = epi_stage
            self.plan.epilogue_visitor = self.epilogue_visitor
            output_args = [kwargs["visitor_args"]]
        else:
            self.plan.activation = epilogue.identity
            output_args = [1.0, 0.0]
        self.plan.tile_description = td
        self.plan.swizzling_functor = swizzle
        operation = self.plan.compile(
            alignment_A=self.align_A, alignment_B=self.align_B, 
            alignment_C=self.align_C
        )
        arguments = BmmArguments2x(
            operation=operation, problem_size=self.problem_size,
            A=args[0], B=args[1], C=args[2], D=args[3], 
            permute_A=self.permute_A, permute_B=self.permute_B,
            output_op=operation.epilogue_type(*output_args)
        )
        operation.run(arguments)
        return self.torch_profiler(operation.run, arguments)


    def get_arguments_no_epilogue(self):
        tensor_A = torch.empty(size=self.shape_A, dtype=self.dtype_A, device="cuda")
        tensor_B = torch.empty(size=self.shape_B, dtype=self.dtype_A, device="cuda")
        tensor_C = torch.empty(size=self.shape_C, dtype=self.dtype_C, device="cuda")
        tensor_D = torch.empty_like(tensor_C)
        return [tensor_A, tensor_B, tensor_C, tensor_D], {}
