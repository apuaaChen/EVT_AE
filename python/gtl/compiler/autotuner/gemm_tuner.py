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
# Autotuner for GEMM EVT
import torch
from cutlass import SwizzlingFunctor, LayoutType
from gtl.compiler.autotuner.tuner_base import AutoTunerBase
from cutlass.shape import GemmCoord
from cutlass.utils.datatypes import torch_type
from cutlass import epilogue
from cutlass import DataTypeTag, LayoutTag
from cutlass.backend.library import TileDescription


class MMTuner(AutoTunerBase):
    VALID_SWIZZLES = ["Identity1", "Identity2", "Identity4", "Identity8"]
    VALID_EPILOGUE_STAGE = [1, 2]
    table_name = "best_config_mm"
    config_columns = {
        "tb_m": "INTEGER",
        "tb_n": "INTEGER",
        "tb_k": "INTEGER",
        "stages": "INTEGER",
        "wcnt_m": "INTEGER",
        "wcnt_n": "INTEGER",
        "wcnt_k": "INTEGER",
        "mi_m": "INTEGER",
        "mi_n": "INTEGER",
        "mi_k": "INTEGER",
        "swizzle": "TEXT",
        "epi_stages": "INTEGER"
    }
    def __init__(self, plan, problem_size: GemmCoord, epilogue_visitor) -> None:
        super().__init__(epilogue_visitor, torch.ops.aten.mm)
        self.plan = plan

        self.ps = problem_size

        self.dtype_A = torch_type(self.plan._element_a)
        self.layout_A = self.plan._layout_a
        self.shape_A = (self.ps.m, self.ps.k) if self.layout_A == LayoutType.RowMajor else (self.ps.k, self.ps.m)
        
        self.dtype_B = torch_type(self.plan._element_b)
        self.layout_B = self.plan._layout_b
        self.shape_B = (self.ps.k, self.ps.n) if self.layout_B == LayoutType.RowMajor else (self.ps.n, self.ps.k)

        self.dtype_C = torch_type(self.plan._element_c)
        self.shape_C = (self.ps.m, self.ps.n)

        self.valid_tds = []
        for td in self.plan.tile_descriptions():
            if td not in self.valid_tds:
                self.valid_tds.append(td)
    
    def construct_config_with_record(self, row):
        _, _, tb_m, tb_n, tb_k, stages, wcnt_m, wcnt_n, wcnt_k, mi_m, mi_n, mi_k, swizzle, epi_stages = row
        td = {
            "threadblock_shape": [tb_m,tb_n,tb_k],
            "warp_count": [wcnt_m, wcnt_n, wcnt_k],
            "stages": stages,
            "instruction_shape": [mi_m, mi_n, mi_k]
        }
        swizzling_functor = SwizzlingFunctor[swizzle]
        return (td, swizzling_functor, epi_stages)
    
    def construct_record_with_config(self, config):
        td, swizzling_functor, epi_stages = config
        if isinstance(td, TileDescription):
            tb = td.threadblock_shape
            stages = td.stages
            wcnt = td.warp_count
            mi = td.math_instruction.instruction_shape
        elif isinstance(td, dict):
            tb = td["threadblock_shape"]
            stages = td["stages"]
            wcnt = td["warp_count"]
            mi = td["instruction_shape"]
        else:
            raise ValueError("Invalid tile description type.")
        swizzle = swizzling_functor.name
        return (tb[0], tb[1], tb[2], stages, wcnt[0], wcnt[1], wcnt[2], mi[0], mi[1], mi[2], swizzle, epi_stages)
    
    @property
    def key_no_epilogue(self):
        """
        The key is {m}_{n}_{k}_{elem_a}_{elem_b}_{layout_a}_{layout_b}
        """
        return f"{self.ps.m}_{self.ps.n}_{self.ps.k}_{DataTypeTag[self.plan._element_a]}_{DataTypeTag[self.plan._element_b]}_{LayoutTag[self.layout_A]}_{LayoutTag[self.layout_B]}"
    
    def profile_with_config(self, config, with_epilogue, *args, **kwargs):
        td, swizzle, epi_stage = config
        if with_epilogue:
            self.epilogue_visitor.epilogue_stages = epi_stage
            self.plan.epilogue_visitor = self.epilogue_visitor
        else:
            self.plan.activation = epilogue.identity
        self.plan.tile_description = td
        self.plan.swizzling_functor = swizzle
        arguments = self.plan.run(*args, **kwargs)
        operation = self.plan.operation
        return self.torch_profiler(operation.run, arguments)
    
    def get_arguments_no_epilogue(self):
        tensor_A = torch.empty(size=self.shape_A, dtype=self.dtype_A, device="cuda")
        if self.layout_A == LayoutType.ColumnMajor:
            tensor_A = torch.transpose(tensor_A, -1, -2)
        tensor_B = torch.empty(size=self.shape_B, dtype=self.dtype_A, device="cuda")
        if self.layout_B == LayoutType.ColumnMajor:
            tensor_B = torch.transpose(tensor_B, -1, -2)
        tensor_C = torch.empty(size=self.shape_C, dtype=self.dtype_C, device="cuda")
        tensor_D = torch.empty_like(tensor_C)
        return [tensor_A, tensor_B, tensor_C, tensor_D], {}

    def profile_best_config_without_epilogue(self):
        configs = [(td, SwizzlingFunctor.Identity1, 1) for td in self.valid_tds]
        best_configs = self.profile_top_k_config(configs, self.num_best_tds, False)
        key = self.key_no_epilogue
        for rank, config in enumerate(best_configs):
            self.insert_record(key, rank, config)
        streamk_configs = [(td, SwizzlingFunctor.StreamK, 1) for td in self.valid_tds]
        best_streamk_configs = self.profile_top_k_config(streamk_configs, self.num_best_tds, False)
        key = self.key_no_epilogue
        for rank, config in enumerate(best_streamk_configs):
            self.insert_record(key, rank+self.num_best_tds, config)
        return best_configs + best_streamk_configs

    def profile_best_config_with_epilogue(self, configs):
        best_config = self.profile_top_k_config(configs, 1, True)[0]
        best_td, best_swizzle, _ = best_config
        if best_swizzle == SwizzlingFunctor.StreamK:
            valid_swizzles = ["StreamK"]
        else:
            valid_swizzles = self.VALID_SWIZZLES
        swizzle_epi_configs = []
        for swizzle in valid_swizzles:
            for epi_stages in self.VALID_EPILOGUE_STAGE:
                swizzle_epi_configs.append((best_td, SwizzlingFunctor[swizzle], epi_stages))
        best_config = self.profile_top_k_config(swizzle_epi_configs, 1, True)[0]
        key = self.key_with_epilogue
        self.insert_record(key, 0, best_config)
        return best_config

    
    
