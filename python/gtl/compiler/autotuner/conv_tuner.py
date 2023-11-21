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
# Autotuner for CONV EVT
from cutlass.shape import GemmCoord
from gtl.compiler.autotuner.tuner_base import AutoTunerBase
import torch
from gtl.compiler.autotuner.gemm_tuner import MMTuner
from cutlass.shape import Conv2DProblemSize
from cutlass.utils.datatypes import torch_type
from cutlass import SwizzlingFunctor
from gtl.compiler.passes.pass_decomposition import convolution_forward_channel_last, convolution_backward_data_channel_last


class ConvFpropTuner(MMTuner):
    VALID_SWIZZLES = ["Identity1", "Identity2", "Identity4", "Identity8"]
    VALID_EPILOGUE_STAGE = [1, 2]
    table_name = "best_config_conv_fprop"
    config_columns = MMTuner.config_columns
    def __init__(self, plan, problem_size: Conv2DProblemSize, epilogue_visitor) -> None:
        super(MMTuner, self).__init__(epilogue_visitor, convolution_forward_channel_last)
        self.plan = plan
        self.ps = problem_size
        self.dtype_A = torch_type(self.plan._element_a)
        self.dtype_B = torch_type(self.plan._element_b)
        self.dtype_C = torch_type(self.plan._element_c)

        self.valid_tds = []
        for td in self.plan.tile_descriptions():
            if td not in self.valid_tds:
                self.valid_tds.append(td)
    
    @property
    def key_no_epilogue(self):
        return f"{self.ps.N}_{self.ps.H}_{self.ps.W}_{self.ps.C}_{self.ps.K}_{self.ps.R}_{self.ps.S}_{self.ps.pad_h}_{self.ps.pad_w}_{self.ps.stride_h}_{self.ps.stride_w}"
    
    def get_arguments_no_epilogue(self):
        tensor_A = torch.empty(
            (self.ps.N, self.ps.C, self.ps.H, self.ps.W), dtype=self.dtype_A, device="cuda").to(memory_format=torch.channels_last)
        tensor_B = torch.empty(
            (self.ps.K, self.ps.C, self.ps.R, self.ps.S), dtype=self.dtype_A, device="cuda").to(memory_format=torch.channels_last)
        tensor_C = torch.empty(
            (self.ps.N, self.ps.K, self.ps.P, self.ps.Q), dtype=self.dtype_A, device="cuda").to(memory_format=torch.channels_last)
        tensor_D = torch.empty_like(tensor_C).to(memory_format=torch.channels_last)

        stride = [self.ps.stride_h, self.ps.stride_w]
        padding = [self.ps.pad_h, self.ps.pad_w]
        dilation = [self.ps.dilation_h, self.ps.dilation_w]
        return [tensor_A, tensor_B, tensor_C, tensor_D, stride, padding, dilation, 1.0, 1.0], {}
    
    def profile_best_config_without_epilogue(self):
        configs = [(td, SwizzlingFunctor.Identity1, 1) for td in self.valid_tds]
        best_configs = self.profile_top_k_config(configs, self.num_best_tds, False)
        key = self.key_no_epilogue
        for rank, config in enumerate(best_configs):
            self.insert_record(key, rank, config)
        return best_configs
    
    def profile_best_config_with_epilogue(self, configs):
        best_config = self.profile_top_k_config(configs, 1, True)[0]
        best_td, _, _ = best_config
        valid_swizzles = self.VALID_SWIZZLES
        swizzle_epi_configs = []
        for swizzle in valid_swizzles:
            for epi_stages in self.VALID_EPILOGUE_STAGE:
                swizzle_epi_configs.append((best_td, SwizzlingFunctor[swizzle], epi_stages))
        best_config = self.profile_top_k_config(swizzle_epi_configs, 1, True)[0]
        key = self.key_with_epilogue
        self.insert_record(key, 0, best_config)
        return best_config


class ConvDgradTuner(MMTuner):
    VALID_SWIZZLES = ["Identity1", "Identity2", "Identity4", "Identity8"]
    VALID_EPILOGUE_STAGE = [1, 2]
    table_name = "best_config_conv_dgrad"
    config_columns = MMTuner.config_columns
    def __init__(self, plan, problem_size: Conv2DProblemSize, epilogue_visitor) -> None:
        super(MMTuner, self).__init__(epilogue_visitor, convolution_backward_data_channel_last)
        self.plan = plan
        self.ps = problem_size
        self.dtype_A = torch_type(self.plan._element_a)
        self.dtype_B = torch_type(self.plan._element_b)
        self.dtype_C = torch_type(self.plan._element_c)

        self.valid_tds = []
        for td in self.plan.tile_descriptions():
            if td not in self.valid_tds:
                self.valid_tds.append(td)
    
    @property
    def key_no_epilogue(self):
        return f"{self.ps.N}_{self.ps.H}_{self.ps.W}_{self.ps.C}_{self.ps.K}_{self.ps.R}_{self.ps.S}_{self.ps.pad_h}_{self.ps.pad_w}_{self.ps.stride_h}_{self.ps.stride_w}"
    
    def get_arguments_no_epilogue(self):
        tensor_A = torch.empty(
            (self.ps.N, self.ps.K, self.ps.P, self.ps.Q), dtype=self.dtype_A, device="cuda").to(memory_format=torch.channels_last)
        tensor_B = torch.empty(
            (self.ps.K, self.ps.C, self.ps.R, self.ps.S), dtype=self.dtype_A, device="cuda").to(memory_format=torch.channels_last)
        tensor_C = torch.empty(
            (self.ps.N, self.ps.C, self.ps.H, self.ps.W), dtype=self.dtype_A, device="cuda").to(memory_format=torch.channels_last)
        tensor_D = torch.empty_like(tensor_C).to(memory_format=torch.channels_last)

        stride = [self.ps.stride_h, self.ps.stride_w]
        padding = [self.ps.pad_h, self.ps.pad_w]
        dilation = [self.ps.dilation_h, self.ps.dilation_w]
        return [tensor_A, tensor_B, tensor_C, tensor_D, stride, padding, dilation, 1.0, 1.0], {}
    
    def profile_best_config_without_epilogue(self):
        if self.ps.stride_h == 1 and self.ps.stride_w == 1:
            swizzling_functor = SwizzlingFunctor.Identity1
        else:
            swizzling_functor = SwizzlingFunctor.StridedDgradIdentity1
        configs = [(td, swizzling_functor, 1) for td in self.valid_tds]
        best_configs = self.profile_top_k_config(configs, self.num_best_tds, False)
        key = self.key_no_epilogue
        for rank, config in enumerate(best_configs):
            self.insert_record(key, rank, config)
        return best_configs
    
    def profile_best_config_with_epilogue(self, configs):
        best_config = self.profile_top_k_config(configs, 1, True)[0]
        best_td, swizzle, _ = best_config
        if self.ps.stride_h == 1 and self.ps.stride_w == 1:
            valid_swizzles = self.VALID_SWIZZLES
        else:
            valid_swizzles = ["StridedDgradIdentity1", "StridedDgradIdentity4"]
        swizzle_epi_configs = []
        for swizzle in valid_swizzles:
            for epi_stages in self.VALID_EPILOGUE_STAGE:
                swizzle_epi_configs.append((best_td, SwizzlingFunctor[swizzle], epi_stages))
        best_config = self.profile_top_k_config(swizzle_epi_configs, 1, True)[0]
        key = self.key_with_epilogue
        self.insert_record(key, 0, best_config)
        return best_config
