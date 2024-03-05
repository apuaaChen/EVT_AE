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
# Autotuner for Reduce-apply operation
from gtl.compiler.autotuner.tuner_base import AutoTunerBase
from gtl.ops.softmax import SoftmaxOperation, SoftmaxArguments
from gtl.ops.softmax_backward import SoftmaxBackwardOperation, SoftmaxBackwardArguments
from gtl.ops.layernorm import LayerNormOperation, LayerNormArguments
from gtl.ops.layernorm_backward import LayerNormBackwardOperation, LayerNormBackwardArguments
from cutlass.shape import MatrixCoord
from cutlass import Tensor, LayoutType, TensorDescription
import cutlass
import torch
from cutlass.backend import compiler
from cutlass.backend.compiler import CompilationOptions
from cutlass.backend.utils.datatypes import torch_to_cutlass
import operator


class ReduceApplyTuner(AutoTunerBase):
    VALID_WARP_COUNT = [1, 2, 4, 8]
    VALID_ROWS_PER_CTA = [1, 2, 4, 8]
    VALID_CACHE_INPUT = [0, 1]
    config_columns = {
        "rows_per_cta": "INTEGER",
        "warp_count": "INTEGER",
        "cache_input": "INTEGER"
    }
    def __init__(self, epilogue_visitor, target, problem_size: MatrixCoord, dtype) -> None:
        super().__init__(epilogue_visitor, target)
        self.ps = problem_size
        self.dtype = dtype
        self.alignment = self.get_alignment(self.ps.column)
        # To ensure utilization
        self.max_rows_per_cta = max(1, int(self.ps.row / 108))
        self.max_warp_count = max(1, int(self.ps.column / 32 / self.alignment))

        # Construct the naive epilogue visitor
        def naive_epilogue(accum):
            out = accum
            return out
        
        example_tensors = {
            "accum": Tensor(
                element=dtype, shape=(self.ps.row, 1, self.ps.column), 
                layout_tag=LayoutType.RowMajor),
            "out": Tensor(
                element=dtype, shape=(self.ps.row, 1, self.ps.column), 
                layout_tag=LayoutType.RowMajor),
        }

        self.naive_epilogue_visitor = cutlass.epilogue.trace(naive_epilogue, example_tensors)
        self.naive_visitor_args = {
            "out": torch.empty((self.ps.row, 1, self.ps.column))
        }
    
    def construct_config_with_record(self, row):
        _,_,rows_per_cta, warp_count, cache_input = row
        return (rows_per_cta, warp_count, cache_input)
    
    def construct_record_with_config(self, config):
        rows_per_cta, warp_count, cache_input = config
        return (rows_per_cta, warp_count, cache_input)
    
    @property
    def key_no_epilogue(self):
        """
        The key is {num_row}_{num_columns}
        """
        return f"{self.ps.row, self.ps.column}"

    def profile_best_config_without_epilogue(self):
        configs = []
        for cache_input in self.VALID_CACHE_INPUT:
            for rows_per_cta in self.VALID_ROWS_PER_CTA:
                for warp_count in self.VALID_WARP_COUNT:
                    if rows_per_cta > self.max_rows_per_cta:
                        continue
                    if warp_count > self.max_warp_count:
                        continue
                    configs.append([rows_per_cta, warp_count, cache_input])
        best_configs = self.profile_top_k_config(configs, self.num_best_tds, False)
        key = self.key_no_epilogue
        for rank, config in enumerate(best_configs):
            self.insert_record(key, rank, config)
        return best_configs
    
    def compile_operation(self, operation):
        MLCOMPILER_SRC_DIR = '/workspace/EVT_AE/src/cuda'
        include_paths = [
            MLCOMPILER_SRC_DIR
        ] + compiler.default_include_paths

        compile_options = CompilationOptions(
            compiler.default_compile_options, 80, include_paths=include_paths
        )
        compiler.add_module([operation], compile_options)

    def profile_best_config_with_epilogue(self, configs):
        best_config = self.profile_top_k_config(configs, 1, True)[0]
        key = self.key_with_epilogue
        self.insert_record(key, 0, best_config)
        return best_config


################################################################################
# Derived for Softmax Operation
################################################################################
class SoftmaxTuner(ReduceApplyTuner):
    table_name = "best_config_softmax"
    def __init__(self, epilogue_visitor, problem_size: MatrixCoord, dtype) -> None:
        super().__init__(epilogue_visitor, torch.ops.aten._softmax, problem_size, dtype)
    
    def profile_with_config(self, config, with_epilogue, *args, **kwargs):
        rows_per_cta, warp_count, cache_input = config
        if with_epilogue:
            visitor_args = kwargs["visitor_args"]
            epilogue_visitor = self.epilogue_visitor
            
        else:
            visitor_args = self.naive_visitor_args
            epilogue_visitor = self.naive_epilogue_visitor
        
        operation = SoftmaxOperation(
            input=TensorDescription(torch_to_cutlass(self.dtype), LayoutType.RowMajor, self.alignment),
            rows_per_cta=rows_per_cta, num_columns=self.ps.column, num_rows=self.ps.row,
            warp_count=warp_count, epilogue_visitor=epilogue_visitor, cache_input=cache_input
        )
        self.compile_operation(operation)
        
        input = args[0]
        arguments = SoftmaxArguments(
            operation, self.ps, input,
            operation.epilogue_type(visitor_args)
        )
        return self.torch_profiler(operation.run, arguments)

    def get_arguments_no_epilogue(self):
        input = torch.empty((self.ps.row, 1, self.ps.column), dtype=self.dtype, device="cuda")
        return [input], {}


################################################################################
# Derived for Softmax Backward Operation
################################################################################
class SoftmaxBackwardTuner(ReduceApplyTuner):
    table_name = "best_config_softmax_backward"
    def __init__(self, epilogue_visitor, problem_size: MatrixCoord, dtype) -> None:
        super().__init__(epilogue_visitor, torch.ops.aten._softmax_backward_data, problem_size, dtype)
    
    def profile_with_config(self, config, with_epilogue, *args, **kwargs):
        rows_per_cta, warp_count, cache_input = config
        if with_epilogue:
            visitor_args = kwargs["visitor_args"]
            epilogue_visitor = self.epilogue_visitor
            
        else:
            visitor_args = self.naive_visitor_args
            epilogue_visitor = self.naive_epilogue_visitor
        
        operation = SoftmaxBackwardOperation(
            input=TensorDescription(torch_to_cutlass(self.dtype), LayoutType.RowMajor, self.alignment),
            rows_per_cta=rows_per_cta, num_columns=self.ps.column, num_rows=self.ps.row,
            warp_count=warp_count, epilogue_visitor=epilogue_visitor, cache_input=cache_input
        )
        self.compile_operation(operation)
        
        grad, softmax = args
        arguments = SoftmaxBackwardArguments(
            operation, self.ps,
            grad, softmax, operation.epilogue_type(visitor_args)
        )
        return self.torch_profiler(operation.run, arguments)
    
    def get_arguments_no_epilogue(self):
        grad = torch.empty((self.ps.row, 1, self.ps.column), dtype=self.dtype, device="cuda")
        softmax = torch.empty_like(grad)
        return [grad, softmax], {}

################################################################################
# Derived for Layernorm Forward
################################################################################
class LayerNormTuner(ReduceApplyTuner):
    table_name = "best_config_layernorm"
    def __init__(self, epilogue_visitor, problem_size: MatrixCoord, dtype) -> None:
        super().__init__(epilogue_visitor, torch.ops.aten.native_layer_norm, problem_size, dtype)
    
    def profile_with_config(self, config, with_epilogue, *args, **kwargs):
        rows_per_cta, warp_count, cache_input = config
        if with_epilogue:
            visitor_args = kwargs["visitor_args"]
            epilogue_visitor = self.epilogue_visitor
            
        else:
            visitor_args = self.naive_visitor_args
            epilogue_visitor = self.naive_epilogue_visitor
        
        operation = LayerNormOperation(
            input=TensorDescription(torch_to_cutlass(self.dtype), LayoutType.RowMajor, self.alignment),
            rows_per_cta=rows_per_cta, num_columns=self.ps.column, num_rows=self.ps.row,
            warp_count=warp_count, epilogue_visitor=epilogue_visitor, cache_input=cache_input
        )
        self.compile_operation(operation)

        input, mean, std = args
        arguments = LayerNormArguments(
            operation, self.ps, input, mean, std,
            operation.epilogue_type(visitor_args)
        )
        return self.torch_profiler(operation.run, arguments)

    def get_arguments_no_epilogue(self):
        input = torch.empty((self.ps.row, 1, self.ps.column), dtype=self.dtype, device="cuda")
        mean = torch.empty((self.ps.row, 1, 1), dtype=torch.float32, device="cuda")
        std = torch.empty((self.ps.row, 1, 1), dtype=torch.float32, device="cuda")
        return [input, mean, std], {}
    
    def get_output_name(self, output):
        if output.target == operator.getitem and output.args[0].target == self.target and output.args[1] == 0:
            return "accum"
        return super().get_output_name(output)

################################################################################
# Derived for Layernorm Backward
################################################################################
class LayerNormBackwardTuner(ReduceApplyTuner):
    table_name = "best_config_layernorm_backward"
    def __init__(self, epilogue_visitor, problem_size: MatrixCoord, dtype) -> None:
        super().__init__(epilogue_visitor, torch.ops.aten.native_layer_norm_backward, problem_size, dtype)
    
    def profile_with_config(self, config, with_epilogue, *args, **kwargs):
        rows_per_cta, warp_count, cache_input = config
        if with_epilogue:
            visitor_args = kwargs["visitor_args"]
            epilogue_visitor = self.epilogue_visitor
            
        else:
            visitor_args = self.naive_visitor_args
            epilogue_visitor = self.naive_epilogue_visitor
        
        operation = LayerNormBackwardOperation(
            input=TensorDescription(torch_to_cutlass(self.dtype), LayoutType.RowMajor, self.alignment),
            rows_per_cta=rows_per_cta, num_columns=self.ps.column, num_rows=self.ps.row,
            warp_count=warp_count, epilogue_visitor=epilogue_visitor, cache_input=cache_input
        )
        self.compile_operation(operation)

        grad, x, mean, invstd, gamma = args

        arguments = LayerNormBackwardArguments(
            operation, self.ps, gamma, grad, x, mean, invstd,
            operation.epilogue_type(visitor_args)
        )

        return self.torch_profiler(operation.run, arguments)

    def get_arguments_no_epilogue(self):
        grad = torch.empty((self.ps.row, 1, self.ps.column), dtype=self.dtype, device="cuda")
        x = torch.empty_like(grad)
        mean = torch.empty((self.ps.row, 1, 1), dtype=torch.float32, device="cuda")
        invstd = torch.empty((self.ps.row, 1, 1), dtype=torch.float32, device="cuda")
        gamma = torch.empty((self.ps.column,), dtype=self.dtype, device="cuda")
        return [grad, x, mean, invstd, gamma], {}
