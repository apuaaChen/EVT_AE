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
from typing import Any
import torch
from cutlass.backend import TensorDescription
from cutlass.backend import MatrixCoord_, DataTypeTag, SubstituteTemplate, DataType
from gtl.ops.reduce_apply_operation import ReduceApplyArguments, ReduceApplyRT, ReduceApplyOperation
from cutlass.backend.c_types import tuple_factory
import ctypes

################################################################################
#
# Data structure modeling a Softmax operation
#
################################################################################


class SoftmaxArguments(ReduceApplyArguments):
    def __init__(
        self, operation: ReduceApplyOperation, problem_size: 'list[int]', 
        input: torch.Tensor, output_op, **kwargs) -> None:
        super().__init__(operation, problem_size, **kwargs)

        self.ptr_input = self.tensor2ptr(input)
        self.output_op = output_op
        if "batch_count" in kwargs:
            self.batch_count = kwargs["batch_count"]
        else:
            self.batch_count = 1

        self.initialize()
    
    def get_arguments(self):

        self.arguments = self.operation.argument_type(
            self.ptr_input,
            self.problem_size,
            self.batch_count,
            self.output_op
        )


class SoftmaxRT(ReduceApplyRT):
    def __init__(self, operation: 'ReduceApplyOperation'):
        super().__init__(operation)
        self.emitter = EmitSoftmaxUniversalInstance('_type')
    
    @staticmethod
    def get_arguments(epilogue_functor):
        _EpilogueOutputOpParams = epilogue_functor.epilogue_type

        stride_example = (10, 1, 10)
        stride_type = tuple_factory(stride_example, "int64_t")

        class _SoftmaxArguments(ctypes.Structure):
            _fields_ = [
                # Mainloop Args
                ("ptr_input", ctypes.c_void_p),
                ("stride_input", stride_type),
                ("problem_size", MatrixCoord_),
                ("batch_count", ctypes.c_int32),
                ("epilogue_args", _EpilogueOutputOpParams)
            ]
            def __init__(self, ptr_input, problem_size, batch_count, epilogue_args) -> None:
                self.ptr_input = int(ptr_input)
                self.problem_size = MatrixCoord_(problem_size.row, problem_size.column)
                self.batch_count = batch_count
                self.epilogue_args = epilogue_args
                self.stride_input = stride_type((problem_size.column, 1, problem_size.row * problem_size.column))

        return _SoftmaxArguments, _EpilogueOutputOpParams


class SoftmaxOperation(ReduceApplyOperation):
    def __init__(
        self, input: TensorDescription, num_columns, num_rows,
        rows_per_cta, warp_count: int,
        epilogue_visitor, element_accumulator=DataType.f32, cache_input=True) -> None:

        self.element = input.element
        self.alignment = input.alignment
        self.cache_input = cache_input
        if warp_count == -1:
            warp_count = self.propose_warp_count(self.alignment, num_columns)
        if rows_per_cta == -1:
            rows_per_cta = self.propose_rows_per_cta(num_rows)

        super().__init__(rows_per_cta, num_columns, warp_count, self.alignment, element_accumulator, epilogue_visitor)

        self.rt_module = SoftmaxRT(self)
        self.argument_type = self.rt_module.argument_type
        self.epilogue_type = self.rt_module.epilogue_type
    
    def procedural_name(self):
        return "softmax_kernel"
    
    def propose_warp_count(self, alignment, num_columns):
        maximum_columns_per_warp = 32 * alignment * 16
        num_warps = int((num_columns + maximum_columns_per_warp - 1) / maximum_columns_per_warp)
        return min(num_warps, 16)
    
    def propose_rows_per_cta(self, rows):
        if rows / 81 >= 8:
            return 8
        elif rows / 81 >= 4:
            return 4
        elif rows / 81 >= 2:
            return 2
        else:
            return 1

    

class EmitSoftmaxUniversalInstance:
    def __init__(self, operation_suffix='') -> None:
        self.operation_suffix = operation_suffix
        self.includes = [
            # "softmax/fake_device.h",
            "cutlass/cutlass.h",
            "cutlass/matrix_shape.h",
            "cutlass/numeric_types.h",
            "cutlass/arch/arch.h",
            "cutlass/arch/mma.h",
            "cutlass/layout/matrix.h",
            "epilogue/threadblock/visitor.hpp",
            "epilogue/threadblock/visitor_load.hpp",
            "epilogue/threadblock/visitor_store.hpp",
            "epilogue/threadblock/visitor_compute.hpp",
            "softmax/threadblock/softmax_reduction.h",
            "reduce_apply/kernel/reduce_apply_with_callbacks.h",
            "reduce_apply/threadblock/reduction_base.h"
        ]

        self.cutlass_template_visitor = """

using OutputTileThreadMap = cutlass::reduce_apply::threadblock::OutputTileThreadLayout1D<
    ${num_threads}, ${element}, ${alignment}, cutlass::MatrixShape<${cta_rows}, ${cta_columns}>>;
        
using ${operation_name}_reduction = 
    cutlass::softmax::threadblock::SoftmaxReduction<
        ${element},
        ${alignment},
        ${element_accumulator},
        OutputTileThreadMap,
        ${cache_input}
    >;

${callback_decl}

// Softmax operator ${operation_name}
using ${operation_name}_base = 
    cutlass::reduce_apply::kernel::ReduceApplyWithCallbacks<
        ${operation_name}_reduction,
        ${callback_name},
        OutputTileThreadMap
    >;

// Define named type
struct ${operation_name}${operation_suffix} :
    public ${operation_name}_base { }; 
"""
    def emit(self, operation):
        callback_name, callback_decl = operation.epilogue_functor.emit(operation)
        cache_input = 'true' if operation.cache_input else 'false'
        values = {
            'num_threads': str(operation.num_threads),
            'element': DataTypeTag[operation.element],
            'alignment': str(operation.alignment),
            'cta_rows': str(operation.threadblock_shape.row),
            'cta_columns': str(operation.threadblock_shape.column),
            'element_accumulator': DataTypeTag[operation.element_accumulator],
            'operation_name': operation.procedural_name(),
            'callback_name': callback_name,
            'callback_decl': callback_decl,
            'operation_suffix': self.operation_suffix,
            'cache_input': cache_input
        }
        
        code = SubstituteTemplate(self.cutlass_template_visitor, values)

        return code