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
# Placeholder function for spmm node
import torch
import ctypes
from gtl.ops.reduce_apply_operation import ReduceApplyArguments, ReduceApplyOperation, ReduceApplyRT
from cutlass.backend.c_types import tuple_factory
from cutlass.backend import GemmCoord, MatrixCoord_, TensorDescription, MatrixCoord, LaunchConfiguration
from cutlass.backend import MatrixCoord_, DataTypeTag, SubstituteTemplate, DataType
from cutlass.backend.evt import EpilogueFunctorVisitor
from cuda import cuda

################################################################################
#
# Data structure modeling a Softmax operation
#
################################################################################

class SpmmArguments(ReduceApplyArguments):
    def __init__(
        self, operation,
        problem_size: GemmCoord, csr_matrix, B, output_op, **kwargs) -> None:
        super().__init__(operation, problem_size, **kwargs)

        self.operation = operation
        self.problem_size = problem_size
        self.ptr_row = self.tensor2ptr(csr_matrix.crow_indices())
        self.ptr_indices = self.tensor2ptr(csr_matrix.col_indices())
        self.ptr_edge_weight = self.tensor2ptr(csr_matrix.values())
        self.ptr_embedding = self.tensor2ptr(B)
        self.output_op = output_op

        self.initialize()
    
    def get_arguments(self):
        
        self.arguments = self.operation.argument_type(
            int(self.ptr_embedding),
            self.problem_size,
            int(self.ptr_row),
            int(self.ptr_indices),
            int(self.ptr_edge_weight),
            self.output_op
        )

class SpmmRT(ReduceApplyRT):
    """
    SpmmRT manages the CUTLASS runtime components
    """
    def __init__(self, operation: ReduceApplyOperation):
        super().__init__(operation)
        self.emitter = EmitSpmmInstance('_type')
    
    @staticmethod
    def get_arguments(epilogue_functor):
        _EpilogueOutputOpParams = epilogue_functor.epilogue_type

        # Stride NK
        stride_example = (1, 10)
        stride_type = tuple_factory(stride_example, "int64_t")
        shape_exmaple = (10, 10)
        shape_type = tuple_factory(shape_exmaple, "int32_t", constants=[])

        class _SpmmArguments(ctypes.Structure):
            _fields_ = [
                # Reduction Args
                ("ptr_embedding", ctypes.c_void_p),
                ("dEmb", stride_type),
                ("sEmb", shape_type),
                ("ptr_row", ctypes.c_void_p),
                ("ptr_indices", ctypes.c_void_p),
                ("ptr_edge_weight", ctypes.c_void_p),
                # Problem size
                ("problem_size", MatrixCoord_),
                # Epilogue
                ("epilogue_args", _EpilogueOutputOpParams)
            ]
            def __init__(
                    self, ptr_embedding, problem_size: GemmCoord,
                    ptr_row, ptr_indices, ptr_edge_weight, epilogue_args) -> None:
                self.ptr_embedding = int(ptr_embedding)
                self.dEmb = stride_type((1, problem_size.n))
                self.sEmb = shape_type((problem_size.n, problem_size.k))
                self.ptr_row = int(ptr_row)
                self.ptr_indices = int(ptr_indices)
                self.ptr_edge_weight = int(ptr_edge_weight)
                self.problem_size = MatrixCoord_(problem_size.m, problem_size.n)
                self.epilogue_args = epilogue_args
        return _SpmmArguments, _EpilogueOutputOpParams

    def plan(self, argument):
        grid_x = int((argument.problem_size.m + self.rows_per_cta - 1)/self.rows_per_cta)

        return LaunchConfiguration(
            [grid_x, 1, 1],
            [self.num_threads, 1, 1],
            self.shared_memory_capacity
        )


class SpmmOperation(ReduceApplyOperation):
    def __init__(
            self, input, element_index, num_columns, num_rows,
            epilogue_visitor,
            element_accumulator=DataType.f32
            ) -> None:

        self.element = input.element
        self.element_index = element_index
        self.alignment = input.alignment

        num_threads_per_row = (num_columns + self.alignment-1) // self.alignment
        assert 32 % num_threads_per_row == 0
        self.rows_per_cta = max(32 // num_threads_per_row, 1)
        self.arch = 80
        self.epilogue_functor = EpilogueFunctorVisitor(self.arch, epilogue_visitor)
        self.num_threads = self.rows_per_cta * num_threads_per_row
        self.column_per_cta = num_threads_per_row * self.alignment
        self.threadblock_shape = MatrixCoord(self.rows_per_cta, self.column_per_cta)
        self.element_accumulator = element_accumulator
        
        self.rt_module = SpmmRT(self)
        self.argument_type = self.rt_module.argument_type
        self.epilogue_type = self.rt_module.epilogue_type

        self.arch = 80
    
    def procedural_name(self):
        return "spmm_kernel"
    
    def run(self, arguments: SpmmArguments, stream=cuda.CUstream(0)) -> cuda.CUresult:
        """
        Configure and launch the cuda kernel with input arguments
        """
        err = self.rt_module.run(
            arguments.host_workspace,
            arguments.device_workspace,
            arguments.launch_config,
            stream)

        if err != cuda.CUresult.CUDA_SUCCESS:
            raise RuntimeError('CUDA Error %s' % str(err))

        return err

class EmitSpmmInstance:
    def __init__(self, operation_suffix='') -> None:
        self.operation_suffix = operation_suffix
        self.includes = [
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
            "spmm/threadblock/spmm_reduction.h",
            "spmm/kernel/spmm_csr_with_callbacks.h",
            "cutlass/epilogue/thread/activation.h"
        ]

        self.cutlass_template_visitor = """

using OutputTileThreadMap = cutlass::spmm::threadblock::OutputTileThreadLayoutSubwarp<
    ${alignment}, cutlass::MatrixShape<${cta_rows}, ${cta_columns}>>;
        
using ${operation_name}_reduction = 
    cutlass::spmm::threadblock::SpmmRowBalanceReduction<
        ${element},
        ${alignment},
        ${element_index},
        ${element_accumulator},
        OutputTileThreadMap
    >;

${callback_decl}

// Softmax operator ${operation_name}
using ${operation_name}_base = 
    cutlass::spmm::kernel::SpmmRowBalanceWithCallbacks<
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
        values = {
            'element': DataTypeTag[operation.element],
            'element_index': DataTypeTag[operation.element_index],
            'alignment': str(operation.alignment),
            'cta_rows': str(operation.threadblock_shape.row),
            'cta_columns': str(operation.threadblock_shape.column),
            'element_accumulator': DataTypeTag[operation.element_accumulator],
            'operation_name': operation.procedural_name(),
            'callback_name': callback_name,
            'callback_decl': callback_decl,
            'operation_suffix': self.operation_suffix
        }
        
        code = SubstituteTemplate(self.cutlass_template_visitor, values)

        return code