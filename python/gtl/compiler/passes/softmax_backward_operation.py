import pycutlass
from pycutlass import *
import os
import nvtx
from gtl.compiler.passes.reduce_apply_operation import ReduceApplyArguments, ReduceApplyRT, ReduceApplyOperation


################################################################################
#
# Data structure modeling a Softmax Backward operation
#
################################################################################

class SoftmaxBackwardArguments(ReduceApplyArguments):
    def __init__(
        self, operation: 'SoftmaxBackwardOperation', problem_size: 'list[int]',
        o_softmax: 'Tensor', grad_softmax: 'Tensor', output_op, **kwargs) -> None:
        #
        super().__init__(operation, problem_size, **kwargs)
        self.ptr_o_softmax = self.tensor2ptr(o_softmax)
        self.ptr_grad_softmax = self.tensor2ptr(grad_softmax)
        self.output_op = output_op

        self.initialize()
    
    def get_arguments(self):
        self.arguments = self.operation.argument_type(
            int(self.ptr_o_softmax),
            int(self.ptr_grad_softmax),
            MatrixCoord_(self.problem_size[0], self.problem_size[1]),
            int(self.ptr_o_softmax),
            int(self.ptr_grad_softmax),
            MatrixCoord_(self.problem_size[0], self.problem_size[1]),
            self.output_op
        )

class SoftmaxBackwardRT(ReduceApplyRT):
    def __init__(self, operation: 'ReduceApplyOperation'):
        super().__init__(operation)
        self.emitter = EmitSoftmaxBackwardUniversalInstance('_type')
    
    @staticmethod
    def get_arguments(epilogue_functor):
        _EpilogueOutputOpParams = epilogue_functor.epilogue_type

        class _SoftmaxBackwardArguments(ctypes.Structure):
            _fields_ = [
                ("ptr_o_softmax", ctypes.c_void_p),
                ("ptr_grad_softmax", ctypes.c_void_p),
                ("problem_size", MatrixCoord_),
                ("ptr_o_softmax_2", ctypes.c_void_p),
                ("ptr_grad_softmax_2", ctypes.c_void_p),
                ("problem_size_2", MatrixCoord_),
                ("epilogue_args", _EpilogueOutputOpParams)
            ]

        return _SoftmaxBackwardArguments, _EpilogueOutputOpParams


class SoftmaxBackwardOperation(ReduceApplyOperation):
    def __init__(
        self, input: 'TensorDescription', output: 'TensorDescription',
        threadblock_tile: 'list[int]', warp_count: 'list[int]',
        element_accumulator, epilogue_functor) -> None:

        self.element_input = input.element
        self.element_output = output.element

        self.alignment_input = input.alignment
        self.alignment_output = output.alignment

        super().__init__(threadblock_tile, warp_count, element_accumulator, epilogue_functor)

        self.rt_module = SoftmaxBackwardRT(self)

        self.argument_type = self.rt_module.argument_type
        self.epilogue_type = self.rt_module.epilogue_type
    
    def procedural_name(self):
        return "softmax_kernel"


class EmitSoftmaxBackwardUniversalInstance:
    def __init__(self, operation_suffix='') -> None:
        self.operation_suffix = operation_suffix
        self.includes = [
            "softmax/fake_device.h",
            "cutlass/cutlass.h",
            "cutlass/numeric_types.h",
            "cutlass/arch/arch.h",
            "cutlass/arch/mma.h",
            "cutlass/layout/matrix.h",
            "epilogue/epilogue_visitor_generic.h",
            "softmax/kernel/softmax_backward_universal.h",
            "softmax/kernel/reduce_apply_universal_with_visitor.h",
            "softmax/epilogue/epilogue_backward_with_visitor.h"
        ]

        self.cutlass_template_visitor = """
using ${operation_name}_default =
    cutlass::softmax::kernel::SoftmaxBackwardUniversal<
        cutlass::MatrixShape<${threadblock_row}, ${threadblock_column}>,
        cutlass::MatrixShape<${warp_count_row}, ${warp_count_column}>,
        ${element_input}, 
        ${alignment_input},
        ${element_output}, 
        ${alignment_output},
        ${element_accumulator}>;

${epilogue_visitor}

// debug3

using ${operation_name}_Epilogue = typename cutlass::softmax::threadblock::EpilogueBackwardWithVisitorFromExistingEpilogue<
    ${operation_name}_EpilogueVisitor,
    typename ${operation_name}_default::Epilogue>::Epilogue;

/// using ${operation_name}_base = ${operation_name}_default;
using ${operation_name}_base = 
    cutlass::softmax::kernel::ReduceApplywithEpilogueVisitor${Mode}<
        typename ${operation_name}_default::Reduction,
        ${operation_name}_Epilogue
    >;

// Define named type
struct ${operation_name}${operation_suffix} :
    public ${operation_name}_base { };
"""
    def emit(self, operation):
        values = {
            'operation_name': operation.procedural_name(),
            'operation_suffix': self.operation_suffix,
            'threadblock_row': str(operation.threadblock_row),
            'threadblock_column': str(operation.threadblock_column),
            'warp_count_row': str(operation.warp_count_row),
            'warp_count_column': str(operation.warp_count_column),
            'element_input': DataTypeTag[operation.element_input],
            'alignment_input': str(operation.alignment_input),
            'element_output': DataTypeTag[operation.element_output],
            'alignment_output': str(operation.alignment_output),
            'element_accumulator': DataTypeTag[operation.element_accumulator],
            'Mode': operation.mode
        }

        values['epilogue_visitor'] = operation.epilogue_functor.emit(operation)
        
        code =  SubstituteTemplate(self.cutlass_template_visitor, values)

        return code