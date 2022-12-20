import pycutlass
from pycutlass import *
import os
import nvtx
from passes.reduce_apply_operation import ReduceApplyArguments, ReduceApplyRT, ReduceApplyOperation


################################################################################
#
# Data structure modeling a Layer Normalization operation
#
################################################################################

class LayerNormArguments(ReduceApplyArguments):
    def __init__(
        self, operation: 'LayerNormOperation', problem_size: 'list[int]', 
        input: 'Tensor', mean: 'Tensor', std: 'Tensor', 
        output_op: 'Tensor', eps=1e-12, **kwargs) -> None:
        super().__init__(operation, problem_size, **kwargs)
        
        self.ptr_input = self.tensor2ptr(input)
        self.ptr_mean = self.tensor2ptr(mean)
        self.ptr_std = self.tensor2ptr(std)
        self.output_op = output_op
        self.eps = eps

        self.initialize()
    
    def get_arguments(self):

        self.arguments = self.operation.argument_type(
            int(self.ptr_input), 
            MatrixCoord_(self.problem_size[0], self.problem_size[1]),
            int(self.ptr_input), 
            float(self.eps),
            MatrixCoord_(self.problem_size[0], self.problem_size[1]),
            int(self.ptr_mean),
            int(self.ptr_std),
            self.output_op
        )

class LayerNormRT(ReduceApplyRT):
    def __init__(self, operation: 'LayerNormOperation'):
        super().__init__(operation)
        self.emitter = EmitLayerNormUniversalInstance('_type')
    
    @staticmethod
    def get_arguments(epilogue_functor):
        _EpilogueOutputOpParams = epilogue_functor.epilogue_type

        class _LayerNormArguments(ctypes.Structure):
            _fields_ = [
                ("ptr_input", ctypes.c_void_p),
                ("problem_size", MatrixCoord_),
                ("ptr_input_2", ctypes.c_void_p),
                ("eps", ctypes.c_float),
                ("problem_size_2", MatrixCoord_),
                ("ptr_mean", ctypes.c_void_p),
                ("ptr_std", ctypes.c_void_p),
                ("epilogue_args", _EpilogueOutputOpParams)
            ]
        
        return _LayerNormArguments, _EpilogueOutputOpParams

class LayerNormOperation(ReduceApplyOperation):
    def __init__(
        self, input: 'TensorDescription', output: 'TensorDescription',
        threadblock_tile: 'list[int]', warp_count: 'list[int]',
        element_accumulator, epilogue_functor) -> None:

        self.element_input = input.element
        self.element_output = output.element
        self.alignment_input = input.alignment
        self.alignment_output = output.alignment

        super().__init__(threadblock_tile, warp_count, element_accumulator, epilogue_functor)

        self.rt_module = LayerNormRT(self)

        self.argument_type = self.rt_module.argument_type
        self.epilogue_type = self.rt_module.epilogue_type
    
    def procedural_name(self):
        return "layernorm_kernel"


class EmitLayerNormUniversalInstance:
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
            "softmax/kernel/default_layernorm_universal.h",
            "softmax/kernel/reduce_apply_universal_with_visitor.h",
            "softmax/epilogue/epilogue_with_visitor.h"
        ]

        self.cutlass_template_visitor = """
using ${operation_name}_default =
    typename cutlass::softmax::kernel::DefaultLayerNormUniversal<
        cutlass::MatrixShape<${threadblock_row}, ${threadblock_column}>,
        cutlass::MatrixShape<${warp_count_row}, ${warp_count_column}>,
        ${element_input}, 
        ${alignment_input},
        ${element_output}, 
        ${alignment_output},
        ${element_accumulator}>::LayerNormKernel;

// debug

${epilogue_visitor}

using ${operation_name}_Epilogue = typename cutlass::softmax::threadblock::LayerNormEpilogueWithVisitorFromExistingEpilogue<
    ${operation_name}_EpilogueVisitor,
    typename ${operation_name}_default::Epilogue>::Epilogue;

// Debug89
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
