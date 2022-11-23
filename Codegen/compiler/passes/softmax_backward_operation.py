import pycutlass
from pycutlass import *
import os
import nvtx


################################################################################
#
# Data structure modeling a GEMM operation
#
################################################################################

def get_softmax_backward_arguments(epilogue_functor):

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

class SoftmaxBackwardArguments:
    def __init__(
        self, operation: 'SoftmaxBackwardOperation', problem_size: 'list[int]',
        o_softmax: 'Tensor', grad_softmax: 'Tensor', output_op, **kwargs) -> None:
        # get pointers
        assert isinstance(o_softmax, torch.Tensor)
        assert isinstance(grad_softmax, torch.Tensor)

        self.ptr_o_softmax = TorchFrontend.argument(o_softmax)
        self.ptr_grad_softmax = TorchFrontend.argument(grad_softmax)
        self.output_op = output_op

        self.operation = operation
        self.problem_size = problem_size

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

    def initialize(self):
        # get launch configuration
        launch_config = self.operation.rt_module.plan(self)

        self.get_arguments()

        res_args = self.operation.rt_module.get_args(ctypes.byref(self.arguments))
        self.host_workspace = bytearray(res_args.contents)
        self.device_workspace = None
        self.launch_config = launch_config


class SoftmaxBackwardRT(ExecutableOperation):
    """
    SoftmaxBackwardRT manages the CUTLASS runtime components
    """

    HostTemplate = r'''
extern "C" {
    // Get the size of params in bytes
    int ${operation_name}_get_param_size(){
        return sizeof(${operation_name}${operation_suffix}::Params);
    }

    // Get the size of dynamic shared memory in bytes
    int ${operation_name}_shared_memory_size() {
        return int(sizeof(${operation_name}${operation_suffix}::SharedStorage));
    }

    // Get the params as byte array
    char* ${operation_name}_get_params(${operation_name}_base::Arguments* argument) {
        ${operation_name}_base::Params* params;
        params = new ${operation_name}_base::Params(*argument);

        char *bytes = ((char*)(params));
        char *output = new char[sizeof(${operation_name}_base::Params)];
        for (unsigned int i = 0; i < sizeof(${operation_name}_base::Params); i ++)
            output[i] = bytes[i];
        return output;
    }
}
    '''

    KernelTemplate = r'''
extern "C"
__global__ void
${operation_name}(${operation_name}${operation_suffix}::Params params) {

    // Dynamic shared memory base pointer
    extern __shared__ int SharedStorageBase[];

    // Declare pointer to dynamic shared memory
    ${operation_name}${operation_suffix}::SharedStorage *shared_storage = 
        reinterpret_cast<${operation_name}${operation_suffix}::SharedStorage *>(SharedStorageBase);
    
    ${operation_name}${operation_suffix} op;

    op(params, *shared_storage);
}
    '''
    def __init__(self, operation: 'SoftmaxBackwardOperation'):
        super().__init__(operation)

        self.emitter = EmitSoftmaxBackwardUniversalInstance('_type')

        self.argument_type, self.epilogue_type = get_softmax_backward_arguments(operation.epilogue_functor)
        self.argtype = [
            ctypes.POINTER(self.argument_type)
        ]
        self.num_threads = operation.warp_count_column * operation.warp_count_row * 32
    
    #
    def emit(self):
        return self.emitter.emit(self.operation)

    #
    def initialize(self):
        err, = cuda.cuFuncSetAttribute(
            self.kernel,
            attrib=cuda.CUfunction_attribute.CU_FUNC_ATTRIBUTE_MAX_DYNAMIC_SHARED_SIZE_BYTES,
            value=self.shared_memory_capacity)
        if err != cuda.CUresult.CUDA_SUCCESS:
            raise RuntimeError('Cuda Error: {}'.format(err))
    
    #
    def plan(self, argument):
        grid_x = argument.problem_size[0]

        return LaunchConfiguration(
            [grid_x, 1, 1],
            [self.num_threads, 1, 1],
            self.shared_memory_capacity
        )


class SoftmaxBackwardOperation:
    """
    CUTLASS Softmax Operation
    """

    def __init__(
        self, input: 'TensorDescription', output: 'TensorDescription',
        threadblock_tile: 'list[int]', warp_count: 'list[int]',
        element_accumulator, epilogue_functor) -> None:


        self.arch = 80
        self.tile_description = None
        self.epilogue_functor = epilogue_functor

        self.element_input = input.element
        self.element_output = output.element
        self.element_accumulator = element_accumulator

        self.threadblock_row = threadblock_tile[0]
        self.threadblock_column = threadblock_tile[1]

        self.warp_count_row = warp_count[0]
        self.warp_count_column = warp_count[1]
        if self.warp_count_column == 1:
            self.mode = "Warp"
        else:
            self.mode = "Block"

        self.alignment_input = input.alignment
        self.alignment_output = output.alignment

        self.rt_module = SoftmaxBackwardRT(self)

        self.argument_type = self.rt_module.argument_type
        self.epilogue_type = self.rt_module.epilogue_type

    
    def procedural_name(self):
        return "softmax_kernel"
    
    def run(self, arguments: SoftmaxBackwardArguments, stream=cuda.CUstream(0)) -> cuda.CUresult:
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
            "softmax/kernel/default_softmax_backward_universal.h",
            "softmax/kernel/softmax_backward_universal_with_visitor.h",
            "softmax/epilogue/epilogue_backward_with_visitor.h"
        ]

        self.cutlass_template_visitor = """
using ${operation_name}_default =
    typename cutlass::softmax::kernel::DefaultSoftmaxBackwardUniversal<
        cutlass::MatrixShape<${threadblock_row}, ${threadblock_column}>,
        cutlass::MatrixShape<${warp_count_row}, ${warp_count_column}>,
        ${element_input}, 
        ${alignment_input},
        ${element_output}, 
        ${alignment_output},
        ${element_accumulator}>::SoftmaxKernel;

${epilogue_visitor}

// debug

using ${operation_name}_Epilogue = typename cutlass::softmax::threadblock::EpilogueBackwardWithVisitorFromExistingEpilogue<
    ${operation_name}_EpilogueVisitor,
    typename ${operation_name}_default::Epilogue>::Epilogue;

/// using ${operation_name}_base = ${operation_name}_default;
using ${operation_name}_base = 
    cutlass::softmax::kernel::SoftmaxBackwardUniversalwithEpilogueVisitor${Mode}<
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