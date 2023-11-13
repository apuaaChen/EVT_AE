import ctypes
import torch
from cutlass.backend import TorchFrontend, ExecutableOperation, LaunchConfiguration, MatrixCoord
from cutlass.backend.evt import EpilogueFunctorVisitor
from cuda import cuda

################################################################################
#
# Data structure modeling a Reduction Apply
#
################################################################################

class ReduceApplyArguments:
    def __init__(
        self, operation: 'ReduceApplyOperation', 
        problem_size: 'list[int]', **kwargs) -> None:
        self.operation = operation
        self.problem_size = problem_size

    def get_arguments(self):
        raise NotImplementedError("function get_arguments is not overrided by child class")

    def initialize(self):
        # get launch configuration
        launch_config = self.operation.rt_module.plan(self)

        self.get_arguments()

        res_args = self.operation.rt_module.get_args(ctypes.byref(self.arguments))
        self.host_workspace = bytearray(res_args.contents)
        self.device_workspace = None
        self.launch_config = launch_config
    
    @staticmethod
    def tensor2ptr(x: torch.Tensor):
        return TorchFrontend.argument(x)


class ReduceApplyRT(ExecutableOperation):
    """
    ReduceApplyRT manages the CUTLASS runtime components
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
    def __init__(self, operation: 'ReduceApplyOperation'):
        super().__init__(operation)

        self.argument_type, self.epilogue_type = self.get_arguments(operation.epilogue_functor)
        self.argtype = [
            ctypes.POINTER(self.argument_type)
        ]
        self.num_threads = operation.num_threads
        self.rows_per_cta = self.operation.rows_per_cta
    
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
        grid_x = int((argument.problem_size.row + self.rows_per_cta - 1)/self.rows_per_cta)

        return LaunchConfiguration(
            [grid_x, 1, 1],
            [self.num_threads, 1, 1],
            self.shared_memory_capacity
        )
    
    #
    @staticmethod
    def get_arguments(epilogue_functor):
        raise NotImplementedError("The function is not overwritten by children class")


class ReduceApplyOperation:
    """
    CUTLASS Reduce Apply Operation
    """

    def __init__(self, rows_per_cta, num_columns, warp_count, alignment, 
                 element_accumulator, epilogue_visitor) -> None:

        self.arch = 80
        self.tile_description = None
        self.epilogue_functor = EpilogueFunctorVisitor(self.arch, epilogue_visitor)
        self.num_threads = warp_count * 32
        self.rows_per_cta = rows_per_cta
        self.column_per_cta = int((num_columns + self.num_threads * alignment - 1)/(self.num_threads * alignment)) * (self.num_threads * alignment)
        self.threadblock_shape = MatrixCoord(rows_per_cta, self.column_per_cta)
        self.element_accumulator = element_accumulator

    
    def procedural_name(self):
        raise NotImplementedError("procedural name")
    
    def run(self, arguments: ReduceApplyArguments, stream=cuda.CUstream(0)) -> cuda.CUresult:
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
