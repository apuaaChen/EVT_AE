from cutlass.backend import *

################################################################################
#
# Data structure modeling an SpMM kernel
#
################################################################################

class SpmmArguments:
    def __init__(
        self, operation: 'SpmmOperation',
        problem_size: 'list[int]', csr_matrix, B, output_op) -> None:
        self.operation = operation
        self.problem_size = problem_size
        self.ptr_row = self.tensor2ptr(csr_matrix.crow_indices())
        self.ptr_indices = self.tensor2ptr(csr_matrix.col_indices())
        self.edge = self.tensor2ptr(csr_matrix.values())
        self.ptr_b = self.tensor2ptr(B)
        self.output_op = output_op

        self.initialize()
    
    def get_arguments(self):

        self.arguments = self.operation.argument_type(
            # csr matrix
            int(self.ptr_row),
            int(self.ptr_indices),
            int(self.edge),
            # dense matrix
            int(self.ptr_b),
            # output problem size
            MatrixCoord_(self.problem_size.row(), self.problem_size.column()),
            # epilogue op
            MatrixCoord_(self.problem_size.row(), self.problem_size.column()),
            # epilogue visitor op
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
    
    @staticmethod
    def tensor2ptr(x: 'Tensor'):
        assert isinstance(x, torch.Tensor)
        return TorchFrontend.argument(x)

class SpmmRT(ExecutableOperation):
    """
    SpmmRT manages the CUTLASS runtime components
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
    def __init__(self, operation: 'SpmmOperation'):
        super().__init__(operation)

        self.argument_type, self.epilogue_type = self.get_arguments(operation.epilogue_functor)
        self.argtype = [
            ctypes.POINTER(self.argument_type)
        ]
        self.num_threads = operation.num_threads
        self.threadblock_row = self.operation.threadblock_row
        self.threadblock_column = self.operation.threadblock_column
        self.emitter = EmitSpmmUniversalInstance('_type')
    
    @staticmethod
    def get_arguments(epilogue_functor):
        _EpilogueOutputOpParams = epilogue_functor.epilogue_type

        class _SpmmArgument(ctypes.Structure):
            _fields_ = [
                ("ptr_row", ctypes.c_void_p),
                ("ptr_indices", ctypes.c_void_p),
                ("edge", ctypes.c_void_p),
                ("ptr_b", ctypes.c_void_p),
                ("problem_size", MatrixCoord_),
                ("problem_size2", MatrixCoord_),
                ("epilogue_args", _EpilogueOutputOpParams)
            ]

        return _SpmmArgument, _EpilogueOutputOpParams

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
        grid_x = (argument.problem_size.row() + self.threadblock_row -1) // self.threadblock_row
        grid_y = argument.problem_size.column() / self.threadblock_column

        return LaunchConfiguration(
            [grid_x, grid_y, 1],
            [self.num_threads, 1, 1],
            self.shared_memory_capacity
        )

class SpmmOperation:
    """
    CUTLASS Spmm Operation
    """
    def __init__(
        self, tile_description: TileDescription, 
        element_input, element_accumulator, 
        alignment_emb, alignment_nnz, epilogue_functor) -> None:
        #
        self.element_input = element_input
        self.element_accumulator = element_accumulator
        self.alignment_emb = alignment_emb
        self.alignment_nnz = alignment_nnz
        self.epilogue_functor = epilogue_functor

        self.tile_description = tile_description

        self.threadblock_row = tile_description.threadblock_shape[0]
        self.threadblock_column = tile_description.threadblock_shape[1]

        self.num_threads = self.threadblock_row * self.threadblock_column // self.alignment_emb

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

class EmitSpmmUniversalInstance:
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
            "spmm/kernel/spmm_csr_universal.h",
            "spmm/epilogue/epilogue_with_visitor.h"
        ]

        self.cutlass_template_visitor = """

using ${operation_name}_default =
    cutlass::spmm::kernel::SpmmRowBalanceDefault<
        ${element_input},
        ${element_accumulator},
        ${alignment_emb},
        ${alignment_nnz},
        ${element_index},
        cutlass::MatrixShape<${threadblock_row}, ${threadblock_column}>
    >;

// debug199

${epilogue_visitor}

using ${operation_name}_Epilogue = typename cutlass::spmm::threadblock::EpilogueWithVisitorFromExistingEpilogue<
    ${operation_name}_EpilogueVisitor,
    typename ${operation_name}_default::Epilogue>::Epilogue;

/// using ${operation_name}_base = ${operation_name}_default;
using ${operation_name}_base = 
    cutlass::spmm::kernel::SpmmRowBalancewithEpilogueVisitor<
        ${element_input},
        ${element_accumulator},
        ${alignment_emb},
        ${alignment_nnz},
        ${element_index},
        cutlass::MatrixShape<${threadblock_row}, ${threadblock_column}>,
        ${operation_name}_Epilogue
    >;

// Define named type
struct ${operation_name}${operation_suffix} :
    public ${operation_name}_base { };
"""

    def emit(self, operation):
        values = {
            'operation_name': operation.procedural_name(),
            'element_input': DataTypeTag[operation.element_input],
            'element_accumulator': DataTypeTag[operation.element_accumulator],
            'alignment_emb': str(operation.alignment_emb),
            'alignment_nnz': str(operation.alignment_nnz),
            'element_index': 'int64_t',
            'threadblock_row': str(operation.threadblock_row),
            'threadblock_column': str(operation.threadblock_column),
            'operation_suffix': self.operation_suffix,
        }

        values['epilogue_visitor'] = operation.epilogue_functor.emit(operation)
        
        code =  SubstituteTemplate(self.cutlass_template_visitor, values)
        
        return code