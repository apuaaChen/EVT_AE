# Generate the code from parameter.
import pycutlass
from pycutlass import *
import cutlass


# Generate the CUDA code following a configuration template
# Input:
#   config_template: a string with a few "%s" to be filled.
#   parameter: a vector of positive integers. This instantiates the CUDA configuration.
#           The length of this vector must be exactly the same as the number of "%s" in config_template
#  
def generate_code(
    parameter, element_a, layout_a, element_b, layout_b, element_c, layout_c, element_accumulator):

    math_inst = MathInstruction(
        instruction_shape=[16, 8, 16],
        element_a=element_a, element_b=element_b,
        element_accumulator=element_accumulator,
        opcode_class=cutlass.OpClass.TensorOp
    )

    threadblock_shape = parameter[0:3]
    threadblock_shape = [int(t) for t in threadblock_shape]
    warp_shape = parameter[3:6]
    warp_count = [t // w for t, w in zip(threadblock_shape, warp_shape)]
    warp_count = [int(t) for t in warp_count]
    stages = int(parameter[6])

    tile_description = TileDescription(
        threadblock_shape, stages, warp_count, math_inst
    )

    A = TensorDescription(element_a, layout_a, 8)
    B = TensorDescription(element_b, layout_b, 8)
    C = TensorDescription(element_c, layout_c, 8)

    epilogue_functor = LinearCombination(
        element_output=C.element, epilogue_vector_length=C.alignment,
        element_accumulator=math_inst.element_accumulator,
        element_epilogue=cutlass.float32
    )

    log_swizzle = parameter[7]
    swizzling_functor = getattr(cutlass, "IdentitySwizzle%d"%int(pow(2, log_swizzle)))

    operation = GemmOperationUniversal(
        arch=80, tile_description=tile_description,
        A=A, B=B, C=C, epilogue_functor=epilogue_functor,
        swizzling_functor=swizzling_functor
    )

    # print(operation.rt_module.emit())

    pycutlass.compiler.add_module([operation])

    return operation

# for test only
if __name__ == "__main__":
    pycutlass.get_memory_pool(init_pool_size=2**10, max_pool_size=2**32)
    pycutlass.compiler.nvcc()
    # Generate configuration code
    operation = generate_code(
        parameter=[128, 16, 64, 64, 16, 64, 2, 3], element_a=cutlass.float16,
        layout_a=cutlass.RowMajor, element_b=cutlass.float16, layout_b=cutlass.ColumnMajor,
        element_c=cutlass.float16, layout_c=cutlass.RowMajor, element_accumulator=cutlass.float32
    )
    input_shape = (3584, 32320, 1024)

    M, N, K = input_shape

    tensor_A = torch.empty(size=(M * K,), dtype=torch.float16, device="cuda")
    tensor_B = torch.empty(size=(N * K,), dtype=torch.float16, device="cuda")
    tensor_C = torch.empty(size=(M * N,), dtype=torch.float16, device="cuda")

    arguments = GemmArguments(
        operation=operation, problem_size=cutlass.gemm.GemmCoord(M, N, K),
        A=tensor_A, B=tensor_B, C=tensor_C, D=tensor_C, 
        output_op = operation.epilogue_type(1.0, 0.0),
        gemm_mode=cutlass.gemm.Mode.Gemm, split_k_slices=1
    )
    
    # warmup iterations
    for _ in range(10):
        operation.run(arguments)