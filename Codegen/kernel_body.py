from utils import generate_code, get_kernel_name
import torch

# design_space = {
#     "dtype": [torch.float16],
#     "lhs_format": ["row", "col", "row_sp", "col_sp"],
#     "rhs_format": ["row", "col", "row_sp", "col_sp"],
#     "out_format": ["row_sp", "col_sp"],
#     "epilogue": ["row", "col", "row_sp", "col_sp", "row_sp_meta", "col_sp_meta", "bias", "target"],
#     "TileM": [16, 32, 64, 128, 256],
#     "TileN": [16, 32, 64, 128, 256],
#     "TileK": [64],
#     "wTileM": [16, 32, 64, 128, 256],
#     "wTileN": [16, 32, 64, 128, 256],
#     "wTileK": [64],
#     "Stage": [1, 2, 3, 4, 5],
#     "batched": [True, False]
# }


def header_prologue():
    return """#pragma once

    """


def get_swizzle(code, config):
    code += """
// Define swizzle
using ThreadblockSwizzle = cutlass::gemm::threadblock::GemmBatchedIdentityThreadblockSwizzle;

    """
    return code
    
def get_meta_domain(config):
    if config["out_format"] in ["row", "col"]:
        return "_Mma"

def generate_argument_list(code, config):
    meta_domain = get_meta_domain(config)

    code += """
template<typename Element, typename _Mma, typename _SharedStorage, typename _Epilogue>
__global__ void %s(
    cutlass::gemm::GemmCoord problem_size,
    cutlass::gemm::GemmCoord grid_tiled_shape,
    typename _Mma::IteratorA::Params params_A,
    Element* __restrict__ ptr_A,
    typename _Mma::IteratorB::Params params_B,
    Element* __restrict__ ptr_B,
    typename _Epilogue::OutputTileIterator::Params params_D,
    Element* __restrict__ ptr_D,
    typename %s::IteratorE::Params params_E,
    typename %s::ElementE* __restrict__ ptr_E,
    typename _Epilogue::OutputOp::Params output_op_,
    int gemm_k_size)
{
    """ % (get_kernel_name(config), meta_domain, meta_domain)
    return code

def shmem_allocation(code):
    code += """// Allocate dynamic shared memory
    extern __shared__ int SharedStorageBase[];

    _SharedStorage& shared_storage = *reinterpret_cast<_SharedStorage *>(SharedStorageBase);
    """

    return code

def get_ctx_offset(code):
    code += """
    // Get threadblock offset
    ThreadblockSwizzle threadblock_swizzle;
    cutlass::gemm::GemmCoord threadblock_tile_offset=threadblock_swizzle.get_tile_offset(grid_tiled_shape);
    """
    return code

def ctx_range_check(code):
    code += """

    // Early exit if CTA is out of range
    if (grid_tiled_shape.m() <= threadblock_tile_offset.m() ||
        grid_tiled_shape.n() <= threadblock_tile_offset.n())
    {
        return;
    }
    """
    return code

def get_thread_indices(code):
    code += """
    int batch_idx = threadblock_swizzle.get_batch_idx();

    // Compute position within threadblock
    int thread_idx = threadIdx.x;
    // Broadcast the warp_id computed by lane 0 to ensure dependent code
    // is compuled as warp-uniform
    int warp_idx = __shfl_sync(0xffffffff, threadIdx.x / 32, 0);
    int lane_idx = threadIdx.x % 32;
    """

    return code


def tb_offset_mma(code, config):
    code += """
    // Compute initial location in logical coordinates
    """
    if (config["lhs_format"] in ["row_sp"]):
        # offset of A and B
        code += """
    cutlass::MatrixCoord tb_offset_A{
        threadblock_tile_offset.m() * _Mma::Shape::kM,
        0
    };

    cutlass::MatrixCoord tb_offset_B{
        0,
        threadblock_tile_offset.n() * _Mma::Shape::kN
    };
    """

    # offset of E
    code += """
    cutlass::MatrixCoord tb_offset_E{
        threadblock_tile_offset.m() * _Mma::Shape::kM,
        0
    };
        """
    return code


def get_problem_size_k(code):
    code += """
    // Problem size
    int problem_size_k = problem_size.k(); //min(problem_size.k(), (threadblock_tile_offset.k() + 1) * gemm_k_size);
    int gemm_k_iterations = (problem_size_k - tb_offset_B.row() + _Mma::Shape::kK - 1) / _Mma::Shape::kK;

    """
    return code


def iterator_mma(code, config):
    code += """
    // Construct iterators to operands
    """
    if (config["lhs_format"] in ["row_sp"]):
        code += """
    typename _Mma::IteratorA iterator_A(
        params_A,
        //ref_A.data(),
        ptr_A,
        {problem_size.m(), problem_size_k / _Mma::kSparse},
        thread_idx,
        tb_offset_A
    );

    typename _Mma::IteratorB iterator_B(
        params_B,
        //ref_B.data(),
        ptr_B,
        {problem_size_k, problem_size.n()},
        thread_idx,
        tb_offset_B
    );
        """
    
        code += """
    typename _Mma::IteratorE iterator_E(
        params_E,
        // ref_E.data(),
        ptr_E,
        {problem_size.m(),
        problem_size_k / _Mma::kSparse / _Mma::kElementsPerElementE},
        thread_idx,
        tb_offset_E
    );
        """
        if (config["batched"]):
            code += """
    // Add batch offsets
    iterator_A.add_pointer_offset(batch_idx * problem_size.m() * problem_size.k() / _Mma::kSparse);
    iterator_B.add_pointer_offset(batch_idx * problem_size.n() * problem_size.k());
    iterator_E.add_pointer_offset(batch_idx * problem_size.m() * problem_size.k() / _Mma::kSparse / _Mma::kElementsPerElementE);
            """
    
    return code


def main_loop(code):

    code += """
    //
    //  Main Loop
    //

    // Construct thread-scoped matrix multiply
    _Mma mma(shared_storage.main_loop, thread_idx, warp_idx, lane_idx);

    typename _Mma::FragmentC accumulators;

    accumulators.clear();

    if (gemm_k_iterations > 0){
        mma(gemm_k_iterations, accumulators, iterator_A, iterator_B, iterator_E, accumulators);
    }
    """
    return code


def epilogue_op(code):

    code += """
    
    //
    //  Epilogue
    //

    typename _Epilogue::OutputOp output_op(output_op_);
    """
    return code

def epilogue_iterators(code, config):

    code += """
    // Construct epilogue iterators
    """
    if (config["out_format"] in ["row", "col"]):
        code += """
    cutlass::MatrixCoord threadblock_offset(
        threadblock_tile_offset.m() * _Mma::Shape::kM,
        threadblock_tile_offset.n() * _Mma::Shape::kN
    );
    
    typename _Epilogue::OutputTileIterator iterator_D(
        params_D,
        ptr_D,
        problem_size.mn(),
        thread_idx,
        threadblock_offset
    );
        """
        if (config["batched"]):
            code += """
    iterator_D.add_pointer_offset(batch_idx * problem_size.m() * problem_size.n());

            """
    
    return code

def construct_epilogue(code):
    code += """
    // Construct epilogue
    _Epilogue epilogue(
        shared_storage.epilogue,
        thread_idx,
        warp_idx,
        lane_idx
    );
    """
    return code

def launch_epilogue(code, config):
    code += """
    // Launch epilogue
    """
    if (config["out_format"] in ["row", "col"]):
        code += """
    epilogue(output_op, iterator_D, accumulators, iterator_D);
        """
    
    return code

def kernel_epilogue(code):
    code += """
}
    """
    return code



######################################################
# Test case
######################################################

code = header_prologue()

config = {
    "dtype": torch.float16,
    "lhs_format": "row_sp",
    "rhs_format": "col",
    "out_format": "row",
    "batched": True
}

kernel_name = get_kernel_name(config)

code = get_swizzle(code, config)
code = generate_argument_list(code, config)
code = shmem_allocation(code)
code = get_ctx_offset(code)
code = ctx_range_check(code)
code = get_thread_indices(code)
code = tb_offset_mma(code, config)
code = get_problem_size_k(code)
code = iterator_mma(code, config)
code = main_loop(code)
code = epilogue_op(code)
code = epilogue_iterators(code, config)
code = construct_epilogue(code)
code = launch_epilogue(code, config)
code = kernel_epilogue(code)


# print(code)

generate_code(code, None, file_name=kernel_name.lower() + ".cuh")

