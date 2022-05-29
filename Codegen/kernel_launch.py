from utils import get_kernel_name, generate_code
import torch


def header_files(code, config):
    headers = ["cuda.h", "torch/extension.h", "cuda_runtime.h", "cutlass/cutlass.h", "cutlass/gemm/device/gemm_sparse.h"]

    # Data type
    if config["dtype"] in [torch.float16]:
        headers.append("cuda_fp16.h")
    elif config["dtype"] in [torch.bfloat16]:
        headers.append("cuda_bf16.h")
    
    # epilogue
    headers.append("epilogue/linear_combination.h")
    if config["out_format"] == "row":
        headers.append("epilogue/default_epilogue_tensor_op.h")
    elif config["out_format"] == "col":
        headers.append("epilogue/pipelined_transpose_epilogue.h")
    
    headers.append(config["name"] + ".cuh")



    code += """// Auto-generated file. DO NOT MODIFY

"""
    for h in headers:
        code += """#include "%s"
"""%h
    return code


def configure_struct(code, config):
    code += """
struct %sConfig{{ 

    using ThreadblockShape = cutlass::gemm::GemmShape<%s, %s, %s>;
    using WarpShape = cutlass::gemm::GemmShape<%s, %s, %s>;
    using InstructionShape = cutlass::gemm::GemmShape<%s, %s, %s>;
    static const int NumStages = %s;
""" % (config["name"].upper(), 
       config["threadblockTileShape"][0], config["threadblockTileShape"][1], config["threadblockTileShape"][2],
       config["warpTileShape"][0], config["warpTileShape"][1], config["warpTileShape"][2],
       config["InstructionShape"][0], config["InstructionShape"][1], config["InstructionShape"][2],
       config["Stage"])

    # rhs layout
    if config["rhs_format"] == "row":
        code += """
    using LayoutB = cutlass::layout::RowMajor;
"""
    elif config["rhs_format"] == "col":
        code += """
    using LayoutB = cutlass::layout::ColumnMajor;
"""

    # data type
    if config["dtype"] == torch.float16:
        code += """
    using Element = cutlass::half_t;
"""
    elif config["dtype"] == torch.bfloat16:
        code += """
    using Element = cutlass::bfloat16_t;
"""

    code += """
    using EpilogueOp = cutlass::epilogue::thread::LinearCombination_<
        Element, 128 / cutlass::sizeof_bits<Element>::value, float, Element,
        cutlass::epilogue::thread::ScaleType::OnlyAlphaScaling>;
"""

    if config["lhs_format"] == "row_sp":
        code += """
    using Mma = typename cutlass::gemm::threadblock::DefaultSparseMma<
        Element, cutlass::layout::RowMajor, 128 / cutlass::sizeof_bits<Element>::value,
        Element, LayoutB, 128 / cutlass::sizeof_bits<Element>::value,
        float, cutlass::layout::RowMajor, cutlass::arch::OpClassTensorOp, cutlass::arch::Sm80,
        ThreadblockShape, WarpShape, InstructionShape, NumStages, cutlass::arch::OpMultiplyAdd>::ThreadblockMma;
"""

    if config["out_format"] == "row":
        code += """
    using Epilogue = cutlass::epilogue::threadblock::DefaultEpilogueTensorOp<
        ThreadblockShape, typename Mma::Operator, ThreadblockShape::kK / WarpShape::kK, EpilogueOp,
        EpilogueOp::kCount>::Epilogue;
"""
    elif config["out_format"] == "col":
        code += """
    using Epilogue = cutlass::epilogue::threadblock::DefaultTransposeEpilogue<
            ThreadblockShape, WarpShape, EpilogueOp, Mma>::Epilogue;
    """

    code += """
    union SharedStorage {{
        typename Mma::SharedStorage main_loop;
        typename Epilogue::SharedStorage epilogue;
    }};  
}};  
""" 
    return code


def generate_launcher(code, config):

    config_name = config["name"].upper() + "Config"
    if config["out_format"] in ["row", "col"]:
        code += """
torch::Tensor {kernel_l}_cuda(
    torch::Tensor tensor_a,
    torch::Tensor tensor_b,
    torch::Tensor tensor_e
){{
    // get problem size
    int m, n, k;
"""
    
    if config["lhs_format"] in ["row_sp"]:
        code += """
    m = tensor_a.size(-2);"""

    if config["rhs_format"] in ["row"]:
        code += """
    n = tensor_b.size(-1);
    k = tensor_b.size(-2);
"""
    elif config["rhs_format"] in ["col"]:
        code += """
    n = tensor_b.size(-2);
    k = tensor_b.size(-1);
"""

    code += """
    int batch_size = tensor_b.numel() / n / k;
"""

    # Construct output matrix
    code += """
    auto options_val = torch::TensorOptions().dtype(tensor_a.dtype()).device(tensor_b.device());"""
    if config["out_format"] == "row":
        code += """
    torch::Tensor output_matrix = torch::empty({{batch_size, m, n}}, options_val);
"""
    elif config["out_format"] == "col":
        code += """
    torch::Tensor output_matrix = torch::empty({{batch_size, n, m}}, options_val);
"""

    # get problem size
    code += """
    cutlass::gemm::GemmCoord problem_size(m, n, k);

    auto layout_a = cutlass::layout::RowMajor::packed(cutlass::make_Coord(problem_size.m(), problem_size.k() / 2));
    auto layout_b = {config}::LayoutB::packed(problem_size.kn());"""
    if config["lhs_format"] == "row_sp":
        code += """
    auto layout_e = {config}::Mma::LayoutE::packed(cutlass::make_Coord(problem_size.m(), problem_size.k()/{config}::Mma::kSparse / {config}::Mma::kElementsPerElementE));"""
    if config["out_format"] in ["row", "col"]:
        code += """
    auto layout_d = cutlass::layout::RowMajor::packed(problem_size.mn());

    int gemm_k_size = ((problem_size.k() + {config}::Mma::Shape::kK - 1) / {config}::Mma::Shape::kK) * {config}::Mma::Shape::kK;
"""

    # alpha and beta
    code += """
    typename {config}::Element alpha = typename {config}::Element(1.0);
    typename {config}::Element beta = typename {config}::Element(0.0);
""" 

    # thread block size
    code += """
    ThreadblockSwizzle threadblock_swizzle;

    cutlass::gemm::GemmCoord grid_tiled_shape = threadblock_swizzle.get_tiled_shape(
        problem_size,
        {{{config}::ThreadblockShape::kM, {config}::ThreadblockShape::kN, {config}::ThreadblockShape::kK}},
        batch_size
    );

    dim3 grid = threadblock_swizzle.get_grid_shape(grid_tiled_shape);
    dim3 block({config}::Mma::WarpCount::kCount * 32, 1, 1);    
""" 
    # shared memory
    code += """
    int smem_size = int(sizeof(typename {config}::SharedStorage));

    cudaFuncSetAttribute({kernel}<typename {config}::Element, {config}::Mma, {config}::SharedStorage, {config}::Epilogue>, cudaFuncAttributeMaxDynamicSharedMemorySize, smem_size);
    cudaFuncSetAttribute({kernel}<typename {config}::Element, {config}::Mma, {config}::SharedStorage, {config}::Epilogue>, cudaFuncAttributePreferredSharedMemoryCarveout, 100);

"""

    # launch kernel
    code += """
    {kernel}<typename {config}::Element, {config}::Mma, {config}::SharedStorage, {config}::Epilogue><<<grid, block, smem_size>>>(
        problem_size, grid_tiled_shape, 
        layout_a, (typename {config}::Element*)tensor_a.data_ptr(),
        layout_b, (typename {config}::Element*)tensor_b.data_ptr(),
        layout_d, (typename {config}::Element*)output_matrix.data_ptr(),
        layout_e, (typename {config}::Mma::ElementE*)tensor_e.data_ptr(),
        {{alpha, beta}}, gemm_k_size);
"""

    # return
    code += """
    return output_matrix;
}}
"""


    code = code.format(config=config_name, kernel=get_kernel_name(config), kernel_l=get_kernel_name(config).lower())


    return code



######################################################
# Test case
######################################################

code = ""

config = {
    "dtype": torch.float16,
    "lhs_format": "row_sp",
    "rhs_format": "col",
    "out_format": "row",
    "batched": True,
    "name": "spmm_ntn",
    "threadblockTileShape": (128, 128, 64),
    "warpTileShape": (64, 64, 64),
    "InstructionShape": (16, 8, 32),
    "Stage": 3
}

code = header_files(code, config)
code = configure_struct(code, config)
code = generate_launcher(code, config)

generate_code(code, None, file_name=get_kernel_name(config).lower() + ".cu")

