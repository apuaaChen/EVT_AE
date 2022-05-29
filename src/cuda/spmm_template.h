#include "cutlass/cutlass.h"
#include "cutlass/gemm/device/gemm_sparse.h"
#include "cuda_bf16.h"
#include <type_traits>
#include "epilogue/default_epilogue_tensor_op.h"
#include "epilogue/pipelined_transpose_epilogue.h"
#include "epilogue/linear_combination.h"
#include "spmm_template.cuh"

// A structure to switch between different configurations

template<typename Element_, typename LayoutB_, bool Trans_, typename ThreadblockShape_, typename WarpShape_, typename InstructionShape_, int NumStages_>
struct SpMMConfigure{

    static const bool Trans = Trans_;
    using LayoutB = LayoutB_;
    using Element = Element_;
    using ThreadblockShape = ThreadblockShape_;
    using WarpShape = WarpShape_;
    using InstructionShape = InstructionShape_;
    static const int NumStages = NumStages_;

    using EpilogueOp = cutlass::epilogue::thread::LinearCombination_<
    Element, 128 / cutlass::sizeof_bits<Element>::value, float, Element,
    cutlass::epilogue::thread::ScaleType::OnlyAlphaScaling>;

    using Mma = typename cutlass::gemm::threadblock::DefaultSparseMma<
    Element, cutlass::layout::RowMajor, 128 / cutlass::sizeof_bits<Element>::value,
    Element, LayoutB, 128 / cutlass::sizeof_bits<Element>::value,
    float, cutlass::layout::RowMajor, cutlass::arch::OpClassTensorOp, cutlass::arch::Sm80,
    ThreadblockShape, WarpShape, InstructionShape, NumStages, cutlass::arch::OpMultiplyAdd>::ThreadblockMma;

    using Epilogue = typename std::conditional<Trans, 
        typename cutlass::epilogue::threadblock::DefaultTransposeEpilogue<
            ThreadblockShape, WarpShape, EpilogueOp, Mma>::Epilogue,
        typename cutlass::epilogue::threadblock::DefaultEpilogueTensorOp<
            ThreadblockShape, typename Mma::Operator, ThreadblockShape::kK / WarpShape::kK, EpilogueOp,
            EpilogueOp::kCount>::Epilogue>::type;
    
    union SharedStorage {
        typename Mma::SharedStorage main_loop;
        typename Epilogue::SharedStorage epilogue;
    }; 
};


template<typename Element, typename Config>
__global__ void cutlassSpmmKernel_16(
    cutlass::gemm::GemmCoord problem_size,
    cutlass::gemm::GemmCoord grid_tiled_shape,
    typename Config::Mma::IteratorA::Params params_A,
    Element* __restrict__ ptr_A,
    typename Config::Mma::IteratorB::Params params_B,
    Element* __restrict__ ptr_B,
    typename Config::Epilogue::OutputTileIterator::Params params_D,
    Element* __restrict__ ptr_D,
    typename Config::Mma::IteratorE::Params params_E,
    typename Config::Mma::ElementE* __restrict__ ptr_E,
    typename Config::Epilogue::OutputOp::Params output_op_,
    int gemm_k_size)
{
    cutlassSpmmKernel_16_<Element, Config::Mma, Config::SharedStorage, Config::Epilogue>(
        problem_size, grid_tiled_shape,
        params_A, ptr_A, params_B, ptr_B,
        params_D, ptr_D, params_E, ptr_E,
        output_op_, gemm_k_size
    );
}

template<typename Config>
torch::Tensor spmm_cuda(
    torch::Tensor tensor_a,
    torch::Tensor tensor_b,
    torch::Tensor tensor_e,
    const float alpha_)
{
    int m, n, k;
    m = tensor_a.size(-2);
    if (std::is_same<typename Config::LayoutB, cutlass::layout::RowMajor>::value){
        n = tensor_b.size(-1); 
        k = tensor_b.size(-2);   
    } else {
        n = tensor_b.size(-2);
        k = tensor_b.size(-1);
    }

    int batch_size = tensor_b.numel() / n / k;

    auto options_val = torch::TensorOptions().dtype(tensor_a.dtype()).device(tensor_b.device());
    torch::Tensor output_matrix;
    if (Config::Trans){
        output_matrix = torch::empty({batch_size, n, m}, options_val);
    } else {
        output_matrix = torch::empty({batch_size, m, n}, options_val);
    }

    // Create a tuple of problem size for matrix multiplication
    cutlass::gemm::GemmCoord problem_size(m, n, k);

    auto layout_a = cutlass::layout::RowMajor::packed(cutlass::make_Coord(problem_size.m(), problem_size.k() / 2));
    auto layout_b = Config::LayoutB::packed(problem_size.kn());
    auto layout_e = Config::Mma::LayoutE::packed(cutlass::make_Coord(problem_size.m(), problem_size.k()/Config::Mma::kSparse / Config::Mma::kElementsPerElementE));
    auto layout_d = cutlass::layout::RowMajor::packed(problem_size.mn());

    typename Config::Element alpha = typename Config::Element(alpha_);
    typename Config::Element beta = typename Config::Element(0.0);
    
    ThreadblockSwizzle threadblock_swizzle;

    cutlass::gemm::GemmCoord grid_tiled_shape = threadblock_swizzle.get_tiled_shape(
        problem_size,
        {Config::ThreadblockShape::kM, Config::ThreadblockShape::kN, Config::ThreadblockShape::kK},
        batch_size
    );

    dim3 grid = threadblock_swizzle.get_grid_shape(grid_tiled_shape);
    dim3 block(Config::Mma::WarpCount::kCount * 32, 1, 1);

    int smem_size = int(sizeof(typename Config::SharedStorage));

    cudaFuncSetAttribute(cutlassSpmmKernel_16<typename Config::Element, Config>, cudaFuncAttributeMaxDynamicSharedMemorySize, smem_size);
    cudaFuncSetAttribute(cutlassSpmmKernel_16<typename Config::Element, Config>, cudaFuncAttributePreferredSharedMemoryCarveout, 100);

    int gemm_k_size = ((problem_size.k() + Config::Mma::Shape::kK - 1) / Config::Mma::Shape::kK) * Config::Mma::Shape::kK;

    cutlassSpmmKernel_16<typename Config::Element, Config><<<grid, block, smem_size>>>(
        problem_size, grid_tiled_shape, 
        layout_a, (typename Config::Element*)tensor_a.data_ptr(),
        layout_b, (typename Config::Element*)tensor_b.data_ptr(),
        layout_d, (typename Config::Element*)output_matrix.data_ptr(),
        layout_e, (typename Config::Mma::ElementE*)tensor_e.data_ptr(),
        {alpha, beta}, gemm_k_size);

    return output_matrix;
}
