// Auto-generated file. DO NOT MODIFY

#include "cuda.h"
#include "torch/extension.h"
#include "cuda_runtime.h"
#include "cutlass/cutlass.h"
#include "cutlass/gemm/device/gemm_sparse.h"
#include "cuda_fp16.h"
#include "epilogue/linear_combination.h"
#include "epilogue/default_epilogue_tensor_op.h"
#include "spmm_ntn.cuh"

struct SPMM_NTNConfig{ 

    using ThreadblockShape = cutlass::gemm::GemmShape<128, 128, 64>;
    using WarpShape = cutlass::gemm::GemmShape<64, 64, 64>;
    using InstructionShape = cutlass::gemm::GemmShape<16, 8, 32>;
    static const int NumStages = 3;

    using LayoutB = cutlass::layout::ColumnMajor;

    using Element = cutlass::half_t;

    using EpilogueOp = cutlass::epilogue::thread::LinearCombination_<
        Element, 128 / cutlass::sizeof_bits<Element>::value, float, Element,
        cutlass::epilogue::thread::ScaleType::OnlyAlphaScaling>;

    using Mma = typename cutlass::gemm::threadblock::DefaultSparseMma<
        Element, cutlass::layout::RowMajor, 128 / cutlass::sizeof_bits<Element>::value,
        Element, LayoutB, 128 / cutlass::sizeof_bits<Element>::value,
        float, cutlass::layout::RowMajor, cutlass::arch::OpClassTensorOp, cutlass::arch::Sm80,
        ThreadblockShape, WarpShape, InstructionShape, NumStages, cutlass::arch::OpMultiplyAdd>::ThreadblockMma;

    using Epilogue = cutlass::epilogue::threadblock::DefaultEpilogueTensorOp<
        ThreadblockShape, typename Mma::Operator, ThreadblockShape::kK / WarpShape::kK, EpilogueOp,
        EpilogueOp::kCount>::Epilogue;

    union SharedStorage {
        typename Mma::SharedStorage main_loop;
        typename Epilogue::SharedStorage epilogue;
    };  
};  

torch::Tensor spmm_ntn_cuda(
    torch::Tensor tensor_a,
    torch::Tensor tensor_b,
    torch::Tensor tensor_e
){
    // get problem size
    int m, n, k;

    m = tensor_a.size(-2);
    n = tensor_b.size(-2);
    k = tensor_b.size(-1);

    int batch_size = tensor_b.numel() / n / k;

    auto options_val = torch::TensorOptions().dtype(tensor_a.dtype()).device(tensor_b.device());
    torch::Tensor output_matrix = torch::empty({batch_size, m, n}, options_val);

    cutlass::gemm::GemmCoord problem_size(m, n, k);

    auto layout_a = cutlass::layout::RowMajor::packed(cutlass::make_Coord(problem_size.m(), problem_size.k() / 2));
    auto layout_b = SPMM_NTNConfig::LayoutB::packed(problem_size.kn());
    auto layout_e = SPMM_NTNConfig::Mma::LayoutE::packed(cutlass::make_Coord(problem_size.m(), problem_size.k()/SPMM_NTNConfig::Mma::kSparse / SPMM_NTNConfig::Mma::kElementsPerElementE));
    auto layout_d = cutlass::layout::RowMajor::packed(problem_size.mn());

    int gemm_k_size = ((problem_size.k() + SPMM_NTNConfig::Mma::Shape::kK - 1) / SPMM_NTNConfig::Mma::Shape::kK) * SPMM_NTNConfig::Mma::Shape::kK;

    typename SPMM_NTNConfig::Element alpha = typename SPMM_NTNConfig::Element(1.0);
    typename SPMM_NTNConfig::Element beta = typename SPMM_NTNConfig::Element(0.0);

    ThreadblockSwizzle threadblock_swizzle;

    cutlass::gemm::GemmCoord grid_tiled_shape = threadblock_swizzle.get_tiled_shape(
        problem_size,
        {SPMM_NTNConfig::ThreadblockShape::kM, SPMM_NTNConfig::ThreadblockShape::kN, SPMM_NTNConfig::ThreadblockShape::kK},
        batch_size
    );

    dim3 grid = threadblock_swizzle.get_grid_shape(grid_tiled_shape);
    dim3 block(SPMM_NTNConfig::Mma::WarpCount::kCount * 32, 1, 1);    

    int smem_size = int(sizeof(typename SPMM_NTNConfig::SharedStorage));

    cudaFuncSetAttribute(SpMM_ntn<typename SPMM_NTNConfig::Element, SPMM_NTNConfig::Mma, SPMM_NTNConfig::SharedStorage, SPMM_NTNConfig::Epilogue>, cudaFuncAttributeMaxDynamicSharedMemorySize, smem_size);
    cudaFuncSetAttribute(SpMM_ntn<typename SPMM_NTNConfig::Element, SPMM_NTNConfig::Mma, SPMM_NTNConfig::SharedStorage, SPMM_NTNConfig::Epilogue>, cudaFuncAttributePreferredSharedMemoryCarveout, 100);


    SpMM_ntn<typename SPMM_NTNConfig::Element, SPMM_NTNConfig::Mma, SPMM_NTNConfig::SharedStorage, SPMM_NTNConfig::Epilogue><<<grid, block, smem_size>>>(
        problem_size, grid_tiled_shape, 
        layout_a, (typename SPMM_NTNConfig::Element*)tensor_a.data_ptr(),
        layout_b, (typename SPMM_NTNConfig::Element*)tensor_b.data_ptr(),
        layout_d, (typename SPMM_NTNConfig::Element*)output_matrix.data_ptr(),
        layout_e, (typename SPMM_NTNConfig::Mma::ElementE*)tensor_e.data_ptr(),
        {alpha, beta}, gemm_k_size);

    return output_matrix;
}
