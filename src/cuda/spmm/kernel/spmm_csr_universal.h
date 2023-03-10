#pragma once

#include "spmm/epilogue/epilogue.h"
#include <stdio.h>

/////////////////////////////////////////////////////////////////////////////////////////////////

namespace cutlass {
namespace spmm {
namespace kernel {

template<
    typename Element,            // data element
    typename ElementAcc,         // accumulator element
    int AlignmentEmb,            // embedding alignment
    int AlignmentNnz,            // nonzeros alignment
    typename Index,              // index type
    typename ThreadblockShape    // Threadblock-level tile size (concept: GemmShape)
>
struct SpmmRowBalanceDefault {
    static const int kElementsPerAccess = AlignmentEmb;
    
    using FragmentAcc = Array<ElementAcc,kElementsPerAccess>;

    using ThreadMap = cutlass::MatrixShape<
        ThreadblockShape::kColumn / AlignmentEmb,
        ThreadblockShape::kRow
    >;

    using Epilogue = cutlass::spmm::threadblock::DefaultEpilogueSpmm<
        ElementAcc,
        Element,
        AlignmentEmb,
        FragmentAcc,
        ThreadblockShape,
        ThreadMap
    >;
}; 


template<
    typename Element,            // data element
    typename ElementAcc,         // accumulator element
    int AlignmentEmb,            // embedding alignment
    int AlignmentNnz,            // nonzeros alignment
    typename Index,              // index type
    typename ThreadblockShape,   // Threadblock-level tile size (concept: GemmShape)
    typename Epilogue            // Epilogue functor
>
struct SpmmRowBalancewithEpilogueVisitor {
public:
    static_assert(ThreadblockShape::kColumn % AlignmentEmb == 0, "Threadblock-level tile size Y should be multiple of AlignmentEmb");
    static_assert(AlignmentNnz == 1, "Alignment of NNZ should be 1");

    // get number of threads
    static const int kNumThreads = ThreadblockShape::kRow * ThreadblockShape::kColumn / AlignmentEmb;
    // thread mapping
    using ThreadMap = cutlass::MatrixShape<
        ThreadblockShape::kRow,
        ThreadblockShape::kColumn / AlignmentEmb
    >;

    static const int kElementsPerAccess = AlignmentEmb;

    using FragmentAcc = Array<ElementAcc,kElementsPerAccess>;

    using LoadType = AlignedArray<Element, kElementsPerAccess>;

    /// Epilogue

    using EpilogueVisitor = typename Epilogue::Visitor;

    union SharedStorage {
        typename EpilogueVisitor::SharedStorage visitor;
    };

    //
    // Structures
    //

    // host constructed argument
    struct Arguments {
        //
        // Data members
        //
        Index* ptr_row;
        Index* ptr_indices;
        Element* ptr_e;
        Element* ptr_b;
        MatrixCoord problem_size;
        typename Epilogue::Arguments epilogue_args;
        typename EpilogueVisitor::Arguments epilogue_visitor;
    };

    // host constructed params
    struct Params {
        Index* ptr_row;
        Index* ptr_indices;
        Element* ptr_e;
        Element* ptr_b;
        MatrixCoord problem_size;
        typename Epilogue::Params epilogue_params;
        typename EpilogueVisitor::Params epilogue_visitor;

        /// constructor
        Params(
            Arguments const &args
        ):
            ptr_row(args.ptr_row),
            ptr_indices(args.ptr_indices),
            ptr_e(args.ptr_e),
            ptr_b(args.ptr_b),
            problem_size(args.problem_size),
            epilogue_params(args.epilogue_args),
            epilogue_visitor(args.epilogue_visitor)
        { }
    };

    /// Execute one SpMM with row balance
    CUTLASS_DEVICE
    void operator()(Params const &params, SharedStorage & shared_storage) {

        // Compute threadblock location
        // TODO: use swizzling
        int row_idx = blockIdx.x * ThreadblockShape::kRow + threadIdx.x / ThreadMap::kColumn;
        int col_idx = blockIdx.y * ThreadblockShape::kColumn + threadIdx.x % ThreadMap::kColumn * kElementsPerAccess;

        if (row_idx >= params.problem_size.row() || col_idx >= params.problem_size.column()) return;

        // grab start value
        Index start = params.ptr_row[row_idx];
        Index end = params.ptr_row[row_idx + 1];

        Element* b_ptr = params.ptr_b + col_idx;

        NumericArrayConverter<ElementAcc, Element, kElementsPerAccess> input2acc;
        cutlass::multiplies<Array<ElementAcc, kElementsPerAccess>> multiply_op;
        cutlass::plus<Array<ElementAcc, kElementsPerAccess>> add_op;

        // initialize accumulator
        FragmentAcc accumulator;
        accumulator.clear();

        // in this mapping, each subwarp actually travers all the nnzs sequentially
        // TODO: vector type for aligned edge weight
        for (int i=start; i < end; i++) {
            // step 0: load the nnz index and value
            Index node_index = *(params.ptr_indices+ i);
            ElementAcc edge_weight = ElementAcc(*(params.ptr_e + i));
            // step 1: load the vector to accumulate
            LoadType* frag_ptr = reinterpret_cast<LoadType*>(b_ptr + node_index * params.problem_size.column());
            
            // step 2: type conversion and accumulation
            accumulator = add_op(multiply_op(input2acc(*frag_ptr), edge_weight), accumulator);
        }

        cutlass::MatrixCoord threadblock_offset = {
            int(blockIdx.x * ThreadblockShape::kRow),
            int(blockIdx.y * ThreadblockShape::kColumn)
        };

        cutlass::gemm::GemmCoord threadblock_tile_offset = {
            int(blockIdx.x),
            int(blockIdx.y),
            int(blockIdx.z)
        };

        int thread_idx = threadIdx.x;

        // link to epilogue
        EpilogueVisitor epilogue_visitor(
            params.epilogue_visitor,
            shared_storage.visitor,
            threadblock_offset,
            threadblock_tile_offset,
            thread_idx,
            params.problem_size
        );

        Epilogue epilogue(
            params.epilogue_params,
            thread_idx,
            row_idx,
            col_idx
        );
        
        // Execute the epilogue operator
        epilogue(epilogue_visitor, accumulator);
    }
};

/////////////////////////////////////////////////////////////////////////////////////////////////

} // namespace kernel
} // namespace spmm
} // namespace cutlass