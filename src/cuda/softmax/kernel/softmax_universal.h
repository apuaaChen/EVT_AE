#pragma once

#include "cutlass/cutlass.h"
#include "softmax/threadblock/row_tile_iterator.h"
#include "softmax/epilogue/epilogue.h"
#include "stdio.h"

#if defined(__NVCC__)
#else
/// cub header requires a set of device functions that are unavailable in g++
/// We use a set of fake functions to workaround this issue
void __syncthreads(){}
void cudaGetLastError(){}
int __syncthreads_and(int p){}
int __syncthreads_or(int p){}
int __any(int p){}
int __all(int p){}
int __ballot(int p){}
int __shfl(unsigned int p, int k){}

#endif
#include <cub/block/block_reduce.cuh>



/////////////////////////////////////////////////////////////////////////////////////////////////

namespace cutlass {
namespace softmax {
namespace kernel {

/////////////////////////////////////////////////////////////////////////////////////////////////

template<
    typename ThreadblockShape_,
    typename WarpCount_,
    typename ElementInput_,
    int AlignmentInput_,
    typename ElementOutput_,
    int AlignmentOutput_,
    typename ElementAccumulator_>
struct SoftmaxUniversal {

    static_assert(AlignmentInput_ == AlignmentOutput_, "Input and output tensor should have the same alignemnt");

    using ThreadblockShape = ThreadblockShape_;
    using WarpCount = WarpCount_;
    using ElementInput = ElementInput_;
    using ElementOutput = ElementOutput_;
    using ElementAccumulator = ElementAccumulator_;
    static const int kElementsPerAccess = AlignmentInput_;

    //
    // Iterators
    //

    using ThreadMapInput = cutlass::softmax::threadblock::RowThreadMap<
        ThreadblockShape, WarpCount, ElementInput, AlignmentInput_>;
    
    using MaxIterator = cutlass::softmax::threadblock::RowTileIterator<ThreadMapInput, ElementInput>;
    using SumExpIterator = cutlass::softmax::threadblock::RowTileIterator<ThreadMapInput, ElementInput>;
    using EpilogueInputIterator = cutlass::softmax::threadblock::RowTileIterator<ThreadMapInput, ElementInput>;

    static const int kNumThreads = WarpCount::kCount * 32;

    using BlockReduce = cub::BlockReduce<ElementAccumulator, kNumThreads>;
    using MaxFragment = Array<ElementAccumulator, kElementsPerAccess>;
    using SumExpFragment = Array<ElementAccumulator, kElementsPerAccess>;

    using Epilogue = cutlass::softmax::threadblock::DefaultEpilogueSoftmax<
        ElementOutput,
        kElementsPerAccess,
        ThreadblockShape,
        WarpCount
    >;


    //
    // Structures
    //

    struct Arguments {
        //
        // Data members
        //
        ElementInput* input;
        ElementOutput* output;
        MatrixCoord problem_size;
    };

    struct Params {
        //
        // Data members
        //
        ElementInput* input;
        ElementOutput* output;
        MatrixCoord problem_size;

        //
        // Memebrs
        //
        CUTLASS_HOST_DEVICE
        Params(Arguments const &args):
            input(args.input),
            output(args.output),
            problem_size(args.problem_size)
        { }

    };

    union SharedStorage {
        typename BlockReduce::TempStorage temp_storage;
        ElementAccumulator broadcast_buffer;
    };


public:

    /// Execute one softmax
    CUTLASS_DEVICE
    void operator()(Params const &params, SharedStorage &shared_storage) {

        int thread_idx = threadIdx.x;
        cutlass::MatrixCoord threadblock_offset{
            int(blockIdx.x), int(blockIdx.y)
        };

        /// Get the max of each row
        MaxIterator max_iterator(params.input, params.problem_size, thread_idx, threadblock_offset);
        MaxFragment max_accumulator;
        // max_accumulator.fill(std::numeric_limits<ElementAccumulator>::lowest());
        // TODO: lowest is a host function and not callable
        max_accumulator.fill(-1e+6);

        NumericArrayConverter<ElementAccumulator, ElementInput, kElementsPerAccess> converter;

        cutlass::maximum<MaxFragment> maximum_op;
        cutlass::minus<SumExpFragment> minus_op;
        cutlass::fast_exp_op<SumExpFragment> exp_op;
        cutlass::plus<SumExpFragment> plus_op;

        for (int i = 0; i < MaxIterator::Iterations::kColumn; i ++) {
            typename MaxIterator::Fragment tmp_input;
            max_iterator.load(tmp_input);
            max_accumulator = maximum_op(converter(tmp_input), max_accumulator);
        }

        /// Get the maximum entry in the array of each thread

        ElementAccumulator row_maxs[kElementsPerAccess];
        #pragma unroll
        for (int i = 0; i < kElementsPerAccess; i ++) {
            row_maxs[i] = *(max_accumulator.data() + i);
        }


        ElementAccumulator row_max = BlockReduce(shared_storage.temp_storage).Reduce(row_maxs, cub::Max());

        if (thread_idx == 0) shared_storage.broadcast_buffer = row_max;
        __syncthreads();
        row_max = shared_storage.broadcast_buffer;

        /// TODO: Warp and block reduction

        // max_accumulator.fill(row_maxs[0]);

        /// Perform Sum Exp in each row

        SumExpIterator sum_exp_iterator(params.input, params.problem_size, thread_idx, threadblock_offset);
        SumExpFragment sum_exp_accumulator;
        sum_exp_accumulator.clear();

        for (int i = 0; i < SumExpIterator::Iterations::kColumn; i ++) {
            typename SumExpIterator::Fragment tmp_input;
            sum_exp_iterator.load(tmp_input);
            SumExpFragment tmp_result = minus_op(converter(tmp_input), row_max);
            tmp_result = exp_op(tmp_result);
            sum_exp_accumulator = plus_op(tmp_result, sum_exp_accumulator);
        }

        /// Get the sum of exp entry in the array of each thread
        ElementAccumulator row_sum_exp[kElementsPerAccess];
         #pragma unroll
        for (int i = 0; i < kElementsPerAccess; i ++) {
            row_sum_exp[i] = *(sum_exp_accumulator.data() + i);
        }

        /// TODO: Warp and block reduction

        ElementAccumulator row_sum = BlockReduce(shared_storage.temp_storage).Sum(row_sum_exp);
        if (thread_idx == 0) shared_storage.broadcast_buffer = row_sum;
        __syncthreads();
        row_sum = shared_storage.broadcast_buffer;


        
        /// Epilogue
        Epilogue epilogue(params.output, params.problem_size, thread_idx, threadblock_offset);

        EpilogueInputIterator epilogue_input_iterator(params.input, params.problem_size, thread_idx, threadblock_offset);

        cutlass::minus<typename EpilogueInputIterator::Fragment> epilogue_minus_op;
        cutlass::fast_exp_op<typename EpilogueInputIterator::Fragment> epilogue_exp_op;
        cutlass::divides<typename EpilogueInputIterator::Fragment> epilogue_div_op;

        for (int i = 0; i < EpilogueInputIterator::Iterations::kColumn; i ++) {
            
            typename EpilogueInputIterator::Fragment tmp_input;
            epilogue_input_iterator.load(tmp_input);
            tmp_input = epilogue_minus_op(tmp_input, ElementInput(row_max));
            tmp_input = epilogue_exp_op(tmp_input);
            tmp_input = epilogue_div_op(tmp_input, ElementInput(row_sum));
            epilogue.store(tmp_input);
        }
    }
};

/////////////////////////////////////////////////////////////////////////////////////////////////

} // namespace kernel
} // namespace softmax
} // namespace cutlass

/////////////////////////////////////////////////////////////////////////////////////////////////