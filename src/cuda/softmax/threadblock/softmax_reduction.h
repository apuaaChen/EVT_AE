#pragma once
#include "softmax/threadblock/row_tile_iterator.h"
#include <cub/block/block_reduce.cuh>
#include <cub/warp/warp_reduce.cuh>




/////////////////////////////////////////////////////////////////////////////////////////////////

namespace cutlass {
namespace softmax {
namespace threadblock {

/////////////////////////////////////////////////////////////////////////////////////////////////


template<
    typename ThreadblockShape_,
    typename WarpCount_,
    typename ElementInput_,
    int AlignmentInput_,
    typename ElementAccumulator_>
struct SoftmaxBlockReduction {
    using ThreadblockShape = ThreadblockShape_;
    using WarpCount = WarpCount_;
    using ElementInput = ElementInput_;
    using ElementAccumulator = ElementAccumulator_;
    static const int kElementsPerAccess = AlignmentInput_;

    //
    // Iterators
    //

    using ThreadMapInput = cutlass::softmax::threadblock::RowThreadMap<
        ThreadblockShape, WarpCount, ElementInput, AlignmentInput_>;
    
    using MaxIterator = cutlass::softmax::threadblock::RowTileIterator<ThreadMapInput, ElementInput>;
    using SumExpIterator = cutlass::softmax::threadblock::RowTileIterator<ThreadMapInput, ElementInput>;

    static const int kNumThreads = WarpCount::kCount * 32;

    using BlockReduce = cub::BlockReduce<ElementAccumulator, kNumThreads>;
    using MaxFragment = Array<ElementAccumulator, kElementsPerAccess>;
    using SumExpFragment = Array<ElementAccumulator, kElementsPerAccess>;


    //
    // Structures
    //

    struct Arguments {
        //
        // Data members
        //
        ElementInput* input;
        MatrixCoord problem_size;
    };

    struct Params {
        //
        // Data members
        //
        ElementInput* input;
        MatrixCoord problem_size;

        //
        // Members
        //
        CUTLASS_HOST_DEVICE
        Params(Arguments const &args):
            input(args.input),
            problem_size(args.problem_size)
        { }

    };

    union SharedStorage {
        typename BlockReduce::TempStorage temp_storage;
        ElementAccumulator broadcast_buffer;
    };

private:

    MaxIterator max_iterator_;
    SumExpIterator sum_exp_iterator_;

    BlockReduce block_reduce_;

    ElementAccumulator* broadcast_buffer_ptr_;

    int thread_idx_;


public:
    /// Constructor
    CUTLASS_DEVICE
    SoftmaxBlockReduction(
        Params const & params,
        SharedStorage &shared_storage,
        int thread_idx,
        MatrixCoord threadblock_offset = MatrixCoord()
    ):
        max_iterator_(
            params.input, params.problem_size, thread_idx, threadblock_offset
        ),
        sum_exp_iterator_(
            params.input, params.problem_size, thread_idx, threadblock_offset
        ),
        block_reduce_(shared_storage.temp_storage),
        broadcast_buffer_ptr_(&shared_storage.broadcast_buffer),
        thread_idx_(thread_idx)
        { }


    /// Execute one softmax
    CUTLASS_DEVICE
    void operator()(ElementAccumulator &row_max, ElementAccumulator &row_sum) {

        /// Get the max of each row
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
            max_iterator_.load(tmp_input);
            max_accumulator = maximum_op(converter(tmp_input), max_accumulator);
        }

        /// Get the maximum entry in the array of each thread

        ElementAccumulator row_maxs[kElementsPerAccess];
        #pragma unroll
        for (int i = 0; i < kElementsPerAccess; i ++) {
            row_maxs[i] = *(max_accumulator.data() + i);
        }


        row_max = block_reduce_.Reduce(row_maxs, cub::Max());

        if (thread_idx_ == 0) *(broadcast_buffer_ptr_) = row_max;
        __syncthreads();
        row_max = *(broadcast_buffer_ptr_);

        /// Perform Sum Exp in each row

        SumExpFragment sum_exp_accumulator;
        sum_exp_accumulator.clear();

        for (int i = 0; i < SumExpIterator::Iterations::kColumn; i ++) {
            typename SumExpIterator::Fragment tmp_input;
            sum_exp_iterator_.load(tmp_input);
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

        row_sum = block_reduce_.Sum(row_sum_exp);
        if (thread_idx_ == 0) *(broadcast_buffer_ptr_) = row_sum;
        __syncthreads();
        row_sum = *(broadcast_buffer_ptr_);
    }
};


template<
    typename ThreadblockShape_,
    typename WarpCount_,
    typename ElementInput_,
    int AlignmentInput_,
    typename ElementAccumulator_>
struct SoftmaxWarpReduction {
    using ThreadblockShape = ThreadblockShape_;
    using WarpCount = WarpCount_;
    using ElementInput = ElementInput_;
    using ElementAccumulator = ElementAccumulator_;
    static const int kElementsPerAccess = AlignmentInput_;

    //
    // Iterators
    //

    using ThreadMapInput = cutlass::softmax::threadblock::RowThreadMap<
        ThreadblockShape, WarpCount, ElementInput, AlignmentInput_>;
    
    using InputIterator = cutlass::softmax::threadblock::RowTileIterator<ThreadMapInput, ElementInput>;

    static const int kNumThreads = 32;

    static const int kInputBufferSize = ThreadblockShape::kColumn / 32;

    using WarpReduce = cub::WarpReduce<ElementAccumulator>;

    using InputFragment = Array<ElementInput, kInputBufferSize>;

    using MaxFragment = Array<ElementAccumulator, kElementsPerAccess>;
    using SumExpFragment = Array<ElementAccumulator, kElementsPerAccess>;

    //
    // Structures
    //

    struct Arguments {
        //
        // Data members
        //
        ElementInput* input;
        MatrixCoord problem_size;
    };

    struct Params {
        //
        // Data members
        //
        ElementInput* input;
        MatrixCoord problem_size;

        //
        // Members
        //
        CUTLASS_HOST_DEVICE
        Params(Arguments const &args):
            input(args.input),
            problem_size(args.problem_size)
        { }

    };

    union SharedStorage {
        typename WarpReduce::TempStorage temp_storage;
    };    

private:

    InputIterator input_iterator_;

    WarpReduce warp_reduce_;

    int thread_idx_;

public:
    /// Constructor
    CUTLASS_DEVICE
    SoftmaxWarpReduction(
        Params const & params,
        SharedStorage &shared_storage,
        int thread_idx,
        MatrixCoord threadblock_offset = MatrixCoord()
    ):
        input_iterator_(
            params.input, params.problem_size, thread_idx, threadblock_offset
        ),
        warp_reduce_(shared_storage.temp_storage),
        thread_idx_(thread_idx)
        { }
    
    /// Execute one softmax
    CUTLASS_DEVICE
    void operator()(InputFragment & input_buffer, ElementAccumulator &row_max, ElementAccumulator &row_sum) {

        /// Get the max of each row
        MaxFragment max_accumulator;
        // max_accumulator.fill(std::numeric_limits<ElementAccumulator>::lowest());
        // TODO: lowest is a host function and not callable
        max_accumulator.fill(-1e+6);

        NumericArrayConverter<ElementAccumulator, ElementInput, kElementsPerAccess> converter;

        cutlass::maximum<MaxFragment> maximum_op;
        cutlass::minus<SumExpFragment> minus_op;
        cutlass::fast_exp_op<SumExpFragment> exp_op;
        cutlass::plus<SumExpFragment> plus_op;

        typename InputIterator::Fragment* tmp_input = reinterpret_cast<typename InputIterator::Fragment*>(&input_buffer);

        for (int i = 0; i < InputIterator::Iterations::kColumn; i ++) {
            input_iterator_.load(*tmp_input);
            max_accumulator = maximum_op(converter(*tmp_input), max_accumulator);
            tmp_input ++;
        }

        /// Get the maximum entry in the array of each thread
        ElementAccumulator row_max_local = ElementAccumulator(-1e+6);
        #pragma unroll
        for (int i = 0; i < kElementsPerAccess; i ++) {
            row_max_local = *(max_accumulator.data() + i) < row_max_local ? row_max_local : *(max_accumulator.data() + i);
        }

        row_max = warp_reduce_.Reduce(row_max_local, cub::Max());

        row_max = __shfl_sync(0xffffffff, row_max, 0);


        SumExpFragment sum_exp_accumulator;
        sum_exp_accumulator.clear();

        tmp_input = reinterpret_cast<typename InputIterator::Fragment*>(&input_buffer);

        for (int i = 0; i < InputIterator::Iterations::kColumn; i ++) {
            SumExpFragment tmp_result = minus_op(converter(*tmp_input), row_max);
            tmp_result = exp_op(tmp_result);
            sum_exp_accumulator = plus_op(tmp_result, sum_exp_accumulator);
            tmp_input ++;
        }

        ElementAccumulator row_sum_exp = ElementAccumulator(0);
        #pragma unroll
        for (int i = 0; i < kElementsPerAccess; i ++) {
            row_sum_exp += *(sum_exp_accumulator.data() + i);
        }

        row_sum = warp_reduce_.Sum(row_sum_exp);

        row_sum = __shfl_sync(0xffffffff, row_sum, 0);
    }
};

/////////////////////////////////////////////////////////////////////////////////////////////////

} // namespace threadblock
} // namespace softmax
} // namespace cutlass

/////////////////////////////////////////////////////////////////////////////////////////////////