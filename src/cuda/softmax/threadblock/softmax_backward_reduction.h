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
struct SoftmaxBackwardBlockReduction {
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
    
    using OSoftmaxIterator = cutlass::softmax::threadblock::RowTileIterator<ThreadMapInput, ElementInput>;
    using GradSoftmaxIterator = cutlass::softmax::threadblock::RowTileIterator<ThreadMapInput, ElementInput>;

    static const int kNumThreads = WarpCount::kCount * 32;

    using BlockReduce = cub::BlockReduce<ElementAccumulator, kNumThreads>;
    using SumFragment = Array<ElementAccumulator, kElementsPerAccess>;

    //
    // Structure
    //

    struct Arguments {
        //
        // Data members
        //
        ElementInput* o_softmax;
        ElementInput* grad_softmax;
        MatrixCoord problem_size;
    };

    struct Params {
        //
        // Data members
        //
        ElementInput* o_softmax;
        ElementInput* grad_softmax;
        MatrixCoord problem_size;

        //
        // Members
        //
        CUTLASS_HOST_DEVICE
        Params(Arguments const &args):
            o_softmax(args.o_softmax),
            grad_softmax(args.grad_softmax),
            problem_size(args.problem_size)
        { }
    };

    union SharedStorage {
        typename BlockReduce::TempStorage temp_storage;
        ElementAccumulator broadcast_buffer;
    };

private:
    
    OSoftmaxIterator o_softmax_iterator_;
    GradSoftmaxIterator grad_softmax_iterator_;

    BlockReduce block_reduce_;

    ElementAccumulator* broadcast_buffer_ptr_;

    int thread_idx_;

public:
    /// Constructor
    CUTLASS_DEVICE
    SoftmaxBackwardBlockReduction(
        Params const & params,
        SharedStorage & shared_storage,
        int thread_idx,
        MatrixCoord threadblock_offset = MatrixCoord()
    ):
        o_softmax_iterator_(params.o_softmax, params.problem_size, thread_idx, threadblock_offset),
        grad_softmax_iterator_(params.grad_softmax, params.problem_size, thread_idx, threadblock_offset),
        block_reduce_(shared_storage.temp_storage),
        broadcast_buffer_ptr_(&shared_storage.broadcast_buffer),
        thread_idx_(thread_idx)
    { }

    /// Execute one softmax backward
    CUTLASS_DEVICE
    void operator()(ElementAccumulator &row_sum) {
        SumFragment sum_accumulator;
        sum_accumulator.fill(0);

        NumericArrayConverter<ElementAccumulator, ElementInput, kElementsPerAccess> converter;
        cutlass::multiplies<SumFragment> mult_op;
        cutlass::plus<SumFragment> plus_op;

        for (int i = 0; i < OSoftmaxIterator::Iterations::kColumn; i ++) {
            typename OSoftmaxIterator::Fragment tmp_o_softmax;
            typename GradSoftmaxIterator::Fragment tmp_grad_softmax;
            o_softmax_iterator_.load(tmp_o_softmax);
            grad_softmax_iterator_.load(tmp_grad_softmax);
            sum_accumulator = plus_op(mult_op(converter(tmp_o_softmax), converter(tmp_grad_softmax)), sum_accumulator);
        }

        /// Get the sum of entries in the array of each thread
        ElementAccumulator row_sum_mult[kElementsPerAccess];
        #pragma unroll
        for (int i = 0; i < kElementsPerAccess; i ++) {
            row_sum_mult[i] = *(sum_accumulator.data() + i);
        }

        row_sum = block_reduce_.Sum(row_sum_mult);
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
struct SoftmaxBackwardWarpReduction {
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

    static const int kInputBufferSize = InputIterator::Iterations::kColumn * kElementsPerAccess;

    using WarpReduce = cub::WarpReduce<ElementAccumulator>;

    using InputFragment = Array<ElementInput, kInputBufferSize>;

    using AccumulatorFragment = Array<ElementAccumulator, kElementsPerAccess>;

    //
    // Structure
    //

    struct Arguments {
        //
        // Data members
        //
        ElementInput* o_softmax;
        ElementInput* grad_softmax;
        MatrixCoord problem_size;
    };

    struct Params {
        //
        // Data members
        //
        ElementInput* o_softmax;
        ElementInput* grad_softmax;
        MatrixCoord problem_size;

        //
        // Members
        //
        CUTLASS_HOST_DEVICE
        Params(Arguments const &args):
            o_softmax(args.o_softmax),
            grad_softmax(args.grad_softmax),
            problem_size(args.problem_size)
        { }
    };

    union SharedStorage {
        typename WarpReduce::TempStorage temp_storage;
    };

private:
    
    InputIterator o_softmax_iterator_;
    InputIterator grad_softmax_iterator_;

    WarpReduce warp_reduce_;

    int thread_idx_;

public:
    /// Constructor
    CUTLASS_DEVICE
    SoftmaxBackwardWarpReduction(
        Params const & params,
        SharedStorage & shared_storage,
        int thread_idx,
        MatrixCoord threadblock_offset = MatrixCoord()
    ):
        o_softmax_iterator_(params.o_softmax, params.problem_size, thread_idx, threadblock_offset),
        grad_softmax_iterator_(params.grad_softmax, params.problem_size, thread_idx, threadblock_offset),
        warp_reduce_(shared_storage.temp_storage),
        thread_idx_(thread_idx)
    { }

    /// Execute one softmax backward
    CUTLASS_DEVICE
    void operator()(InputFragment & o_softmax_buffer, InputFragment & grad_softmax_buffer, ElementAccumulator &row_sum) {
        AccumulatorFragment sum_accumulator;
        sum_accumulator.fill(0);

        NumericArrayConverter<ElementAccumulator, ElementInput, kElementsPerAccess> converter;
        cutlass::multiplies<AccumulatorFragment> mult_op;
        cutlass::plus<AccumulatorFragment> plus_op;

        typename InputIterator::Fragment* tmp_o_softmax = reinterpret_cast<typename InputIterator::Fragment*>(&o_softmax_buffer);
        typename InputIterator::Fragment* tmp_grad_softmax = reinterpret_cast<typename InputIterator::Fragment*>(&grad_softmax_buffer);

        for (int i = 0; i < InputIterator::Iterations::kColumn; i ++) {
            o_softmax_iterator_.load(*tmp_o_softmax);
            grad_softmax_iterator_.load(*tmp_grad_softmax);
            sum_accumulator = plus_op(mult_op(converter(*tmp_o_softmax), converter(*tmp_grad_softmax)), sum_accumulator);
            tmp_o_softmax ++;
            tmp_grad_softmax ++;
        }

        /// Get the sum of entries in the array of each thread
        ElementAccumulator row_sum_mult = 0;
        #pragma unroll
        for (int i = 0; i < kElementsPerAccess; i ++) {
            row_sum_mult += *(sum_accumulator.data() + i);
        }

        row_sum = warp_reduce_.Sum(row_sum_mult);
        row_sum = __shfl_sync(0xffffffff, row_sum, 0);
    }
};

/////////////////////////////////////////////////////////////////////////////////////////////////

} // namespace threadblock
} // namespace softmax
} // namespace cutlass

/////////////////////////////////////////////////////////////////////////////////////////////////