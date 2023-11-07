#pragma once
#include "softmax/threadblock/row_tile_iterator.h"
#include "softmax/threadblock/reduction_base.h"
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
struct SoftmaxBackwardBlockReduction :
    cutlass::reduce_apply::threadblock::BlockReductionBase <
        ThreadblockShape_, WarpCount_, ElementInput_,
        AlignmentInput_, ElementAccumulator_> {
    
    using Base = cutlass::reduce_apply::threadblock::BlockReductionBase <
        ThreadblockShape_,
        WarpCount_,
        ElementInput_,
        AlignmentInput_,
        ElementAccumulator_>;

    //
    // Structure
    //

    struct Arguments {
        typename Base::ElementInput* o_softmax;
        typename Base::ElementInput* grad_softmax;
        MatrixCoord problem_size;
    };

    struct Params {
        typename Base::ElementInput* o_softmax;
        typename Base::ElementInput* grad_softmax;
        MatrixCoord problem_size;

        CUTLASS_HOST_DEVICE
        Params(Arguments const &args):
            o_softmax(args.o_softmax),
            grad_softmax(args.grad_softmax),
            problem_size(args.problem_size)
        { }
    };

    struct InputCache {};

    struct ReductionResult {
        typename Base::ElementAccumulator row_sum;
    };

    union SharedStorage {
        typename Base::SharedStorage shared_storage;
        // ElementAccumulator broadcast_buffer;
    };

private:
    
    typename Base::InputIterator o_softmax_iterator_;
    typename Base::InputIterator grad_softmax_iterator_;


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
        Base(thread_idx, shared_storage.shared_storage)
    { }

    /// Execute one softmax backward
    CUTLASS_DEVICE
    void operator()(ReductionResult &reduction_result) {
        typename Base::AccumulatorFragment sum_accumulator;
        sum_accumulator.fill(0);

        for (int i = 0; i < Base::InputIterator::Iterations::kColumn; i ++) {
            typename Base::InputFragment tmp_o_softmax = o_softmax_iterator_.load();
            typename Base::InputFragment tmp_grad_softmax = grad_softmax_iterator_.load();
            sum_accumulator = input2acc(tmp_o_softmax) * input2acc(tmp_grad_softmax) + sum_accumulator;
        }

        /// Get the sum of entries in the array of each thread
        this->sum(sum_accumulator, reduction_result.row_sum);
    }
};


template<
    typename ThreadblockShape_,
    typename WarpCount_,
    typename ElementInput_,
    int AlignmentInput_,
    typename ElementAccumulator_>
struct SoftmaxBackwardWarpReduction :
    cutlass::reduce_apply::threadblock::WarpReductionBase <
        ThreadblockShape_, WarpCount_, ElementInput_,
        AlignmentInput_, ElementAccumulator_> {

    using Base = cutlass::reduce_apply::threadblock::WarpReductionBase <
        ThreadblockShape_,
        WarpCount_,
        ElementInput_,
        AlignmentInput_,
        ElementAccumulator_>;

    //
    // Structure
    //

    struct Arguments {
        //
        // Data members
        //
        typename Base::ElementInput* o_softmax;
        typename Base::ElementInput* grad_softmax;
        MatrixCoord problem_size;
    };

    struct Params {
        //
        // Data members
        //
        typename Base::ElementInput* o_softmax;
        typename Base::ElementInput* grad_softmax;
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

    struct InputCache {
        static const int kInputBufferSize = 
            Base::InputIterator::Iterations::kColumn * Base::kElementsPerAccess;
        Array<typename Base::ElementInput, kInputBufferSize> o_softmax_buffer;
        Array<typename Base::ElementInput, kInputBufferSize> grad_softmax_buffer;
    };

    struct ReductionResult {
        typename Base::ElementAccumulator row_sum;
    };

    union SharedStorage {
        typename Base::SharedStorage shared_storage;
    }; 

private:
    
    typename Base::InputIterator o_softmax_iterator_;
    typename Base::InputIterator grad_softmax_iterator_;

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
        Base(thread_idx, shared_storage.shared_storage)
    { }

    /// Execute one softmax backward
    CUTLASS_DEVICE
    void operator()(InputCache & input_cache, ReductionResult &reduction_result) {
        typename Base::AccumulatorFragment sum_accumulator;
        sum_accumulator.fill(0);

        typename Base::InputFragment* tmp_o_softmax = reinterpret_cast<typename Base::InputFragment*>(&input_cache.o_softmax_buffer);
        typename Base::InputFragment* tmp_grad_softmax = reinterpret_cast<typename Base::InputFragment*>(&input_cache.grad_softmax_buffer);

        for (int i = 0; i < Base::InputIterator::Iterations::kColumn; i ++) {
            o_softmax_iterator_.load(*tmp_o_softmax, typename Base::InputIterator::Element(0));
            grad_softmax_iterator_.load(*tmp_grad_softmax, typename Base::InputIterator::Element(0));
            sum_accumulator = input2acc(*tmp_o_softmax) * input2acc(*tmp_grad_softmax) + sum_accumulator;
            tmp_o_softmax ++;
            tmp_grad_softmax ++;
        }

        /// Get the sum of entries in the array of each thread
        this->sum(sum_accumulator, reduction_result.row_sum);
    }
};

/////////////////////////////////////////////////////////////////////////////////////////////////

} // namespace threadblock
} // namespace softmax
} // namespace cutlass

/////////////////////////////////////////////////////////////////////////////////////////////////