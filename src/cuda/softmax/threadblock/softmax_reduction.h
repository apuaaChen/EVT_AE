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
struct SoftmaxBlockReduction : 
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
    // Structures
    //

    struct Arguments {
        typename Base::ElementInput* input;
        MatrixCoord problem_size;
    };

    struct Params {
        typename Base::ElementInput* input;
        MatrixCoord problem_size;

        CUTLASS_HOST_DEVICE
        Params(Arguments const &args):
            input(args.input),
            problem_size(args.problem_size)
        { }
    };

    struct ReductionResult {
        typename Base::ElementAccumulator row_max;
        typename Base::ElementAccumulator row_sum;
    };

    struct InputCache {};

    union SharedStorage {
        typename Base::SharedStorage shared_storage;
    };

private:

    typename Base::InputIterator max_iterator_;
    typename Base::InputIterator sum_exp_iterator_;

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
        Base(thread_idx, shared_storage.shared_storage)
        { }


    /// Execute one softmax
    CUTLASS_DEVICE
    void operator()(ReductionResult &reduction_result) {

        /// Get the max of each row
        typename Base::AccumulatorFragment max_accumulator;
        max_accumulator.fill(-1e+6);

        for (int i = 0; i < Base::InputIterator::Iterations::kColumn; i ++) {
            typename Base::InputFragment tmp_input = max_iterator_.load();
            max_accumulator = max(input2acc(tmp_input), max_accumulator);
        }

        this->max(max_accumulator, reduction_result.row_max);

        typename Base::AccumulatorFragment sum_exp_accumulator;
        sum_exp_accumulator.clear();

        for (int i = 0; i < Base::InputIterator::Iterations::kColumn; i ++) {
            typename Base::InputFragment tmp_input = sum_exp_iterator_.load();
            sum_exp_accumulator = exp(input2acc(tmp_input) - reduction_result.row_max) + sum_exp_accumulator;
        }

        this->sum(sum_exp_accumulator, reduction_result.row_sum);
    }
};


template<
    typename ThreadblockShape_,
    typename WarpCount_,
    typename ElementInput_,
    int AlignmentInput_,
    typename ElementAccumulator_>
struct SoftmaxWarpReduction :
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
    // Structures
    //

    struct Arguments {
        typename Base::ElementInput* input;
        MatrixCoord problem_size;
    };

    struct Params {
        typename Base::ElementInput* input;
        MatrixCoord problem_size;

        CUTLASS_HOST_DEVICE
        Params(Arguments const &args):
            input(args.input),
            problem_size(args.problem_size)
        { }

    };

    struct InputCache {
        static const int kInputBufferSize = 
            Base::InputIterator::Iterations::kColumn * Base::kElementsPerAccess;
        Array<typename Base::ElementAccumulator, kInputBufferSize> input_buffer;
    };

    struct ReductionResult {
        typename Base::ElementAccumulator row_max;
        typename Base::ElementAccumulator row_sum;
    };

    union SharedStorage {
        typename Base::SharedStorage shared_storage;
    };    

private:

    typename Base::InputIterator input_iterator_;

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
        Base(thread_idx, shared_storage.shared_storage) { }
    
    /// Execute one softmax
    CUTLASS_DEVICE
    void operator()(InputCache & input_cache, ReductionResult & reduction_result) {

        /// Get the max of each row
        typename Base::AccumulatorFragment max_accumulator;
        max_accumulator.fill(-1e+6);

        typename Base::AccumulatorFragment* input_buffer_ptr = 
            reinterpret_cast<typename Base::AccumulatorFragment*>(&input_cache.input_buffer);

        for (int i = 0; i < Base::InputIterator::Iterations::kColumn; i ++) {
            typename Base::InputFragment tmp_input= input_iterator_.load();
            *input_buffer_ptr = input2acc(tmp_input);
            max_accumulator = max(*input_buffer_ptr, max_accumulator);
            input_buffer_ptr ++;
        }

        this->max(max_accumulator, reduction_result.row_max);

        typename Base::AccumulatorFragment sum_exp_accumulator;
        sum_exp_accumulator.clear();

        input_buffer_ptr = reinterpret_cast<typename Base::AccumulatorFragment*>(&input_cache.input_buffer);

        for (int i = 0; i < Base::InputIterator::Iterations::kColumn; i ++) {
            *input_buffer_ptr = exp(*input_buffer_ptr - reduction_result.row_max);
            sum_exp_accumulator = *input_buffer_ptr + sum_exp_accumulator;
            input_buffer_ptr ++;
        }

        this->sum(sum_exp_accumulator, reduction_result.row_sum);
    }
};

/////////////////////////////////////////////////////////////////////////////////////////////////

} // namespace threadblock
} // namespace softmax
} // namespace cutlass

/////////////////////////////////////////////////////////////////////////////////////////////////