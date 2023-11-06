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
struct LayerNormWarpReduction :
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
        Array<typename Base::ElementInput, kInputBufferSize> input_buffer;
    };

    struct ReductionResult {
        typename Base::ElementAccumulator row_m1;
        typename Base::ElementAccumulator row_m2;
    };

    union SharedStorage {
        typename Base::SharedStorage shared_storage;
    };    

private:

    typename Base::InputIterator input_iterator_;

    typename Base::ElementAccumulator numel_;

public:
    /// Constructor
    CUTLASS_DEVICE
    LayerNormWarpReduction(
        Params const & params,
        SharedStorage &shared_storage,
        int thread_idx,
        MatrixCoord threadblock_offset = MatrixCoord()
    ):
        input_iterator_(
            params.input, params.problem_size, thread_idx, threadblock_offset
        ),
        Base(thread_idx, shared_storage.shared_storage),
        numel_(ElementAccumulator(params.problem_size.column()))
        { }
    
    /// Execute one softmax
    CUTLASS_DEVICE
    void operator()(InputCache & input_cache, ReductionResult & reduction_result) {

        /// Get the sum(x) and sum(x^2) of each row
        typename Base::AccumulatorFragment x_accumulator;
        typename Base::AccumulatorFragment x2_accumulator;
        
        x_accumulator.clear();
        x2_accumulator.clear();

        typename Base::InputFragment* input_buffer_ptr = reinterpret_cast<typename Base::InputFragment*>(&input_cache.input_buffer);
        typename Base::AccumulatorFragment tmp_input;

        #pragma unroll
        for (int i = 0; i < Base::InputIterator::Iterations::kColumn; i ++) {
            input_iterator_.load(*input_buffer_ptr);
            tmp_input = input2acc(*input_buffer_ptr);
            x_accumulator = tmp_input + x_accumulator;
            x2_accumulator = tmp_input * tmp_input + x2_accumulator;
            input_buffer_ptr ++;
        }

        /// Get the maximum entry in the array of each thread
        this->sum(x_accumulator, reduction_result.row_m1);
        this->sum(x2_accumulator, reduction_result.row_m2);
        reduction_result.row_m1 /= numel_;
        reduction_result.row_m2 /= numel_;
    }
};

/////////////////////////////////////////////////////////////////////////////////////////////////

} // namespace threadblock
} // namespace softmax
} // namespace cutlass

/////////////////////////////////////////////////////////////////////////////////////////////////