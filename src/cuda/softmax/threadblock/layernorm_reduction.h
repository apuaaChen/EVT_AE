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
struct LayerNormWarpReduction {
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

    using InputFragment = Array<ElementAccumulator, kInputBufferSize>;
    using AccumulatorFragment = Array<ElementAccumulator, kElementsPerAccess>;

    using SumFragment = Array<ElementAccumulator, kElementsPerAccess>;

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
    LayerNormWarpReduction(
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
    void operator()(InputFragment & input_buffer, ElementAccumulator &row_m1, ElementAccumulator &row_m2) {

        /// Get the sum(x) and sum(x^2) of each row
        SumFragment x_accumulator;
        SumFragment x2_accumulator;
        
        x_accumulator.clear();
        x2_accumulator.clear();

        NumericArrayConverter<ElementAccumulator, ElementInput, kElementsPerAccess> converter;

        cutlass::plus<SumFragment> plus_op;
        cutlass::multiplies<SumFragment> mul_op;

        AccumulatorFragment* input_buffer_ptr = reinterpret_cast<AccumulatorFragment*>(&input_buffer);

        for (int i = 0; i < InputIterator::Iterations::kColumn; i ++) {
            typename InputIterator::Fragment tmp_input;
            input_iterator_.load(tmp_input);
            *input_buffer_ptr = converter(tmp_input);
            x_accumulator = plus_op(*input_buffer_ptr, x_accumulator);
            x2_accumulator = plus_op(
                mul_op(*input_buffer_ptr, *input_buffer_ptr),
                x2_accumulator
            );
            input_buffer_ptr ++;
        }

        /// Get the maximum entry in the array of each thread
        ElementAccumulator row_x = ElementAccumulator(0);
        ElementAccumulator row_x2 = ElementAccumulator(0);
        #pragma unroll
        for (int i = 0; i < kElementsPerAccess; i ++) {
            row_x += *(x_accumulator.data() + i);
            row_x2 += *(x2_accumulator.data() + i);
        }

        row_m1 = warp_reduce_.Sum(row_x);
        row_m2 = warp_reduce_.Sum(row_x2);

        row_m1 = __shfl_sync(0xffffffff, row_m1, 0);
        row_m2 = __shfl_sync(0xffffffff, row_m2, 0);
    }
};

/////////////////////////////////////////////////////////////////////////////////////////////////

} // namespace threadblock
} // namespace softmax
} // namespace cutlass

/////////////////////////////////////////////////////////////////////////////////////////////////