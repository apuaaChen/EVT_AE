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
struct LayerNormBackwardWarpReduction {
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
        ElementInput* gamma;
        ElementInput* grad_layernorm;
        ElementInput* x;
        ElementAccumulator* mean;
        ElementAccumulator* std;
        MatrixCoord problem_size;
    };

    struct Params {
        //
        // Data members
        //
        ElementInput* gamma;
        ElementInput* grad_layernorm;
        ElementInput* x;
        ElementAccumulator* mean;
        ElementAccumulator* std;
        MatrixCoord problem_size;

        //
        // Members
        //
        CUTLASS_HOST_DEVICE
        Params(Arguments const &args):
            gamma(args.gamma),
            grad_layernorm(args.grad_layernorm),
            x(args.x),
            mean(args.mean),
            std(args.std),
            problem_size(args.problem_size)
        { }
    };

    union SharedStorage {
        typename WarpReduce::TempStorage temp_storage;
    };

private:
    
    InputIterator gamma_iterator_;
    InputIterator grad_layernorm_iterator_;
    InputIterator x_iterator_;
    ElementInput mean_;
    ElementInput std_;
    
    WarpReduce warp_reduce_;

    int thread_idx_;

public:
    /// Constructor
    CUTLASS_DEVICE
    LayerNormBackwardWarpReduction(
        Params const & params,
        SharedStorage & shared_storage,
        int thread_idx,
        MatrixCoord threadblock_offset = MatrixCoord()
    ):
        gamma_iterator_(params.gamma, params.problem_size, thread_idx, MatrixCoord(0, threadblock_offset.column())),
        grad_layernorm_iterator_(params.grad_layernorm, params.problem_size, thread_idx, threadblock_offset),
        x_iterator_(params.x, params.problem_size, thread_idx, threadblock_offset),
        warp_reduce_(shared_storage.temp_storage),
        thread_idx_(thread_idx)
    {
        int row_idx = threadblock_offset.row();
        mean_ = ElementInput(*(params.mean + row_idx));
        std_ = ElementInput(*(params.std + row_idx));
    }

    /// Execute one softmax backward
    CUTLASS_DEVICE
    void operator()(
        InputFragment & gamma_grad_y_buffer, InputFragment & x_hat_buffer, 
        ElementAccumulator &row_sum_t1, ElementAccumulator &row_sum_t2) {
        //
        AccumulatorFragment sum_t1_accumulator;
        AccumulatorFragment sum_t2_accumulator;
        sum_t1_accumulator.fill(0);
        sum_t2_accumulator.fill(0);

        NumericArrayConverter<ElementAccumulator, ElementInput, kElementsPerAccess> converter;
        cutlass::multiplies<Array<ElementInput, kElementsPerAccess>> mult_op;
        cutlass::plus<AccumulatorFragment> plus_op;
        cutlass::minus<Array<ElementInput, kElementsPerAccess>> sub_op;

        typename InputIterator::Fragment* tmp_gamma_grad_y = reinterpret_cast<typename InputIterator::Fragment*>(&gamma_grad_y_buffer);
        typename InputIterator::Fragment* tmp_x_hat = reinterpret_cast<typename InputIterator::Fragment*>(&x_hat_buffer);

        typename InputIterator::Fragment tmp_gamma;

        for (int i = 0; i < InputIterator::Iterations::kColumn; i ++) {
            gamma_iterator_.load(tmp_gamma);
            grad_layernorm_iterator_.load(*tmp_gamma_grad_y);
            x_iterator_.load(*tmp_x_hat);
            *tmp_gamma_grad_y = mult_op(mult_op(tmp_gamma, *tmp_gamma_grad_y), std_);
            *tmp_x_hat = mult_op(sub_op(*tmp_x_hat, mean_), std_);

            sum_t1_accumulator = plus_op(converter(*tmp_gamma_grad_y), sum_t1_accumulator);
            sum_t2_accumulator = plus_op(converter(mult_op(*tmp_gamma_grad_y, *tmp_x_hat)), sum_t2_accumulator);

            tmp_gamma_grad_y ++;
            tmp_x_hat ++;
        }

        /// Get the sum of entries in the array of each thread
        row_sum_t1 = 0;
        row_sum_t2 = 0;
        #pragma unroll
        for (int i = 0; i < kElementsPerAccess; i ++) {
            row_sum_t1 += *(sum_t1_accumulator.data() + i);
            row_sum_t2 += *(sum_t2_accumulator.data() + i);
        }

        row_sum_t1 = warp_reduce_.Sum(row_sum_t1);
        row_sum_t2 = warp_reduce_.Sum(row_sum_t2);
        row_sum_t1 = __shfl_sync(0xffffffff, row_sum_t1, 0);
        row_sum_t2 = __shfl_sync(0xffffffff, row_sum_t2, 0);
    }
};

/////////////////////////////////////////////////////////////////////////////////////////////////

} // namespace threadblock
} // namespace softmax
} // namespace cutlass

/////////////////////////////////////////////////////////////////////////////////////////////////