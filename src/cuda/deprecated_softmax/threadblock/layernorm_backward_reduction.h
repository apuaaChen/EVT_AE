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
struct LayerNormBackwardWarpReduction :
    cutlass::reduce_apply::threadblock::WarpReductionBase <
        ThreadblockShape_, WarpCount_, ElementInput_,
        AlignmentInput_, ElementAccumulator_>  {
    
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
        typename Base::ElementInput* gamma;
        typename Base::ElementInput* grad_layernorm;
        typename Base::ElementInput* x;
        typename Base::ElementAccumulator* mean;
        typename Base::ElementAccumulator* std;
        MatrixCoord problem_size;
    };

    struct Params {
        typename Base::ElementInput* gamma;
        typename Base::ElementInput* grad_layernorm;
        typename Base::ElementInput* x;
        typename Base::ElementAccumulator* mean;
        typename Base::ElementAccumulator* std;
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

    struct InputCache {
        static const int kInputBufferSize = 
            Base::InputIterator::Iterations::kColumn * Base::kElementsPerAccess;
        Array<typename Base::ElementInput, kInputBufferSize> gamma_grad_y_buffer;
        Array<typename Base::ElementInput, kInputBufferSize> x_hat_buffer;
    };

    struct ReductionResult {
        typename Base::ElementAccumulator row_sum_t1;
        typename Base::ElementAccumulator row_sum_t2;
    };

    union SharedStorage {
        typename Base::SharedStorage shared_storage;
    };

private:
    
    typename Base::InputIterator gamma_iterator_;
    typename Base::InputIterator grad_layernorm_iterator_;
    typename Base::InputIterator x_iterator_;
    typename Base::ElementInput mean_;
    typename Base::ElementInput std_;

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
        Base(thread_idx, shared_storage.shared_storage)
    {
        int row_idx = threadblock_offset.row();
        mean_ = ElementInput(*(params.mean + row_idx));
        std_ = ElementInput(*(params.std + row_idx));
    }

    /// Execute one softmax backward
    CUTLASS_DEVICE
    void operator()(InputCache & input_cache, ReductionResult & reduction_result) {
        //
        typename Base::AccumulatorFragment sum_t1_accumulator;
        typename Base::AccumulatorFragment sum_t2_accumulator;
        sum_t1_accumulator.fill(0);
        sum_t2_accumulator.fill(0);

        typename Base::InputFragment* tmp_gamma_grad_y = reinterpret_cast<typename Base::InputFragment*>(&input_cache.gamma_grad_y_buffer);
        typename Base::InputFragment* tmp_x_hat = reinterpret_cast<typename Base::InputFragment*>(&input_cache.x_hat_buffer);

        typename Base::InputFragment tmp_gamma;

        for (int i = 0; i < Base::InputIterator::Iterations::kColumn; i ++) {
            gamma_iterator_.load(tmp_gamma);
            grad_layernorm_iterator_.load(*tmp_gamma_grad_y);
            x_iterator_.load(*tmp_x_hat);
            *tmp_gamma_grad_y = tmp_gamma * (*tmp_gamma_grad_y) * std_;
            *tmp_x_hat = ((*tmp_x_hat) - mean_) * std_;

            sum_t1_accumulator = input2acc(*tmp_gamma_grad_y) + sum_t1_accumulator;
            typename Base::InputFragment tmp_mult = (*tmp_gamma_grad_y) * (*tmp_x_hat);
            sum_t2_accumulator = input2acc(tmp_mult) + sum_t2_accumulator;

            tmp_gamma_grad_y ++;
            tmp_x_hat ++;
        }

        /// Get the sum of entries in the array of each thread
        this->sum(sum_t1_accumulator, reduction_result.row_sum_t1);
        this->sum(sum_t2_accumulator, reduction_result.row_sum_t2);
    }
};

/////////////////////////////////////////////////////////////////////////////////////////////////

} // namespace threadblock
} // namespace softmax
} // namespace cutlass

/////////////////////////////////////////////////////////////////////////////////////////////////