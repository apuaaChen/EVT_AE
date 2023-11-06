/********************************************************************************
* Copyright [yyyy] [name of copyright owner]
*
* Licensed under the Apache License, Version 2.0 (the "License");
* you may not use this file except in compliance with the License.
* You may obtain a copy of the License at
*
*     http://www.apache.org/licenses/LICENSE-2.0
*
* Unless required by applicable law or agreed to in writing, software
* distributed under the License is distributed on an "AS IS" BASIS,
* WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
* See the License for the specific language governing permissions and
* limitations under the License.
********************************************************************************/
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

// Reduction of softmax performed in a threadblock
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

    // The scalar reduction results per row
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

  template <class GTensor, class CTensor>
  struct ReductionCallbacks{
    CUTLASS_DEVICE
    ReductionCallbacks(
      GTensor&& tC_gRow,
      CTensor&& tC_cRow,
      Params const* params_ptr
    ):
      tC_gRow(cute::forward<GTensor>(tC_gRow)),
      params_ptr(params_ptr) { }
    
    GTensor tC_gRow;
    Params const* params_ptr;

    template <class ReductionResult>
    CUTLASS_DEVICE
    void operator()(ReductionResult &reduction_result) {

      auto src_v = filter(tC_gRow);
      auto coord_v = filter(tC_cRow);

      // Get the max of each row
      Array<ElementAccumulator, FragmentSize> max_accum;
      max_accum.fill(-1e+6);

      for (int i=0; i < size(tC_gRow); ++i) {
        Tensor dst_v = make_tensor_like(src_v(i));
        bool guard = get<1>(coord_v(i)) < n;
        cutlass::arch::global_load<VecType, sizeof(VecType)>(dst_v, (void const*)&src_v(i), guard);
        max(max_accumu, tmp_input);
      }

      this->max(max_accum, reduction_result.row_max);

      Array<ElementAccumulator, FragmentSize> sum_exp_accumulator;
      sum_exp_accumulator.clear();

      for (int i=0; i < size(tC_gRow); ++i) {
        Tensor dst_v = make_tensor_like(src_v(i));
        bool guard = get<1>(coord_v(i)) < n;
        cutlass::arch::global_load<VecType, sizeof(VecType)>(dst_v, (void const*)&src_v(i), guard);
        sum_exp_accumulator = exp(input2acc(dst_v) - reduction_result.row_max) + sum_exp_accumulator;
      }

      this->sum(sum_exp_acumulator, reduction_result.row_sum);
    }
  };

  template <class ProblemShape>
  CUTLASS_DEVICE auto
  get_callbacks(
    MatrixCoord threadblock_tile_offset,
    int thread_idx,
    ProblemShape problem_shape
  ) {
    Tensor mRow = make_tensor(
      make_gmem_ptr(params_ptr->ptr_row),
      problem_shape,
      params_ptr->dRow);
    
    Tensor tC_gRow = recast<VecType>(
      ThreadMap::partition(mRow, thread_idx, threadblock_tile_offset)
    );

    // Generate the pred tensor
    Tensor cRow = make_identity_tensor(mRow.shape());
    Tensor tC_cRow = local_partition(
      ThreadMap::partition(cRow, thread_idx, threadblock_tile_offset),
      Shape<Int<VecLength>>{},
      (_0{})
    );

    return ReductionCallbacks<
      decltype(tC_gRow), decltype(tC_cRow)>(
      cute::move(tC_gRow),
      cute::move(tC_cRow),
      params_ptr
    );
  }

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