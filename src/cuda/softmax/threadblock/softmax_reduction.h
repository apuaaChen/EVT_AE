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
#include "reduce_apply/threadblock/reduction_base.h"
#include "cutlass/arch/memory.h"

using namespace cute;
using cute::tuple;

/////////////////////////////////////////////////////////////////////////////////////////////////

namespace cutlass {
namespace softmax {
namespace threadblock {

/////////////////////////////////////////////////////////////////////////////////////////////////

// Reduction of softmax
template<  
  class Element,                   // Data type of input data
  int Alignment,                   // Alignment of input data
  class ElementAccumulator_,        // Data type to perform reduction
  class ThreadMap,                 // Thread map
  int kNumThreads                  // Number of warps per threadblock
> 
struct SoftmaxReduction : 
  reduce_apply::threadblock::ReductionBase<
    ElementAccumulator_, kNumThreads> {
    
  using ElementAccumulator = ElementAccumulator_;

  using Base = reduce_apply::threadblock::ReductionBase<
    ElementAccumulator, kNumThreads>;

  using StrideMNL = cute::Stride<int64_t, _1, int64_t>;

  // The reduction result produced by the current softmax
  struct ReductionResult{
    ElementAccumulator row_max;
    ElementAccumulator row_sum;
  };

  static const int kElementsPerAccess = Alignment;

  struct Arguments {
    Element const* ptr_input = nullptr;
    StrideMNL dInput = {};
  };

  struct Params {
    Element const* ptr_input = nullptr;
    StrideMNL dInput = {};
    Element null_default = std::numeric_limits<Element>::lowest();
  };

  template <class ProblemShape>
  static constexpr Params
  to_underlying_arguments(ProblemShape const& problem_shape, Arguments const& args) {
    return {args.ptr_input, args.dInput};
  }

  using SharedStorage = typename Base::SharedStorage;

  //
  // Constructor and data members
  //

  CUTLASS_HOST_DEVICE
  SoftmaxReduction() { }

  CUTLASS_HOST_DEVICE
  SoftmaxReduction(Params const& params, SharedStorage& shared_storage, int thread_idx)
    : params_ptr(&params),
    Base(thread_idx, shared_storage) { }
  
  Params const* params_ptr;

  //
  // Input cache
  //

  static int constexpr vec_bits = ThreadMap::kElementsPerAccess * sizeof_bits<Element>::value;
  using VecType = uint_bit_t<cute::min(128, vec_bits)>;
  static int constexpr VecLength = sizeof(VecType) / sizeof(Element);

  template <class GTensor, class CTensor, class ProblemShape>
  struct InputCache {
    CUTLASS_DEVICE
    InputCache(
      GTensor&& tC_gInput,
      CTensor&& tC_cInput,
      ProblemShape problem_shape,
      Element null_default
    ):
      tC_gInput(cute::forward<GTensor>(tC_gInput)),
      tC_cInput(cute::forward<CTensor>(tC_cInput)),
      iterations_per_row(cute::get<1>(tC_gInput.shape())),
      problem_shape(problem_shape),
      null_default(null_default)
      { }
    
    GTensor tC_gInput;
    CTensor tC_cInput;
    int iterations_per_row;
    ProblemShape problem_shape;
    Element null_default;

    CUTLASS_DEVICE
    Array<Element, VecLength> get(int row_idx, int column_idx) {
      bool guard = elem_less(tC_cInput(row_idx, column_idx), problem_shape);
      Array<Element, VecLength> value;
      VecType* value_ptr = reinterpret_cast<VecType*>(&value);
      cutlass::arch::global_load<VecType, sizeof(VecType)>(*value_ptr, (void*)&tC_gInput(row_idx, column_idx), guard);
      if (!guard) value.fill(null_default);
      return value;
    }
  };

  template <class ProblemShape>
  CUTLASS_DEVICE auto
  get_input_cache(
    MatrixCoord threadblock_tile_offset,
    int thread_idx,
    ProblemShape problem_shape
  ) {
    Tensor mInput = make_tensor(
      make_gmem_ptr(params_ptr->ptr_input),
      problem_shape,
      params_ptr->dInput
    );

    // ITERATION_ROW, ITERATION_COLUMN
    Tensor tC_gInput = recast<VecType>(
      ThreadMap::partition(mInput, thread_idx, threadblock_tile_offset))(_,_0{},_);

    // Generate the pred tensor
    Tensor cInput = make_identity_tensor(mInput.shape());
    Tensor tC_cInput = ThreadMap::partition(
      cInput, thread_idx, threadblock_tile_offset)(_,_0{},_);

    return InputCache<
      decltype(tC_gInput), decltype(tC_cInput), 
      decltype(problem_shape)>(
      cute::move(tC_gInput),
      cute::move(tC_cInput),
      problem_shape,
      params_ptr->null_default
    );
  }

  //
  // Functions
  //

  template <class InputCache>
  CUTLASS_DEVICE
  void reduce(ReductionResult &reduction_result, InputCache &inputs, int row_idx) {

    using ConvertInput = NumericArrayConverter<ElementAccumulator, Element, VecLength>;
    ConvertInput convert_input{};

    Array<ElementAccumulator, VecLength> max_accum;
    max_accum.fill(-1e+6);

    for (int column_idx=0; column_idx < inputs.iterations_per_row; ++column_idx) {
      Array<Element, VecLength> value = inputs.get(row_idx, column_idx);
      max_accum = max(max_accum, convert_input(value));
    }

    this->max(max_accum, reduction_result.row_max);

    Array<ElementAccumulator, VecLength> sum_exp_accum;
    sum_exp_accum.clear();

    for (int column_idx=0; column_idx < inputs.iterations_per_row; ++column_idx) {
      Array<Element, VecLength> value = inputs.get(row_idx, column_idx);
      sum_exp_accum = exp(convert_input(value) - reduction_result.row_max) + sum_exp_accum;
    }

    this->sum(sum_exp_accum, reduction_result.row_sum);
  }

  template <class InputCache>
  CUTLASS_DEVICE
  Array<ElementAccumulator, VecLength> apply(ReductionResult &reduction_result, InputCache &inputs, int row_idx, int column_idx) {
    using ConvertInput = NumericArrayConverter<ElementAccumulator, Element, VecLength>;
    ConvertInput convert_input{};
    Array<ElementAccumulator, VecLength> compute_frg = convert_input(inputs.get(row_idx, column_idx));

    return exp(compute_frg - reduction_result.row_max) / reduction_result.row_sum;
  }
};

/////////////////////////////////////////////////////////////////////////////////////////////////

} // namespace threadblock
} // namespace softmax
} // namespace cutlass

/////////////////////////////////////////////////////////////////////////////////////////////////