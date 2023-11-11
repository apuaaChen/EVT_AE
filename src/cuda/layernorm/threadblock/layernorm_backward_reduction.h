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
namespace layernorm {
namespace threadblock {

/////////////////////////////////////////////////////////////////////////////////////////////////

// Reduction of Layernorm
template<
  class Element,
  int Alignment,
  class ElementAccumulator_,
  class ThreadMap,
  bool CacheInput = true
>
struct LayerNormBackwardReduction :
  reduce_apply::threadblock::ReductionBase<
    ElementAccumulator_, ThreadMap::kNumThreads> {
  
  static const int kNumThreads = ThreadMap::kNumThreads;
  using ElementAccumulator = ElementAccumulator_;

  using Base = reduce_apply::threadblock::ReductionBase<
    ElementAccumulator, kNumThreads>;

  using StrideMNL = cute::Stride<_0, _1, int64_t>;

  // The reduction result produced by the current softmax
  struct ReductionResult{
    ElementAccumulator sum_t1;
    ElementAccumulator sum_t2;
  };

  static const int kElementsPerAccess = Alignment;

  struct Arguments {
    Element const* ptr_grad = nullptr;
    Element const* ptr_x = nullptr;
    StrideMNL dTensor = {};
    Element const* ptr_gamma = nullptr;
    ElementAccumulator const* ptr_mean = nullptr;
    ElementAccumulator const* ptr_invstd = nullptr;
  };

  struct Params {
    Element const* ptr_grad = nullptr;
    Element const* ptr_x = nullptr;
    StrideMNL dTensor = {};
    Element const* ptr_gamma = nullptr;
    ElementAccumulator const* ptr_mean = nullptr;
    ElementAccumulator const* ptr_invstd = nullptr;
    Element null_default = Element(0);
  };

  template <class ProblemShape>
  static constexpr Params
  to_underlying_arguments(ProblemShape const& problem_shape, Arguments const& args) {
    return {args.ptr_grad, args.ptr_x, args.dTensor, 
            args.ptr_gamma, args.ptr_mean, args.ptr_invstd};
  }

  using SharedStorage = typename Base::SharedStorage;

  //
  // Constructor and data members
  //

  CUTLASS_HOST_DEVICE
  LayerNormBackwardReduction() { }

  CUTLASS_HOST_DEVICE
  LayerNormBackwardReduction(Params const& params, SharedStorage& shared_storage, int thread_idx)
    : params_ptr(&params),
    Base(thread_idx, shared_storage) { }
  
  Params const* params_ptr;

  //
  // Input cache
  //

  static int constexpr vec_bits = ThreadMap::kElementsPerAccess * sizeof_bits<Element>::value;
  using VecType = uint_bit_t<cute::min(128, vec_bits)>;
  static int constexpr VecLength = sizeof(VecType) / sizeof(Element);

  template <class GTensorGrad, class GTensorX, class CTensor, class ProblemShape>
  struct InputCache {
    CUTLASS_DEVICE
    InputCache(
      GTensorGrad&& tC_gGrad,
      GTensorX&& tC_gX,
      CTensor&& tC_cTensor,
      ProblemShape problem_shape,
      Params const* params_ptr
    ):
      tC_gGrad(cute::forward<GTensorGrad>(tC_gGrad)),
      tC_gX(cute::forward<GTensorX>(tC_gX)),
      tC_cTensor(cute::forward<CTensor>(tC_cTensor)),
      problem_shape(problem_shape),
      params_ptr(params_ptr),
      numel_factor(ElementAccumulator(1) / ElementAccumulator(get<1>(problem_shape))) 
      { }
    
    GTensorGrad tC_gGrad;
    GTensorX tC_gX;
    CTensor tC_cTensor;
    ProblemShape problem_shape;
    Params const* params_ptr;
    ElementAccumulator numel_factor;
    

    // Temporary scalars updated in each row
    Element mean;
    Element invstd;

    CUTLASS_DEVICE
    int get_row_idx(int row_idx) {
      return cute::get<2>(tC_cTensor(_0{}, row_idx));
    }

    CUTLASS_DEVICE
    int get_column_idx(int column_idx) {
      return cute::get<1>(tC_cTensor(column_idx, 0));
    }

    CUTLASS_DEVICE
    void begin_row(int row_idx) {
      int global_row_idx = get_row_idx(row_idx);
      if (global_row_idx < get<2>(problem_shape)) {
        mean = Element(*(params_ptr->ptr_mean + global_row_idx));
        invstd = Element(*(params_ptr->ptr_invstd + global_row_idx));
      }
    }

    CUTLASS_DEVICE
    Array<Element, VecLength> get_x_hat(int row_idx, int column_idx) {
      bool guard = elem_less(tC_cTensor(column_idx, row_idx), problem_shape);
      Array<Element, VecLength> value;
      VecType* value_ptr = reinterpret_cast<VecType*>(&value);
      cutlass::arch::global_load<VecType, sizeof(VecType)>(*value_ptr, (void*)&tC_gX(column_idx, row_idx), guard);
      value = (value - mean) * invstd;
      if (!guard) value.fill(params_ptr->null_default);
      return value;
    }

    CUTLASS_DEVICE
    Array<Element, VecLength> get_gamma_grad_y(int row_idx, int column_idx) {
      bool guard = elem_less(tC_cTensor(column_idx, row_idx), problem_shape);
      Array<Element, VecLength> value;
      VecType* value_ptr = reinterpret_cast<VecType*>(&value);
      cutlass::arch::global_load<VecType, sizeof(VecType)>(*value_ptr, (void*)&tC_gGrad(column_idx, row_idx), guard);
      Array<Element, VecLength> gamma;
      VecType* gamma_ptr = reinterpret_cast<VecType*>(&gamma);
      Element const* gmem_gamma_ptr = params_ptr->ptr_gamma + get_column_idx(column_idx);
      cutlass::arch::global_load<VecType, sizeof(VecType)>(*gamma_ptr, (void*)gmem_gamma_ptr, guard);
      value = gamma * value * invstd;
      if (!guard) value.fill(params_ptr->null_default);
      return value;
    }
  };

  template <
    class GTensorGrad, class GTensorX, 
    class RTensorGrad, class RTensorX,
    class CTensor, class ProblemShape>
  struct InputCacheReg : InputCache<GTensorGrad, GTensorX, CTensor, ProblemShape> {
    using Base = InputCache<GTensorGrad, GTensorX, CTensor, ProblemShape>;

    CUTLASS_DEVICE
    InputCacheReg(
      GTensorGrad&& tC_gGrad,
      GTensorX&& tC_gX,
      RTensorGrad&& tC_rGrad,
      RTensorX&& tC_rX,
      CTensor&& tC_cTensor,
      ProblemShape problem_shape,
      Params const* params_ptr
    ):
      Base(
        cute::move(tC_gGrad), cute::move(tC_gX), 
        cute::move(tC_cTensor), problem_shape, params_ptr),
      tC_rGrad(cute::forward<RTensorGrad>(tC_rGrad)),
      tC_rX(cute::forward<RTensorX>(tC_rX)) { }
    
    RTensorGrad tC_rGrad;
    RTensorX tC_rX;

    CUTLASS_DEVICE
    void begin_row(int row_idx) {
      Base::begin_row(row_idx);
      CUTLASS_PRAGMA_UNROLL
      for (int column_idx=0; column_idx < ThreadMap::kIterationColumn; ++column_idx) {
        tC_rGrad(column_idx) = Base::get_gamma_grad_y(row_idx, column_idx);
        tC_rX(column_idx) = Base::get_x_hat(row_idx, column_idx);
      }
    }

    CUTLASS_DEVICE
    Array<Element, VecLength> get_x_hat(int row_idx, int column_idx) {
      return tC_rX(column_idx);
    }

    CUTLASS_DEVICE
    Array<Element, VecLength> get_gamma_grad_y(int row_idx, int column_idx) {
      return tC_rGrad(column_idx);
    }
  };

  template <class ProblemShape>
  CUTLASS_DEVICE auto
  get_input_cache(
    gemm::GemmCoord threadblock_tile_offset,
    int thread_idx,
    ProblemShape problem_shape
  ) {
    Tensor mGrad = make_tensor(
      make_gmem_ptr(params_ptr->ptr_grad),
      problem_shape,
      params_ptr->dTensor
    );

    Tensor mX = make_tensor(
      make_gmem_ptr(params_ptr->ptr_x),
      problem_shape,
      params_ptr->dTensor
    );

    // ITERATION_COLUMN, ITERATION_ROW
    Tensor tC_gGrad = recast<VecType>(
      ThreadMap::partition(mGrad, thread_idx, threadblock_tile_offset))(_0{},_,_);
    
    Tensor tC_gX = recast<VecType>(
      ThreadMap::partition(mX, thread_idx, threadblock_tile_offset))(_0{},_,_);
    
    // Generate the pred tensor
    Tensor cTensor = make_identity_tensor(mGrad.shape());
    Tensor tC_cTensor = ThreadMap::partition(
      cTensor, thread_idx, threadblock_tile_offset)(_0{},_,_);
    
    if constexpr (!CacheInput) {
      return InputCache<
        decltype(tC_gGrad), decltype(tC_gX),
        decltype(tC_cTensor), decltype(problem_shape)>(
          cute::move(tC_gGrad),
          cute::move(tC_gX),
          cute::move(tC_cTensor),
          problem_shape,
          params_ptr
        );
    } else {
      Tensor tC_rGrad = recast<Array<Element, Alignment>>(
        make_tensor_like(tC_gGrad(_,_0{}))
      );
      Tensor tC_rX = recast<Array<Element, Alignment>>(
        make_tensor_like(tC_gX(_,_0{}))
      );

      return InputCacheReg<
        decltype(tC_gGrad), decltype(tC_gX),
        decltype(tC_rGrad), decltype(tC_rX),
        decltype(tC_cTensor), decltype(problem_shape)>(
          cute::move(tC_gGrad),
          cute::move(tC_gX),
          cute::move(tC_rGrad),
          cute::move(tC_rX),
          cute::move(tC_cTensor),
          problem_shape,
          params_ptr
        );
    }
  }

  //
  // Functions
  //

  template <class InputCache>
  CUTLASS_DEVICE
  void reduce(ReductionResult &reduction_result, InputCache &inputs, int row_idx) {
    using ConvertInput = NumericArrayConverter<ElementAccumulator, Element, VecLength>;
    ConvertInput convert_input{};

    Array<ElementAccumulator, VecLength> t1_accum;
    Array<ElementAccumulator, VecLength> t2_accum;
    t1_accum.clear();
    t2_accum.clear();

    CUTLASS_PRAGMA_UNROLL
    for (int column_idx=0; column_idx < ThreadMap::kIterationColumn; ++column_idx) {
      Array<Element, VecLength> gamma_grad_y = inputs.get_gamma_grad_y(row_idx, column_idx);
      Array<Element, VecLength> x_hat = inputs.get_x_hat(row_idx, column_idx);

      t1_accum = convert_input(gamma_grad_y) + t1_accum;
      t2_accum = convert_input(gamma_grad_y * x_hat) + t2_accum;
    }

    this->sum(t1_accum, reduction_result.sum_t1);
    this->sum(t2_accum, reduction_result.sum_t2);
  }

  template <class InputCache>
  CUTLASS_DEVICE
  Array<ElementAccumulator, VecLength> apply(ReductionResult &reduction_result, InputCache &inputs, int row_idx, int column_idx) {
    using ConvertInput = NumericArrayConverter<ElementAccumulator, Element, VecLength>;
    ConvertInput convert_input{};
    Array<ElementAccumulator, VecLength> gamma_grad_y = convert_input(
      inputs.get_gamma_grad_y(row_idx, column_idx));
    Array<ElementAccumulator, VecLength> x_hat = convert_input(
      inputs.get_x_hat(row_idx, column_idx));
    
    return gamma_grad_y - (
      (
        (x_hat * reduction_result.sum_t2) + reduction_result.sum_t1 
      ) * inputs.numel_factor
    );
  }


};

/////////////////////////////////////////////////////////////////////////////////////////////////

} // namespace threadblock
} // namespace layernorm
} // namespace cutlass

/////////////////////////////////////////////////////////////////////////////////////////////////