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

/*! \file
  \brief Visitor tree compute operations for the CUTLASS 2x epilogue
*/

#pragma once

#include "epilogue/threadblock/visitor.hpp"

/////////////////////////////////////////////////////////////////////////////////////////////////

namespace cutlass::epilogue::threadblock {

using namespace cute;
using namespace detail;

/////////////////////////////////////////////////////////////////////////////////////////////////

/////////////////////////////////////////////////////////////////////////////////////////////////
//
// N-nary Elementwise Compute Operation
//
/////////////////////////////////////////////////////////////////////////////////////////////////

template<
  template <class> class ComputeFn,
  class ElementOutput,
  class ElementCompute,
  FloatRoundStyle RoundStyle,
  class = void
>
struct VisitorCompute : VisitorImpl2x<> {

  using VisitorImpl2x<>::VisitorImpl2x;

  struct Callbacks : EmptyCallbacks {
    template <typename ElementAccumulator, typename... ElementInputs, int FragmentSize>
    CUTLASS_DEVICE Array<ElementOutput, FragmentSize>
    visit(int row_idx, int column_idx,
          Array<ElementAccumulator, FragmentSize> const& frg_acc,
          Array<ElementInputs, FragmentSize> const&... frg_inputs) {
      return transform_apply(cute::make_tuple(frg_inputs...),
        [&] (auto&& frg_input) {
          using ElementInput = typename cute::remove_cvref_t<decltype(frg_input)>::Element;
          using ConvertInput = NumericArrayConverter<ElementCompute, ElementInput, FragmentSize, RoundStyle>;
          ConvertInput convert_input{};

          return convert_input(frg_input);
        },
        [&] (auto&&... cvt_frg_inputs) {
          using ComputeOutput = ComputeFn<Array<ElementCompute, FragmentSize>>;
          using ConvertOutput = NumericArrayConverter<ElementOutput, ElementCompute, FragmentSize, RoundStyle>;
          ComputeOutput compute_output{};
          ConvertOutput convert_output{};

          return convert_output(compute_output(cvt_frg_inputs...));
        }
      );
    }

  };

  template <class ProblemShape>
  CUTLASS_DEVICE auto
  get_callbacks(
    gemm::GemmCoord threadblock_tile_offset,
    int thread_idx,
    ProblemShape problem_shape
  ) {
    return Callbacks();
  }

};

/////////////////////////////////////////////////////////////////////////////////////////////////

} // namespace cutlass::epilogue::threadblock

/////////////////////////////////////////////////////////////////////////////////////////////////

namespace cutlass{
  template <typename T, int N>
  struct greater_equal<Array<T, N>> {

    using OutputConvert = NumericArrayConverter<T, bool, N>;

    CUTLASS_HOST_DEVICE
    Array<T, N> operator()(Array<T, N> const &lhs, Array<T, N> const &rhs) const {

      Array<bool, N> result;
      greater_equal<T> scalar_op;

      OutputConvert converter{};

      CUTLASS_PRAGMA_UNROLL
      for (int i = 0; i < N; ++i) {
        result[i] = scalar_op(lhs[i], rhs[i]);
      }

      return converter(result);
    }

    CUTLASS_HOST_DEVICE
    Array<T, N> operator()(Array<T, N> const &lhs, T const &scalar) const {

      Array<bool, N> result;
      greater_equal<T> scalar_op;

      OutputConvert converter{};

      CUTLASS_PRAGMA_UNROLL
      for (int i = 0; i < N; ++i) {
        result[i] = scalar_op(lhs[i], scalar);
      }

      return converter(result);
    }

    CUTLASS_HOST_DEVICE
    Array<T, N> operator()( T const &scalar, Array<T, N> const &rhs) const {

      Array<bool, N> result;
      greater_equal<T> scalar_op;

      OutputConvert converter{};

      CUTLASS_PRAGMA_UNROLL
      for (int i = 0; i < N; ++i) {
        result[i] = scalar_op(scalar, rhs[i]);
      }

      return converter(result);
    }
  };
} // namespace cutlass