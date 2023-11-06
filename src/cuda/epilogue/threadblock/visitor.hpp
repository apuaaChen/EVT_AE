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
#include "cutlass/numeric_conversion.h"
#include "cutlass/arch/memory.h"
#include "cutlass/epilogue/fusion/sm90_visitor_tma_warpspecialized.hpp"

/////////////////////////////////////////////////////////////////////////////////////////////////

namespace cutlass::epilogue::threadblock {

using namespace cute;
using cute::tuple;

/////////////////////////////////////////////////////////////////////////////////////////////////

namespace detail {
template <class... Ops>
struct VisitorImpl2x: fusion::detail::Sm90VisitorImplBase<Ops...> {
  using fusion::detail::Sm90VisitorImplBase<Ops...>::Sm90VisitorImplBase;
  using fusion::detail::Sm90VisitorImplBase<Ops...>::ops;

  template <class CallbacksTuple>
  struct Callbacks {
    // Callbacks can store non-persistent variables (e.g. tensors) or copies of persistent variables
    CallbacksTuple callbacks_tuple;

    /// Called at the start of the epilogue just before iterating over accumulator slices
    CUTLASS_DEVICE void
    begin_epilogue() {
      for_each(callbacks_tuple,
        [] (auto& callbacks) {
          callbacks.begin_epilogue();
        }
      );
    }

    /// Called at the start of a row
    CUTLASS_DEVICE void
    begin_row(int row_idx) {
      for_each(callbacks_tuple,
        [&] (auto& callbacks) {
          callbacks.begin_row(row_idx);
        }
      );
    }

    /// Called after accumulators have been exchanged for each accumulator vector
    template <typename ElementAccumulator, typename... ElementInputs, int FragmentSize>
    CUTLASS_DEVICE auto // returns an Array
    visit(int row_idx, int column_idx, 
          Array<ElementAccumulator, FragmentSize> const& frg_acc,
          Array<ElementInputs, FragmentSize> const&... frg_inputs) // depends on the N-naryness of the op
      = delete; // Must be implemented for each operation
    
    /// Called at the end of a row
    CUTLASS_DEVICE void
    end_row(int row_idx) {
      for_each(callbacks_tuple,
        [&] (auto& callbacks) {
          callbacks.end_row(row_idx);
        }
      );
    }

    /// Called after all steps have been completed
    CUTLASS_DEVICE void
    end_epilogue() {
      for_each(callbacks_tuple,
        [] (auto& callbacks) {
          callbacks.end_epilogue();
        }
      );
    }
  };

  // Callbacks factory
  // All operations must redefine this
  template <class ProblemShape>
  CUTLASS_DEVICE auto
  get_callbacks(
    MatrixCoord threadblock_tile_offset,
    int thread_idx,
    ProblemShape problem_shape
  ) {
    return transform_apply(ops,
      [&] (auto& op) {
        return op.get_callbacks(
          threadblock_tile_offset,
          thread_idx,
          problem_shape);
      },
      [] (auto&&... callbacks) {
        auto callbacks_tuple = cute::make_tuple(callbacks...);
        return Callbacks<decltype(callbacks_tuple)>{callbacks_tuple};
      }
    );
  }
};

/////////////////////////////////////////////////////////////////////////////////////////////////

// Convenience aliases
using EmptyCallbacks = VisitorImpl2x<>::Callbacks<cute::tuple<>>;

/////////////////////////////////////////////////////////////////////////////////////////////////

} // namespace detail

using namespace detail;

/////////////////////////////////////////////////////////////////////////////////////////////////
//
// Tree visitor
//
/////////////////////////////////////////////////////////////////////////////////////////////////

template <class NodeOp, class... ChildOps>
struct TreeVisitor2x : VisitorImpl2x<ChildOps..., NodeOp> {

  using VisitorImpl2x<ChildOps..., NodeOp>::VisitorImpl2x;

  template<class CallbacksImpl>
  struct Callbacks : CallbacksImpl {
    CUTLASS_DEVICE
    Callbacks(CallbacksImpl&& impl)
      : CallbacksImpl(cute::forward<CallbacksImpl>(impl)) {}
    
    using CallbacksImpl::callbacks_tuple;

    template <typename ElementAccumulator, int FragmentSize>
    CUTLASS_DEVICE auto
    visit(int row_idx, int column_idx,
          Array<ElementAccumulator, FragmentSize> const& frg_acc) {
      constexpr int Rm1 = sizeof...(ChildOps);
      return cute::detail::tapply(callbacks_tuple,
        [&] (auto& child_callbacks) {
          return child_callbacks.visit(row_idx, column_idx, frg_acc);
        },
        [&] (auto&&... frg_inputs) {
          return get<Rm1>(callbacks_tuple).visit(row_idx, column_idx, frg_acc, frg_inputs...);
        },
        make_seq<Rm1>{}
      );
    }
  };

  // Callbacks factory
  template <class ProblemShape>
  CUTLASS_DEVICE auto
  get_callbacks(
    MatrixCoord threadblock_tile_offset,
    int thread_idx,
    ProblemShape problem_shape
  ) {
    return Callbacks<
    decltype(VisitorImpl2x<ChildOps..., NodeOp>::
      get_callbacks(
        threadblock_tile_offset,
        thread_idx,
        problem_shape
      ))>(
      VisitorImpl2x<ChildOps..., NodeOp>::
      get_callbacks(
        threadblock_tile_offset,
        thread_idx,
        problem_shape
      )
    );
  }
};

/////////////////////////////////////////////////////////////////////////////////////////////////
template<
  class ElementCompute,
  class EdgeTuple,
  class... Ops
>
struct TopologicalVisitor2x : VisitorImpl2x<Ops...> {
  static_assert(is_static_v<EdgeTuple>);
  static_assert(rank(EdgeTuple{}) == sizeof...(Ops));
  static_assert(sizeof...(Ops) > 1);

  using VisitorImpl2x<Ops...>::VisitorImpl2x;

  template<class CallbacksImpl>
  struct Callbacks : CallbacksImpl {
    CUTLASS_DEVICE
    Callbacks(CallbacksImpl&& impl)
      : CallbacksImpl(cute::forward<CallbacksImpl>(impl)) {}
    
    using CallbacksImpl::callbacks_tuple;

    template <typename ElementAccumulator, int FragmentSize>
    CUTLASS_DEVICE auto
    visit(int row_idx, int column_idx,
          Array<ElementAccumulator, FragmentSize> const& frg_acc) {
      constexpr int Rm1 = sizeof...(Ops) - 1;
      auto frg_compute_tuple = cute::repeat<Rm1>(Array<ElementCompute, FragmentSize>{});
      
      return cute::detail::tapply(EdgeTuple{}, callbacks_tuple, frg_compute_tuple,
        // Visit the first R-1 ops in topological order
        [&] (auto&& edge_seq, auto& callbacks, auto& frg_compute) {
          frg_compute = cute::detail::apply(frg_compute_tuple,
          // Compute the current op with children inputs
          [&] (auto const&... frg_inputs) {
            auto frg_output = callbacks.visit(row_idx, column_idx, frg_acc, frg_inputs...);
            using ElementOutput = typename decltype(frg_output)::Element;
            using ConvertOutput = NumericArrayConverter<ElementCompute, ElementOutput, FragmentSize>;
            ConvertOutput convert_output{};

            return convert_output(frg_output);
          },
          // Get inputs in the sequence given by the children indices of the current op
          edge_seq
        );
        return frg_compute;
      },
      // Visit the last op
      [&] (auto const&...) {
        return cute::detail::apply(frg_compute_tuple,
          // Compute the last op with children inputs
          [&] (auto const&... frg_inputs) {
            return get<Rm1>(callbacks_tuple).visit(row_idx, column_idx, frg_acc, frg_inputs...);
          },
          // Get inputs in the sequence given by the children indices of the last op
          get<Rm1>(EdgeTuple{})
        );
      },
      // Transform to visit R-1 ops, apply to visit last op
      make_seq<Rm1>{}
      );
    }
  };

  // Callbacks factory
  template <class ProblemShape>
  CUTLASS_DEVICE auto
  get_callbacks(
    MatrixCoord threadblock_tile_offset,
    int thread_idx,
    ProblemShape problem_shape
  ) {
    return Callbacks<decltype(
      VisitorImpl2x<Ops...>::
      get_callbacks(
        threadblock_tile_offset,
        thread_idx,
        problem_shape
      ))>(
      VisitorImpl2x<Ops...>::
      get_callbacks(
        threadblock_tile_offset,
        thread_idx,
        problem_shape
      )
    );
  }
};


template <class NodeOp, class... ChildOps>
using Sm80EVT = TreeVisitor2x<NodeOp, ChildOps...>;

template<
  class ElementCompute,
  class EdgeTuple,
  class... Ops
>
using Sm80TopologicalVisitor = TopologicalVisitor2x<ElementCompute, EdgeTuple, Ops...>;

/////////////////////////////////////////////////////////////////////////////////////////////////

} // namespace cutlass::epilogue::threadblock

/////////////////////////////////////////////////////////////////////////////////////////////////