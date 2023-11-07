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
  \brief Visitor tree load operations for the CUTLASS 2x epilogue
*/

#pragma once

#include "epilogue/threadblock/visitor.hpp"
#include "cute/tensor.hpp"
#include "curand.h"
#include "curand_kernel.h"

/////////////////////////////////////////////////////////////////////////////////////////////////

namespace cutlass::epilogue::threadblock {

using namespace cute;
using namespace detail;

using X = Underscore;

/////////////////////////////////////////////////////////////////////////////////////////////////

/////////////////////////////////////////////////////////////////////////////////////////////////
//
// Elementwise Fetch Operations
//
/////////////////////////////////////////////////////////////////////////////////////////////////

// returns accumulator
struct VisitorAccFetch : VisitorImpl2x<> {

  using VisitorImpl2x<>::VisitorImpl2x;

  struct Callbacks : EmptyCallbacks {
    template <class ElementAccumulator, int FragmentSize>
    CUTLASS_DEVICE Array<ElementAccumulator, FragmentSize>
    visit(int row_idx, int column_idx, Array<ElementAccumulator, FragmentSize> const& frg_acc) {
      return frg_acc;
    }
  };

  template <class ProblemShape>
  CUTLASS_DEVICE auto
  get_callbacks(
    gemm::GemmCoord threadblock_tile_offset,
    int thread_idx,
    ProblemShape problem_shape
  ) {
    return Callbacks{};
  }
};


/////////////////////////////////////////////////////////////////////////////////////////////////
//
// Random Operation
//
/////////////////////////////////////////////////////////////////////////////////////////////////

template<
  class Element
>
struct VisitorRand {

  struct SharedStorage { };

  struct Arguments {
    uint64_t seed;
    uint64_t offset;
  };

  using Params = Arguments;

  template <class ProblemShape>
  static constexpr Params
  to_underlying_arguments(ProblemShape const& problem_shape, Arguments const& args, void* workspace) {
    return args;
  }

  CUTLASS_HOST_DEVICE
  VisitorRand() { }

  CUTLASS_HOST_DEVICE
  VisitorRand(Params const& params, SharedStorage const& shared_storage)
      : params_ptr(&params) { }
  
  Params const* params_ptr;

  struct Callbacks: EmptyCallbacks {
    CUTLASS_DEVICE
    Callbacks(
      uint64_t seed,
      uint64_t offset,
      int thread_idx
    ) {
      curand_init(seed + offset, uint64_t(thread_idx), 0, &state);
    }

    curandStatePhilox4_32_10_t state;

    template <class ElementAccumulator, int FragmentSize>
    CUTLASS_DEVICE auto // returns an Array
    visit(int row_idx, int column_idx, 
          Array<ElementAccumulator, FragmentSize> const& frg_acc) {

      Array<float, FragmentSize> rand;
      float4* rand_ptr = reinterpret_cast<float4*>(rand.data());

      CUTLASS_PRAGMA_UNROLL
      for (int i=0; i < FragmentSize; i+=4) {
        *rand_ptr = curand_uniform4(&state);
        rand_ptr ++;
      }

      Array<Element, FragmentSize> frg_rand;

      using ConvertOutput = NumericArrayConverter<Element, float, FragmentSize>;
      ConvertOutput convert_output{};

      return convert_output(rand);
    }

  };

  template <class ProblemShape>
  CUTLASS_DEVICE auto
  get_callbacks(
    gemm::GemmCoord threadblock_tile_offset,
    int thread_idx,
    ProblemShape problem_shape
  ) {
    return Callbacks(params_ptr->seed, params_ptr->offset, thread_idx);
  }

};


/////////////////////////////////////////////////////////////////////////////////////////////////
//
// Broadcast Load Operations
//
/////////////////////////////////////////////////////////////////////////////////////////////////
// Scalar broadcast
template<
  class Element,
  class StrideMNL = Stride<_0,_0,_0>,
  int BroadcastCount = 1,
  template <class> class ReductionFn = multiplies
>
struct VisitorScalarBroadcast {
  static_assert(
    (cute::is_same_v<StrideMNL, Stride<_0,_0,_0>>) || // scalar broadcast, e.g. alpha
    (cute::is_same_v<StrideMNL, Stride<_0,_0,_1>>) ||
    (cute::is_same_v<StrideMNL, Stride<_0,_0,int>>));  // batched scalar broadcast, e.g. per-batch alpha
  
  struct SharedStorage { };

  struct Arguments {
    Element scalars[BroadcastCount] = {};
    Element const* scalar_ptrs[BroadcastCount] = {};
    StrideMNL dScalar = {};
  };

  using Params = Arguments;

  template <class ProblemShape>
  static constexpr Params
  to_underlying_arguments(ProblemShape const& problem_shape, Arguments const& args, void* workspace) {
    return args;
  }

  CUTLASS_HOST_DEVICE
  VisitorScalarBroadcast() { }

  CUTLASS_HOST_DEVICE
  VisitorScalarBroadcast(Params const& params, SharedStorage const& shared_storage)
      : params_ptr(&params) {
    // Get the scalar for non-batched broadcast
    if constexpr (cute::is_same_v<StrideMNL, Stride<_0,_0,_0>>) {
      update_scalar();
    }
  }

  Element scalar;
  Params const* params_ptr;

  struct Callbacks: EmptyCallbacks {
    CUTLASS_DEVICE
    Callbacks(Element scalar)
      : scalar(scalar) {}
    
    Element scalar;

    template <class ElementAccumulator, int FragmentSize>
    CUTLASS_DEVICE auto // returns an Array
    visit(int row_idx, int column_idx, 
          Array<ElementAccumulator, FragmentSize> const& frg_acc) {
      Array<Element, FragmentSize> frg_scalar;
      frg_scalar.fill(scalar);

      return frg_scalar;
    }
  };

  template <class ProblemShape>
  CUTLASS_DEVICE auto
  get_callbacks(
    gemm::GemmCoord threadblock_tile_offset,
    int thread_idx,
    ProblemShape problem_shape
  ) {
    // Get the scalar for batched broadcast
    if constexpr (
      cute::is_same_v<StrideMNL, Stride<_0,_0,_1>> ||
      cute::is_same_v<StrideMNL, Stride<_0,_0,int>>) {
      update_scalar(threadblock_tile_offset.k());
    }
    return Callbacks(scalar);
  }

private:
  CUTLASS_DEVICE void
  update_scalar(int l_coord = 0) {
    int l_offset = l_coord * size<2>(params_ptr->dScalar);

    if (params_ptr->scalar_ptrs[0] != nullptr) {
      scalar = params_ptr->scalar_ptrs[0][l_offset];
    } else {
      // batch stride is ignored for nullptr fallback
      scalar = params_ptr->scalars[0];
    }

    // Do reduction over multiple broadcasts if necessary
    ReductionFn<Element> reduction_fn;
    CUTLASS_PRAGMA_UNROLL
    for (int i = 1; i < BroadcastCount; ++i) {
      if (params_ptr->scalar_ptrs[i] != nullptr) {
        scalar = reduction_fn(scalar, params_ptr->scalar_ptrs[i][l_offset]);
      } else {
        // batch stride is ignored for nullptr fallback
        scalar = reduction_fn(scalar, params_ptr->scalars[i]);
      }
    }
  }

};

/////////////////////////////////////////////////////////////////////////////////////////////////
//
// Elementwise Load Operations
//
/////////////////////////////////////////////////////////////////////////////////////////////////

template<
  class ThreadMap,
  class Element,
  class StrideMNL
>
struct VisitorAuxLoad{

  using ShapeL = decltype(repeat_like(get<2>(StrideMNL{}), int32_t(0)));
  struct Arguments {
    Element* ptr_aux = nullptr;
    Element null_default = Element(0);
    StrideMNL dAux = {};
    ShapeL sAux = {};
  };

  using Params = Arguments;

  template <class ProblemShape>
  static constexpr Params
  to_underlying_arguments(ProblemShape const& problem_shape, Arguments const& args, void* workspace) {
    if constexpr (!is_tuple<ShapeL>::value) {
      return {args.ptr_aux, args.null_default, args.dAux, get<2>(problem_shape)};
    } else {
      return args;
    }
  }

  // Software pipeline stages
  static const int Stages = ThreadMap::Stages;

  struct SharedStorage {};

  // Global load type
  static int constexpr vec_bits = ThreadMap::kElementsPerAccess * sizeof_bits<Element>::value;
  using VecType = uint_bit_t<cute::min(128, vec_bits)>;
  static int constexpr VecLength = sizeof(VecType) / sizeof(Element);  

  CUTLASS_HOST_DEVICE
  VisitorAuxLoad() { }

  CUTLASS_HOST_DEVICE
  VisitorAuxLoad(Params const& params, SharedStorage const& shared_storage)
    : params_ptr(&params) { }

  Params const* params_ptr;

  template <class GTensor, class CTensor, class ProblemShape>
  struct Callbacks : EmptyCallbacks {
    CUTLASS_DEVICE
    Callbacks(
      GTensor&& tC_gAux,
      CTensor&& tC_cAux,
      ProblemShape problem_shape,
      Params const* params_ptr
    ):
      tC_gAux(cute::forward<GTensor>(tC_gAux)),
      tC_cAux(cute::forward<CTensor>(tC_cAux)),
      problem_shape(problem_shape),
      params_ptr(params_ptr) { }

    GTensor tC_gAux;
    CTensor tC_cAux;
    Params const* params_ptr;
    ProblemShape problem_shape;

    template <class ElementAccumulator, int FragmentSize>
    CUTLASS_DEVICE auto // returns an Array
    visit(int row_idx, int column_idx, 
          Array<ElementAccumulator, FragmentSize> const& frg_acc) {
      bool guard = elem_less(tC_cAux(column_idx, row_idx), problem_shape);
      Array<Element, VecLength> value;
      VecType* value_ptr = reinterpret_cast<VecType*>(&value);
      cutlass::arch::global_load<VecType, sizeof(VecType)>(
        *value_ptr, (void const*)&tC_gAux(column_idx, row_idx), guard);
      if (!guard) value.fill(params_ptr->null_default);
      return value;
    }
  };

  template <class ProblemShape>
  CUTLASS_DEVICE auto
  get_callbacks(
    gemm::GemmCoord threadblock_tile_offset,
    int thread_idx,
    ProblemShape problem_shape
  ) { 
    Tensor mAux = make_tensor(
      make_gmem_ptr(params_ptr->ptr_aux), 
      make_shape(get<0>(problem_shape), get<1>(problem_shape), params_ptr->sAux),
      params_ptr->dAux);   // (M,N,L)
    // VECTOR, ITERATION_ROW, ITERATION_COLUMN
    Tensor tC_gAux = recast<VecType>(
      ThreadMap::partition(mAux, thread_idx, threadblock_tile_offset))(_0{},_,_);

    // Generate the pred tensor
    Tensor cAux = make_identity_tensor(mAux.shape());
    Tensor tC_cAux = ThreadMap::partition(
      cAux, thread_idx, threadblock_tile_offset)(_0{},_,_);

    return Callbacks<
      decltype(tC_gAux), decltype(tC_cAux), ProblemShape>(
      cute::move(tC_gAux),
      cute::move(tC_cAux),
      problem_shape,
      params_ptr
    );
  }
};

/////////////////////////////////////////////////////////////////////////////////////////////////

// Row vector broadcast
template<
  class ThreadMap,
  class Element,
  class StrideMNL
>
struct VisitorRowBroadcast {

  using ShapeL = decltype(repeat_like(get<2>(StrideMNL{}), int32_t(0)));
  struct Arguments {
    Element const* ptr_row = nullptr;
    Element null_default = Element(0);
    StrideMNL dRow = {};
    ShapeL sRow = {};
  };

  using Params = Arguments;

  template <class ProblemShape>
  static constexpr Params
  to_underlying_arguments(ProblemShape const& problem_shape, Arguments const& args, void* workspace) {
    if constexpr (!is_tuple<ShapeL>::value) {
      return {args.ptr_row, args.null_default, args.dRow, get<2>(problem_shape)};
    } else {
      return args;
    }
  }

  struct SharedStorage {};

  // Global load type
  static int constexpr vec_bits = ThreadMap::kElementsPerAccess * sizeof_bits<Element>::value;
  using VecType = uint_bit_t<cute::min(128, vec_bits)>;
  static int constexpr VecLength = sizeof(VecType) / sizeof(Element);

  CUTLASS_HOST_DEVICE
  VisitorRowBroadcast() { }

  CUTLASS_HOST_DEVICE
  VisitorRowBroadcast(Params const& params, SharedStorage const& shared_storage)
    : params_ptr(&params) { }
  
  Params const* params_ptr;

  template <class GTensor, class CTensor, class ProblemShape>
  struct Callbacks : EmptyCallbacks {
    CUTLASS_DEVICE
    Callbacks(
      GTensor&& tC_gRow,
      CTensor&& tC_cRow,
      ProblemShape problem_shape,
      Params const* params_ptr
    ):
      tC_gRow(cute::forward<GTensor>(tC_gRow)),
      tC_cRow(cute::forward<CTensor>(tC_cRow)),
      n(get<1>(problem_shape)),
      params_ptr(params_ptr) { }
    
    GTensor tC_gRow;
    CTensor tC_cRow;
    Params const* params_ptr;
    int n;

    template <class ElementAccumulator, int FragmentSize>
    CUTLASS_DEVICE auto // returns an Array
    visit(int row_idx, int column_idx,
          Array<ElementAccumulator, FragmentSize> const& frg_acc) {
      bool guard = get<1>(tC_cRow(column_idx)) < n;
      Array<Element, VecLength> value;
      VecType* value_ptr = reinterpret_cast<VecType*>(&value);
      cutlass::arch::global_load<VecType, sizeof(VecType)>(
        *value_ptr, (void const*)&tC_gRow(column_idx), guard);
      if (!guard) value.fill(params_ptr->null_default);
      return value;
    }
  };

  template <class ProblemShape>
  CUTLASS_DEVICE auto
  get_callbacks(
    gemm::GemmCoord threadblock_tile_offset,
    int thread_idx,
    ProblemShape problem_shape
  ) {
    Tensor mRow = make_tensor(
      make_gmem_ptr(params_ptr->ptr_row), 
      make_shape(get<0>(problem_shape), get<1>(problem_shape), params_ptr->sRow),
      params_ptr->dRow);   // (M,N,L)
    // VECTOR, ITERATION_ROW, ITERATION_COLUMN
    Tensor tC_gRow = recast<VecType>(
      ThreadMap::partition(mRow, thread_idx, threadblock_tile_offset))(_0{},_,_0{});

    // Generate the pred tensor
    Tensor cRow = make_identity_tensor(mRow.shape());
    Tensor tC_cRow = ThreadMap::partition(cRow, thread_idx, threadblock_tile_offset)(_0{},_,_0{});
    
    return Callbacks<
      decltype(tC_gRow), decltype(tC_cRow), ProblemShape>(
      cute::move(tC_gRow),
      cute::move(tC_cRow),
      problem_shape,
      params_ptr
    );
  }

};

/////////////////////////////////////////////////////////////////////////////////////////////////

// Column vector broadcast
template<
  class ThreadMap,
  class Element,
  class StrideMNL = Stride<_1,_0,_0>
>
struct VisitorColBroadcast {

  using ShapeL = decltype(repeat_like(get<2>(StrideMNL{}), int32_t(0)));
  struct Arguments {
    Element const* ptr_col = nullptr;
    Element null_default = Element(0);
    StrideMNL dCol = {};
    ShapeL sCol = {};
  };

  using Params = Arguments;

  template <class ProblemShape>
  static constexpr Params
  to_underlying_arguments(ProblemShape const& problem_shape, Arguments const& args, void* workspace) {
    if constexpr (!is_tuple<ShapeL>::value) {
      return {args.ptr_col, args.null_default, args.dCol, get<2>(problem_shape)};
    } else {
      return args;
    }
  }

  struct SharedStorage { };

  CUTLASS_HOST_DEVICE
  VisitorColBroadcast() { }

  CUTLASS_HOST_DEVICE
  VisitorColBroadcast(Params const& params, SharedStorage const& shared_storage)
    : params_ptr(&params) { }
  
  Params const* params_ptr;

  template <class GTensor, class RTensor, class CTensor, class ProblemShape>
  struct Callbacks : EmptyCallbacks {
    CUTLASS_DEVICE
    Callbacks(
      GTensor&& tC_gCol,
      RTensor&& tC_rCol,
      CTensor&& tC_cCol,
      ProblemShape problem_shape,
      Params const* params_ptr
    ):
      tC_gCol(cute::forward<GTensor>(tC_gCol)),
      tC_rCol(cute::forward<RTensor>(tC_rCol)),
      tC_cCol(cute::forward<CTensor>(tC_cCol)),
      m(get<0>(problem_shape)),
      params_ptr(params_ptr) { }
    
    GTensor tC_gCol;
    RTensor tC_rCol;
    CTensor tC_cCol;
    Params const* params_ptr;
    int m;

    CUTLASS_DEVICE void
    begin_epilogue() {
      clear(tC_rCol);
      Tensor pred = make_tensor<bool>(shape(tC_gCol));
      CUTLASS_PRAGMA_UNROLL
      for (int i = 0; i < size(pred); ++i) {
        pred(i) = get<0>(tC_cCol(i)) < m;
      }
      copy_if(pred, tC_gCol, tC_rCol);
    }

    template <class ElementAccumulator, int FragmentSize>
    CUTLASS_DEVICE auto // returns an Array
    visit(int row_idx, int column_idx, 
          Array<ElementAccumulator, FragmentSize> const& frg_acc) {
      Array<Element, FragmentSize> frg_col;
      frg_col.fill(tC_rCol(row_idx));
      return frg_col;
    }
  };

  template <class ProblemShape>
  CUTLASS_DEVICE auto
  get_callbacks(
    gemm::GemmCoord threadblock_tile_offset,
    int thread_idx,
    ProblemShape problem_shape
  ) {
    Tensor mCol = make_tensor(
      make_gmem_ptr(params_ptr->ptr_col), 
      make_shape(get<0>(problem_shape), get<1>(problem_shape), params_ptr->sCol),
      params_ptr->dCol);   // (M,N,L)

    // ITERATION_ROW, VECTOR, ITERATION_COLUMN
    Tensor tC_gCol = ThreadMap::partition(
      mCol, thread_idx, threadblock_tile_offset)(_0{},_0{},_);
    Tensor tC_rCol = make_tensor_like(tC_gCol);

    // Generate the pred tensor
    Tensor cCol = make_identity_tensor(mCol.shape());
    Tensor tC_cCol = ThreadMap::partition(
      cCol, thread_idx, threadblock_tile_offset)(_0{},_0{},_);

    return Callbacks<
      decltype(tC_gCol), decltype(tC_rCol),
      decltype(tC_cCol), ProblemShape>(
      cute::move(tC_gCol),
      cute::move(tC_rCol),
      cute::move(tC_cCol),
      problem_shape,
      params_ptr
    );
  }
};

/////////////////////////////////////////////////////////////////////////////////////////////////

} // namespace cutlass::epilogue::threadblock

/////////////////////////////////////////////////////////////////////////////////////////////////
