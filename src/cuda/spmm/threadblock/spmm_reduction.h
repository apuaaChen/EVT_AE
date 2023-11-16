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

/////////////////////////////////////////////////////////////////////////////////////////////////

namespace cutlass {
namespace spmm {
namespace threadblock {

using namespace cute;
using cute::tuple;
using X = Underscore;


template <
  class Element,
  int Alignment,
  class ElementIndex,
  class ElementAccumulator_,
  class ThreadMap
>
struct SpmmRowBalanceReduction {

  using ElementAccumulator = ElementAccumulator_;

  using CtaShapeNK = cute::Shape<
    Int<ThreadMap::kColumnsPerBlock>,
    _1
  >;

  using ThreadShapeNK = cute::Shape<
    Int<Alignment>,
    _1,
  >;

  using StrideNK = Stride<_1, int64_t>;
  using ShapeNK = Shape<int32_t, int32_t>;

  struct Arguments {
    Element const* ptr_embedding = nullptr;
    StrideNK dEmb = {};
    ShapeNK sEmb = {};
    ElementIndex const* ptr_row;
    ElementIndex const* ptr_indices;
    Element const* ptr_edge_weight;
  };

  using Params = Arguments;

  struct SharedStorage {};

  struct ReductionResult {};

  static int constexpr vec_bits = ThreadMap::kElementsPerAccess * sizeof_bits<Element>::value;
  using VecType = uint_bit_t<cute::min(128, vec_bits)>;
  static int constexpr VecLength = sizeof(VecType) / sizeof(Element);

  template <class ProblemShape>
  static constexpr Params
  to_underlying_arguments(ProblemShape const& problem_shape, Arguments const& args) {
    return args;
  }

  //
  // Constructor and data members
  //

  CUTLASS_HOST_DEVICE
  SpmmRowBalanceReduction() { }

  CUTLASS_HOST_DEVICE
  SpmmRowBalanceReduction(Params const& params, SharedStorage& shared_storage, int thread_idx)
    : params_ptr(&params) { }
  
  Params const* params_ptr;
  

  //
  // Functions
  //
  template <class ProblemShape>
  CUTLASS_DEVICE
  Array<ElementAccumulator, Alignment> reduce(
    gemm::GemmCoord threadblock_tile_offset, 
    ProblemShape problem_shape,
    int thread_idx) {

    Tensor cInput = make_identity_tensor(problem_shape);
    // VECTOR
    Tensor tC_cInput = ThreadMap::partition(cInput, thread_idx, threadblock_tile_offset);
    int row_idx = get<2>(tC_cInput(_0{},_0{},_0{}));

    auto [column_coord, row_coord] = ThreadMap::tid2coord(thread_idx);

    if (!elem_less(tC_cInput(_0{},_0{},_0{}), problem_shape)) return;

    // grab start value
    ElementIndex start = params_ptr->ptr_row[row_idx];
    ElementIndex end = params_ptr->ptr_row[row_idx + 1];

    // (N, K)
    Tensor mEmb = make_tensor(
      make_gmem_ptr(params_ptr->ptr_embedding),
      params_ptr->sEmb,
      params_ptr->dEmb
    );

    // (kColumnsPerBlock, K)
    Tensor bC_gEmb = local_tile(
      mEmb, CtaShapeNK{}, make_coord(_,_), Step<_1,_1>{}
    )(_,_0{},_0{},_);

    Tensor tC_gEmb = recast<VecType>(local_tile(
      bC_gEmb, ThreadShapeNK{}, make_coord(_,_), Step<_1,_1>{}
    )(_,_0{},column_coord,_))(_0{},_);

    using ConvertInput = NumericArrayConverter<ElementAccumulator, Element, VecLength>;
    ConvertInput convert_input{};

    Array<ElementAccumulator, Alignment> accum;

    accum.clear();

    for (int i=start; i < end; ++i) {
      ElementIndex r_idx = *(params_ptr->ptr_indices + i);
      ElementAccumulator edge_weight = ElementAccumulator(*(params_ptr->ptr_edge_weight + i));
      // Step 1: load the vector to accumulator
      Array<Element, Alignment> value;
      VecType* value_ptr = reinterpret_cast<VecType*>(&value);
      cutlass::arch::global_load<VecType, sizeof(VecType)>(*value_ptr, (void*)&tC_gEmb(r_idx), true);
      accum = accum + convert_input(value) * edge_weight;
    }

    return accum;
  }

};


/////////////////////////////////////////////////////////////////////////////////////////////////
// ThreadLayout

template<
  int EmbAlignment,       // Alignment of the embedding
  class ThreadblockShape  // The tile size handled by each threadblock
>
struct OutputTileThreadLayoutSubwarp {
  static const int kRowsPerBlock = ThreadblockShape::kRow;
  static const int kColumnsPerBlock = ThreadblockShape::kColumn;

  static const int kThreadsPerRow = kColumnsPerBlock / EmbAlignment;
  static const int kNumThreads = kRowsPerBlock * kThreadsPerRow;

  static const int kElementsPerAccess = EmbAlignment;

  // The shape of CTA Tile
  using CtaShapeMNL = cute::Shape<
    _1,
    Int<kColumnsPerBlock>,
    Int<kRowsPerBlock>
  >;

  // The shape of CTA Tile
  using ThreadMapShape = cute::Shape<
    // N
    Int<EmbAlignment>,
    Int<kThreadsPerRow>,
    _1,                // ITERATION_COLUMN
    // L
    Int<kRowsPerBlock>,
    _1                 // ITERATION_ROW
  >;

  using ThreadShape = cute::Shape<
    Int<kThreadsPerRow>,
    Int<kRowsPerBlock>
  >;

  CUTLASS_DEVICE
  static auto tid2coord(int thread_idx) {
    return make_layout(ThreadShape{})[thread_idx];
  }
  

  template <class TensorInput>
  CUTLASS_DEVICE
  static auto partition(TensorInput &&xT, int thread_idx, gemm::GemmCoord threadblock_tile_offset) {
    // (kColumnsPerBlock, kRowsPerBlock)
    Tensor bCxT = local_tile(
      xT, CtaShapeMNL{}, make_coord(_,_,_), Step<_1,_1,_1>{}
    )(_0{},_,_,threadblock_tile_offset.m(),threadblock_tile_offset.n(), threadblock_tile_offset.k());

    // Transform to column-major
    Tensor tCxT = bCxT.compose(make_layout(ThreadMapShape{}));

    auto [lane_col_coord, lane_row_coord] = tid2coord(thread_idx);

    return tCxT(_,lane_col_coord,_,lane_row_coord,_);
  }

};

/////////////////////////////////////////////////////////////////////////////////////////////////

} // namespace threadblock
} // namespace spmm
} // namespace cutlass

/////////////////////////////////////////////////////////////////////////////////////////////////