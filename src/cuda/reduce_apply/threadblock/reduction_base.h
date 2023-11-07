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
#include <cub/block/block_reduce.cuh>
#include <cub/warp/warp_reduce.cuh>


////////////////////////////////////////////////////////////////////////////////

namespace cutlass {

template <typename T, int N>
CUTLASS_DEVICE
Array<T, N> operator+(Array<T, N> const &a, T const &b) {
    cutlass::plus<Array<T, N>> plus_op_;
    return plus_op_(a, b);
}

template <typename T, int N>
CUTLASS_DEVICE
Array<T, N> operator-(Array<T, N> const &a, T const &b) {
    cutlass::minus<Array<T, N>> minus_op_;
    return minus_op_(a, b);
}

template <typename T, int N>
CUTLASS_DEVICE
Array<T, N> operator/(Array<T, N> const &a, T const &b) {
    cutlass::divides<Array<T, N>> div_op_;
    return div_op_(a, b);
}

template <typename T, int N>
CUTLASS_DEVICE
Array<T, N> exp(Array<T, N> const &a) {
    cutlass::fast_exp_op<Array<T, N>> exp_op_;
    return exp_op_(a);
}

template <typename T, int N>
CUTLASS_DEVICE
Array<T, N> max(Array<T, N> const &a, Array<T, N> const &b) {
    cutlass::maximum<Array<T, N>> maximum_op_;
    return maximum_op_(a, b);
}

////////////////////////////////////////////////////////////////////////////////

namespace reduce_apply {
namespace threadblock {

////////////////////////////////////////////////////////////////////////////////

// Perform reduction within the threadblock
template<
  typename ElementAccumulator,
  int kNumThreads>
struct ReductionBase {
public:

  //
  // Reduction
  //
  using BlockReduce = cub::BlockReduce<ElementAccumulator, kNumThreads>;

  union SharedStorage {
    typename BlockReduce::TempStorage temp_storage;
    ElementAccumulator broadcast_buffer;
  };

private:

  /// Reduction related members
  BlockReduce block_reduce_;
  ElementAccumulator* broadcast_buffer_ptr_;
  int thread_idx_;   

public:
  //
  // Constructor
  //
  CUTLASS_DEVICE
  ReductionBase (
    int thread_idx,
    SharedStorage & shared_storage
  ):
    block_reduce_(shared_storage.temp_storage),
    broadcast_buffer_ptr_(&shared_storage.broadcast_buffer),
    thread_idx_(thread_idx) { }

  /// Sum reduction
  template <int FragmentSize>
  CUTLASS_DEVICE
  void sum(Array<ElementAccumulator, FragmentSize> & partial_sum, 
    ElementAccumulator &row_sum) {
    typedef ElementAccumulator ArrayAcc[FragmentSize];
    row_sum = block_reduce_.Sum(
      *(reinterpret_cast<ArrayAcc*>(partial_sum.raw_data()))
    );

    if (thread_idx_ == 0) *(broadcast_buffer_ptr_) = row_sum;
    __syncthreads();
    row_sum = *(broadcast_buffer_ptr_);
  }

  /// Max reduction
  template <int FragmentSize>
  CUTLASS_DEVICE
  void max(Array<ElementAccumulator, FragmentSize> & partial_max,
    ElementAccumulator & row_max) {
    typedef ElementAccumulator ArrayAcc[FragmentSize];
    row_max = block_reduce_.Reduce(
        *(reinterpret_cast<ArrayAcc*>(partial_max.raw_data())), cub::Max());

    if (thread_idx_ == 0) *(broadcast_buffer_ptr_) = row_max;
    __syncthreads();
    row_max = *(broadcast_buffer_ptr_);
  }

};


////////////////////////////////////////////////////////////////////////////////
// ThreadLayout

using namespace cute;
using cute::tuple;
using X = Underscore;

template <
  int kNumThreads_,
  typename Element, 
  int Alignment,
  typename ThreadblockShape  // The tile size handled by each threadblock
>
struct OutputTileThreadLayout1D {
  static const int kNumThreads = kNumThreads_;
  static const int kRowsPerBlock = ThreadblockShape::kRow;
  static const int kColumnsPerBlock = ThreadblockShape::kColumn;

  static const int kElementsPerAccess = Alignment;

  // The shape of CTA Tile
  using CtaShapeMNL = cute::Shape<
    Int<kRowsPerBlock>, 
    Int<kColumnsPerBlock>,
    _1
  >;

  static const int kIterationColumn = kColumnsPerBlock / (kElementsPerAccess * kNumThreads);
  static_assert(kIterationColumn >= 1);

  // The shape of CTA Tile
  using ThreadMapShape = cute::Shape<
    // Column
    Int<kElementsPerAccess>,
    Int<kNumThreads>,
    Int<kIterationColumn>,
    // Row
    Int<kRowsPerBlock>
  >;

  template <class TensorInput>
  CUTLASS_DEVICE
  static auto partition(TensorInput &&xT, int thread_idx, gemm::GemmCoord threadblock_tile_offset) {
    // (kRowsPerBlock, kColumnsPerBlock)
    Tensor bCxT = local_tile(
      xT, CtaShapeMNL{}, make_coord(_,_,_), Step<_1,_1,X>{}
    )(_,_,threadblock_tile_offset.m(),threadblock_tile_offset.n(), threadblock_tile_offset.k());
    
    // Transform to column-major
    // VECTOR, THREADS, ITERATION_COLUMN, ITERATION_ROW
    Tensor bCxT_nm = make_tensor(
      std::forward<decltype(bCxT)>(bCxT).data(), make_layout(get<1>(bCxT.layout()), get<0>(bCxT.layout()))
    ).compose(make_layout(ThreadMapShape{}));

    return bCxT_nm(_,thread_idx,_,_);
  }
};


// Specialized for kNumThreads=32
template<
  typename ElementAccumulator>
struct ReductionBase<ElementAccumulator, 32> {
public:
  //
  // Reduction
  //
  using WarpReduce = cub::WarpReduce<ElementAccumulator>;
  union SharedStorage {
    typename WarpReduce::TempStorage temp_storage;
  };

private:
  WarpReduce warp_reduce_;
  int thread_idx_;

public:
  //
  // Constructor
  //
  CUTLASS_DEVICE
  ReductionBase (
    int thread_idx,
    SharedStorage & shared_storage
  ):
    warp_reduce_(shared_storage.temp_storage),
    thread_idx_(thread_idx) { }

  /// Sum reduction
  template <int FragmentSize>
  CUTLASS_DEVICE
  void sum(Array<ElementAccumulator, FragmentSize> & partial_sum, ElementAccumulator &row_sum) {
    /// Get the sum in the array of each thread
    ElementAccumulator row_sum_local = ElementAccumulator(0);
    #pragma unroll
    for (int i = 0; i < FragmentSize; i ++) {
      row_sum_local += *(partial_sum.data() + i);
    }
    row_sum = warp_reduce_.Sum(row_sum_local);

    row_sum = __shfl_sync(0xffffffff, row_sum, 0);
  }

  /// Max reduction
  template <int FragmentSize>
  CUTLASS_DEVICE
  void max(Array<ElementAccumulator, FragmentSize> & partial_max, ElementAccumulator & row_max) {
    ElementAccumulator row_max_local = ElementAccumulator(-1e+6);
    #pragma unroll
    for (int i = 0; i < FragmentSize; i ++) {
      row_max_local = *(partial_max.data() + i) < row_max_local ? row_max_local : *(partial_max.data() + i);
    }

    row_max = warp_reduce_.Reduce(row_max_local, cub::Max());

    row_max = __shfl_sync(0xffffffff, row_max, 0);
  }
};

////////////////////////////////////////////////////////////////////////////////

} // namespace threadblock
} // namespace reduce_apply
} // namespace cutlass

////////////////////////////////////////////////////////////////////////////////