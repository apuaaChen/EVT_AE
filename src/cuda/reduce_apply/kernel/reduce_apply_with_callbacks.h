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

/////////////////////////////////////////////////////////////////////////////////////////////////

namespace cutlass {
namespace reduce_apply {
namespace kernel {

/////////////////////////////////////////////////////////////////////////////////////////////////

// The entry point of all reduce-apply kernels
template<
  class RedCallbacks,
  class EpiCallbacks,
  class ThreadMap
>
struct ReduceApplyWithCallbacks {

  static const int kElementsPerAccess = RedCallbacks::kElementsPerAccess;
  using ElementAccumulator = typename RedCallbacks::ElementAccumulator;

  struct Arguments {
    typename RedCallbacks::Arguments red_args;
    MatrixCoord problem_size;
    int batch_count;
    typename EpiCallbacks::Arguments epi_args;
  };

  struct Params {
    typename RedCallbacks::Params red_params;
    typename EpiCallbacks::Params epi_params;
    cute::Shape<int32_t, int32_t, int32_t> problem_shape;

    /// Constructor
    Params(Arguments const& args):
      problem_shape(
        {args.problem_size.row(), args.problem_size.column(), args.batch_count}),
      red_params(
        RedCallbacks::to_underlying_arguments(
          problem_shape, args.red_args)),
      epi_params(
        EpiCallbacks::to_underlying_arguments(
          problem_shape, args.epi_args, nullptr))
    { }
  };

  struct SharedStorage {
    typename RedCallbacks::SharedStorage red_smem;
    typename EpiCallbacks::SharedStorage epi_smem;
  };

  using ReductionResult = typename RedCallbacks::ReductionResult;

  /// Execute one reduce-apply op
  CUTLASS_DEVICE
  void operator()(
    Params const& params, 
    SharedStorage &shared_storage) {

    int thread_idx = threadIdx.x;
    // It is assumed that each row has a single threadblock
    // So that the blockIdx.x is assigned to rows
    gemm::GemmCoord threadblock_tile_offset{
      int(blockIdx.x), int(blockIdx.y), int(blockIdx.z)
    };

    // Instantiate callbacks
    RedCallbacks reduction_callbacks{
      params.red_params, shared_storage.red_smem, thread_idx};

    auto inputs = reduction_callbacks.get_input_cache(
      threadblock_tile_offset, thread_idx, params.problem_shape
    );

    EpiCallbacks epilogue{params.epi_params, shared_storage.epi_smem};
    auto epi_callbacks = epilogue.get_callbacks(
      threadblock_tile_offset, thread_idx, params.problem_shape
    );

    // Launch the kernel
    epi_callbacks.begin_epilogue();

    CUTLASS_PRAGMA_UNROLL
    for (int row_idx = 0; row_idx < ThreadMap::kRowsPerBlock; ++row_idx) {
      ReductionResult reduction_result;
      inputs.begin_row(row_idx);
      epi_callbacks.begin_row(row_idx);
      reduction_callbacks.reduce(reduction_result, inputs, row_idx);
      CUTLASS_PRAGMA_UNROLL
      for (int column_idx = 0; column_idx < ThreadMap::kIterationColumn; ++column_idx) {
        Array<ElementAccumulator, kElementsPerAccess> accum = reduction_callbacks.apply(
          reduction_result, inputs, row_idx, column_idx);
        epi_callbacks.visit(row_idx, column_idx, accum);
      }
      epi_callbacks.end_row(row_idx);
    }
    epi_callbacks.end_epilogue();
  }
};

/////////////////////////////////////////////////////////////////////////////////////////////////

} // namespace kernel
} // namespace reduce_apply
} // namespace cutlass

/////////////////////////////////////////////////////////////////////////////////////////////////