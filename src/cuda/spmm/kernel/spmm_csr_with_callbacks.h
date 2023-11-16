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
#include "reduce_apply/kernel/reduce_apply_with_callbacks.h"

/////////////////////////////////////////////////////////////////////////////////////////////////

namespace cutlass {
namespace spmm {
namespace kernel {

/////////////////////////////////////////////////////////////////////////////////////////////////


// SpMM kernel with Row Balance
template<
  class RedCallbacks,
  class EpiCallbacks,
  class ThreadMap 
>
struct SpmmRowBalanceWithCallbacks:
  reduce_apply::kernel::ReduceApplyWithCallbacks<
  RedCallbacks, EpiCallbacks, ThreadMap> 
{
  using Base = reduce_apply::kernel::ReduceApplyWithCallbacks<
    RedCallbacks, EpiCallbacks, ThreadMap>;
  
  using Arguments = typename Base::Arguments;
  using Params = typename Base::Params;
  using SharedStorage = typename Base::SharedStorage;
  using ElementAccumulator = typename Base::ElementAccumulator;
  static const int kElementsPerAccess = ThreadMap::kElementsPerAccess;

  /// Execute one spmm op
  CUTLASS_DEVICE
  void operator()(
    Params const& params,
    SharedStorage &shared_storage) {

    int thread_idx = threadIdx.x;

    // Instantiate callbacks
    RedCallbacks reduction_callbacks{
      params.red_params, shared_storage.red_smem, thread_idx
    };

    EpiCallbacks epilogue{params.epi_params, shared_storage.epi_smem};

    gemm::GemmCoord threadblock_tile_offset{
      int(blockIdx.z), int(blockIdx.y), int(blockIdx.x)
    };

    auto epi_callbacks = epilogue.get_callbacks(
      threadblock_tile_offset, thread_idx, params.problem_shape
    );

    // Launch the kernel
    Array<ElementAccumulator, kElementsPerAccess> accum = reduction_callbacks.reduce(
      threadblock_tile_offset, params.problem_shape, thread_idx
    );

    epi_callbacks.begin_epilogue();
    epi_callbacks.begin_row(0);
    epi_callbacks.visit(0, 0, accum);
    epi_callbacks.end_row(0);
    epi_callbacks.end_epilogue();

  }
};

/////////////////////////////////////////////////////////////////////////////////////////////////

} // namespace kernel
} // namespace spmm
} // namespace cutlass

/////////////////////////////////////////////////////////////////////////////////////////////////