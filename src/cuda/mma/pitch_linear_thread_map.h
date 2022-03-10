/***************************************************************************************************
 * Copyright (c) 2017-2021, NVIDIA CORPORATION.  All rights reserved.
 *
 * Redistribution and use in source and binary forms, with or without modification, are permitted
 * provided that the following conditions are met:
 *     * Redistributions of source code must retain the above copyright notice, this list of
 *       conditions and the following disclaimer.
 *     * Redistributions in binary form must reproduce the above copyright notice, this list of
 *       conditions and the following disclaimer in the documentation and/or other materials
 *       provided with the distribution.
 *     * Neither the name of the NVIDIA CORPORATION nor the names of its contributors may be used
 *       to endorse or promote products derived from this software without specific prior written
 *       permission.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND ANY EXPRESS OR
 * IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND
 * FITNESS FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL NVIDIA CORPORATION BE LIABLE
 * FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING,
 * BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS;
 * OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT,
 * STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
 * OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 *
 **************************************************************************************************/
/*! \file
    \brief Templates implementing how threads are mapped to a given tile. 

*/

#pragma once

#include "cutlass/cutlass.h"
#include "cutlass/array.h"
#include "cutlass/coord.h"
#include "cutlass/predicate_vector.h"
#include "cutlass/tensor_ref.h"
#include "cutlass/tensor_view.h"
#include "cutlass/layout/pitch_linear.h"

////////////////////////////////////////////////////////////////////////////////

namespace cutlass {
namespace transform {


////////////////////////////////////////////////////////////////////////////////

/// Policy defining a warp-raked arrangement in which a shape is partitioned into contiguous
/// elements.
///
/// This ThreadMap is used by tensor core kernels.
template <
  typename Shape_,
  int Threads,
  typename WarpThreadArrangement_,
  int ElementsPerAccess = 1
>
struct PitchLinearWarpRakedThreadMapV2 {

  /// Tensor coordinate
  using TensorCoord = layout::PitchLinearCoord;

  /// Tile shape
  using Shape = Shape_;

  /// Number of threads total
  static int const kThreads = Threads;

  /// Extract vector length from Layout
  static int const kElementsPerAccess = ElementsPerAccess;

  /// Shape of access by each thread
  using ThreadAccessShape = layout::PitchLinearShape<kElementsPerAccess, 1>;

  /// Internal details made public to facilitate introspection
  struct Detail {

    /// Fixed arrangement of threads within a warp (units of threads).
    using WarpThreadArrangement = WarpThreadArrangement_;

    /// Number of threads per warp
    static int const kWarpSize = WarpThreadArrangement::kCount;

    /// Number of participating warps
    static int const kWarpCount = kThreads / kWarpSize;

    static_assert(
      !(Shape::kContiguous % kElementsPerAccess),
      "Shape must be divisible by vector length.");

    /// Compute the 'shape' of the overall tile in units of vectors
    using ShapeInAccesses = layout::PitchLinearShape<
      Shape::kContiguous / kElementsPerAccess,
      Shape::kStrided
    >;

    static_assert(
      !(ShapeInAccesses::kContiguous % WarpThreadArrangement::kContiguous),
      "ShapeInAccesses must be divisible by WarpThreadArrangement.");

    static_assert(
      !(ShapeInAccesses::kStrided % WarpThreadArrangement::kStrided),
      "ShapeInAccesses must be divisible by WarpThreadArrangement.");

    // compute number of warp-level accesses total
    using WarpAccessIterations = layout::PitchLinearShape<
      ShapeInAccesses::kContiguous / WarpThreadArrangement::kContiguous,
      ShapeInAccesses::kStrided / WarpThreadArrangement::kStrided
    >;

    // Divide it into the number of warps, first partitioning the strided dimension then the
    // contiguous.
    static int const kWarpsStrided =
        (WarpAccessIterations::kStrided >= kWarpCount
             ? kWarpCount
             : WarpAccessIterations::kStrided);

    static int const kWarpsContiguous =
        (kWarpCount > WarpAccessIterations::kStrided
             ? kWarpCount / kWarpsStrided
             : 1);

    /// Arrangement of warps within a threadblock-scoped tile
    using WarpArrangement = layout::PitchLinearShape<
      kWarpsContiguous, kWarpsStrided
    >;
  };

  ///< Iterations along each dimension (concept: PitchLinearShape)
  using Iterations = layout::PitchLinearShape<
    Detail::WarpAccessIterations::kContiguous / Detail::kWarpsContiguous,
    Detail::WarpAccessIterations::kStrided / Detail::kWarpsStrided
  >;

  static_assert(Iterations::kCount,
    "Number of iterations must be non-zero");

  ///< Delta betweeen accesses (units of elements, concept: PitchLinearShape)
  using Delta = layout::PitchLinearShape<
    Detail::WarpThreadArrangement::kContiguous * kElementsPerAccess,
    Detail::WarpThreadArrangement::kStrided
  >;

  /// Maps thread ID to a coordinate offset within the tensor's logical coordinate space
  CUTLASS_HOST_DEVICE
  static TensorCoord initial_offset(int thread_id) {

    int warp_id = (thread_id / Detail::kWarpSize);
    int lane_id = (thread_id % Detail::kWarpSize);

    //
    // compute warp-level offset
    //

    // This is the shape of the entire area covered by a warp's memory access (in units of vectors)
    layout::PitchLinearCoord warp_footprint{
      Detail::WarpThreadArrangement::kContiguous * Iterations::kContiguous,
      Detail::WarpThreadArrangement::kStrided * Iterations::kStrided
    };

    // This is the offset of a specific warp (in units of vectors)
    layout::PitchLinearCoord warp_offset{
      (warp_id % Detail::kWarpsContiguous),
      (warp_id / Detail::kWarpsContiguous)
    };

    // This is the offset of a specific thread within a warp (units of vectors)
    layout::PitchLinearCoord thread_offset_in_warp{
      lane_id % Detail::WarpThreadArrangement::kContiguous,
      lane_id / Detail::WarpThreadArrangement::kContiguous
    };

    // This is the offset of a thread within a threadblock tile (units of vectors)
    layout::PitchLinearCoord thread_offset_in_threadblock_tile_vec = 
      warp_footprint * warp_offset + thread_offset_in_warp;

    // This is the offset of a thread within a threadblock tile (units of elements)
    layout::PitchLinearCoord thread_offset_in_threadblock_tile_base{
      thread_offset_in_threadblock_tile_vec.contiguous() * kElementsPerAccess,
      thread_offset_in_threadblock_tile_vec.strided()
    };

    return thread_offset_in_threadblock_tile_base;
  }
};


} // namespace transform
} // namespace cutlass

////////////////////////////////////////////////////////////////////////////////