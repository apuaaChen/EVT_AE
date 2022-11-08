#pragma once

#include "cutlass/cutlass.h"
#include "softmax/threadblock/softmax_reduction.h"
#include "softmax/epilogue/epilogue.h"
#include "stdio.h"



/////////////////////////////////////////////////////////////////////////////////////////////////

namespace cutlass {
namespace softmax {
namespace kernel {

/////////////////////////////////////////////////////////////////////////////////////////////////

template<
    typename ThreadblockShape_,
    typename WarpCount_,
    typename ElementInput_,
    int AlignmentInput_,
    typename ElementOutput_,
    int AlignmentOutput_,
    typename ElementAccumulator_>
struct SoftmaxUniversal {
public:
    static_assert(AlignmentInput_ == AlignmentOutput_, "Input and output tensor should have the same alignemnt");

    using ThreadblockShape = ThreadblockShape_;
    using WarpCount = WarpCount_;
    using ElementInput = ElementInput_;
    using ElementOutput = ElementOutput_;
    using ElementAccumulator = ElementAccumulator_;
    static const int kElementsPerAccess = AlignmentInput_;

    //
    // Iterators
    //

    using BlockReduction = cutlass::softmax::threadblock::SoftmaxBlockReduction<
        ThreadblockShape, 
        WarpCount, 
        ElementInput, 
        kElementsPerAccess, 
        ElementAccumulator
    >;

    using WarpReduction = cutlass::softmax::threadblock::SoftmaxWarpReduction<
        ThreadblockShape, 
        WarpCount, 
        ElementInput, 
        kElementsPerAccess, 
        ElementAccumulator
    >;

    using Reduction = typename platform::conditional<WarpCount::kColumn == 1,
                                                     WarpReduction,
                                                     BlockReduction>::type;

    using Epilogue = cutlass::softmax::threadblock::DefaultEpilogueSoftmax<
        ElementAccumulator,
        ElementOutput,
        kElementsPerAccess,
        ThreadblockShape,
        WarpCount
    >;


    //
    // Structures
    //

    struct Arguments {
        //
        // Data members
        //
        typename Reduction::Arguments reduction_args;
        typename Epilogue::Arguments epilogue_args;
    };

    struct Params {
        //
        // Data members
        //
        typename Reduction::Params reduction_params;
        typename Epilogue::Params epilogue_params;

        //
        // Memebrs
        //
        CUTLASS_HOST_DEVICE
        Params(Arguments const &args):
            reduction_params(args.reduction_args),
            epilogue_params(args.epilogue_args)
        { }

    };

    union SharedStorage {
        typename Reduction::SharedStorage reduction_storage;
    };


public:

    /// Execute one softmax
    CUTLASS_DEVICE
    void operator()(Params const &params, SharedStorage &shared_storage) {

        int thread_idx = threadIdx.x;
        cutlass::MatrixCoord threadblock_offset{
            int(blockIdx.x), int(blockIdx.y)
        };

        Reduction softmax_reduction(params.reduction_params, shared_storage.reduction_storage, thread_idx, threadblock_offset);

        ElementAccumulator row_max;
        ElementAccumulator row_sum;

        softmax_reduction(row_max, row_sum);
        
        /// Epilogue
        Epilogue epilogue(params.epilogue_params, thread_idx, threadblock_offset);

        epilogue(row_max, row_sum);
    }
};

/////////////////////////////////////////////////////////////////////////////////////////////////

} // namespace kernel
} // namespace softmax
} // namespace cutlass

/////////////////////////////////////////////////////////////////////////////////////////////////