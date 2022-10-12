#pragma once

#include "cutlass/cutlass.h"

/////////////////////////////////////////////////////////////////////////////////////////////////

namespace cutlass {
namespace softmax {
namespace kernel {

/////////////////////////////////////////////////////////////////////////////////////////////////

template<
    typename ElementInput_,
    typename ElementOutput_,
    typename ElementAccumulator_,
    typename Epilogue_>
struct SoftmaxUniversalwithEpilogueVisitor {

    using ElementInput = ElementInput_;
    using ElementOutput = ElementOutput_;
    using ElementAccumulator = ElementAccumulator_;

    static const int kALIGN_BYTES = 16;
    static const int kILP = kALIGN_BYTES / sizeof(ElementInput);

    using Epilogue = Epilogue_;
    using EpilogueVisitor = typename Epilogue::Visitor;


    //
    // Structures
    //

    struct Arguments {
        //
        // Data members
        //
        ElementInput* input;
        MatrixCoord problem_size;

        typename EpilogueVisitor::Arguments epilogue_visitor;
    };

    struct Params {
        //
        // Data members
        //
        ElementInput* input;
        MatrixCoord problem_size;
        
        typename EpilogueVisitor::Params epilogue_visitor;


    };


public:

    /// Execute one softmax
    CUTLASS_DEVICE
    void operator()(Params const &params, SharedStorage &shared_storage) {
        MatrixCoord threadblock_tile_offset = MatrixCoord(blockIdx.x, blockIdx.y);

        ElementInput* input = params.input + threadblock_tile_offset.row() * params.problem_size.column();

        const int shift = ((uint64_t)input) % kALIGN_BYTES / sizeof(ElementInput);

        using LoadT = at::native::memory::aligned_vector<ElementInput, kILP>;
        using StoreT = at::native::memory::aligned_vector<ElementOutput, kILP>;

        // find the max
        ElementAccumulator thread_max = ilpReduction<MaxFloat, kILP, ElementInput, ElementAccumulator>(
            shift, input, params.problem_size.column(), MaxFloat<ElementInput, ElementAccumulator>(), -at::numeric_limits<ElementAccumulator>::max()
        );

        ElementAccumulator max_k = blockReduce<Max, ElementAccumulator>(
            shared_storage.main_loop, thread_max, Max<ElementAccumulator>(), -at::numeric_limits<ElementAccumulator>::max()
        );

        // reduce all values
        ElementAccumulator thread_exp = ilpReduce<SumExpFloat, kILP, ElementInput, ElementAccumulator>(
            shift, input, params.problem_size.column(), SumExpFloat<ElementInput, ElementAccumulator>(max_k), static_cast<ElementAccumulator>(0)
        );

        ElementAccumulator sum_all = blockReduce<Add, ElementAccumulator>(
            shared_storage.main_loop, thread_exp, Add<ElementAccumulator>(), static_cast<ElementAccumulator>(0)
        );

        Epilogue<ElementInput, ElementAccumulator, ElementOutput> epilogue(max_k, sum_all);

        if (shift == output_shift) {
            WriteFpropResultsVectorized<kILP, ElementInput, ElementAccumulator, ElementOutput, Epilogue>(params.epilogue_visitor);
        } else {
            WriteFpropResults<kILP, ElementAccumulator, ElementOutput, Epilogue>(params.epilogue_visitor);
        }
    }
};

/////////////////////////////////////////////////////////////////////////////////////////////////

} // namespace kernel
} // namespace softmax
} // namespace cutlass

/////////////////////////////////////////////////////////////////////////////////////////////////