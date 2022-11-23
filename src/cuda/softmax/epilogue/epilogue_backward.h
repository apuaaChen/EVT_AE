#pragma once

#include "cutlass/cutlass.h"
#include "softmax/threadblock/row_tile_iterator.h"


/////////////////////////////////////////////////////////////////////////////////////////////////

namespace cutlass {
namespace softmax {
namespace threadblock {

/////////////////////////////////////////////////////////////////////////////////////////////////

template<
    typename ElementAccumulator_,
    typename ElementOutput_,
    int AlignmentOutput_,
    typename ThreadblockShape_,
    typename WarpCount_
>
class DefaultEpilogueSoftmaxBackward {

public:
    using Element = ElementOutput_;
    static const int kElementsPerAccess = AlignmentOutput_;
    using ElementAccumulator = ElementAccumulator_;
    using ThreadblockShape = ThreadblockShape_;
    using WarpCount = WarpCount_;

    using ThreadMap = cutlass::softmax::threadblock::RowThreadMap<
        ThreadblockShape_, WarpCount_, Element, kElementsPerAccess>;
    
    using InputTileIterator = cutlass::softmax::threadblock::RowTileIterator<ThreadMap, Element>;
    using OutputTileIterator = cutlass::softmax::threadblock::RowTileIterator<ThreadMap, Element>;

    using Fragment = typename OutputTileIterator::Fragment;

    using AccumulatorFragment = Array<ElementAccumulator, kElementsPerAccess>;

    using Mult = cutlass::multiplies<AccumulatorFragment>;
    using Sub = cutlass::minus<AccumulatorFragment>;

    //
    // Structures
    //
    struct Arguments {
        //
        // Data members
        //
        Element* o_softmax;
        Element* grad_softmax;
        Element* output;
        MatrixCoord problem_size;
    };

    struct Params {
        //
        // Data members
        //
        Element* o_softmax;
        Element* grad_softmax;
        Element* output;
        MatrixCoord problem_size;

        //
        // Members
        //
        CUTLASS_HOST_DEVICE
        Params(Arguments const &args):
            o_softmax(args.o_softmax),
            grad_softmax(args.grad_softmax),
            output(args.output),
            problem_size(args.problem_size)
        { }
    };

private:

    InputTileIterator o_softmax_iterator_;
    InputTileIterator grad_softmax_iterator_;

    OutputTileIterator iterator_;

    Mult mult_op;
    Sub sub_op;

    int row_idx;
    int column_idx;

public:
    /// Constructor
    CUTLASS_DEVICE
    DefaultEpilogueSoftmaxBackward(
        Params const & params,
        int thread_idx,
        MatrixCoord threadblock_offset = MatrixCoord()
    ):
        o_softmax_iterator_(
            typename InputTileIterator::Params(params.problem_size.column()),
            params.o_softmax,
            params.problem_size,
            thread_idx,
            threadblock_offset
        ),
        grad_softmax_iterator_(
            typename InputTileIterator::Params(params.problem_size.column()),
            params.grad_softmax,
            params.problem_size,
            thread_idx,
            threadblock_offset
        ),
        iterator_(
            typename OutputTileIterator::Param(params.problem_size.column()),
            params.output,
            params.problem_size,
            thread_idx,
            threadblock_offset
        ),
        row_idx(threadblock_offset.row()),
        column_idx((InputTileIterator::ThreadMap::initial_offset(thread_idx) + threadblock_offset).column())
    { }

    CUTLASS_DEVICE
    void operator()(
        ElementAccumulator row_sum
    ) {
        typename InputTileIterator::Fragment o_softmax_frag;
        typename InputTileIterator::Fragment grad_softmax_frag;
        AccumulatorFragment compute_frag;

        NumericArrayConverter<ElementAccumulator, Element, kElementsPerAccess> input_converter;
        NumericArrayConverter<Element, ElementAccumulator, kElementsPerAccess> output_converter;

        for (int i = 0; i < InputTileIterator::Iterations::kColumn; i ++) {
            o_softmax_iterator_.load(o_softmax_frag);
            grad_softmax_iterator_.load(grad_softmax_frag);
            compute_frag = mult_op(input_converter(o_softmax_frag), sub_op(input_converter(grad_softmax_frag), row_sum));
            iterator_.store(output_converter(compute_frag));
        }
    }

};



/////////////////////////////////////////////////////////////////////////////////////////////////

} // namespace threadblock
} // namespace softmax
} // namespace cutlass

/////////////////////////////////////////////////////////////////////////////////////////////////