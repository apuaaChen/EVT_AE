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
class DefaultEpilogueSoftmax {

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

    using Minus = cutlass::minus<AccumulatorFragment>;
    using Div = cutlass::divides<AccumulatorFragment>;

    //
    // Structures
    //

    struct Arguments {
        //
        // Data members
        //
        Element* input;
        Element* output;
        MatrixCoord problem_size;
    };

    struct Params {
        //
        // Data members
        //
        Element* input;
        Element* output;
        MatrixCoord problem_size;

        //
        // Members
        //
        CUTLASS_HOST_DEVICE
        Params(Arguments const &args):
            input(args.input),
            output(args.output),
            problem_size(args.problem_size)
        { }
    };

private:

    OutputTileIterator iterator_;
    InputTileIterator input_iterator_;

    Minus minus_op;
    Div div_op;

public:
    /// Constructor
    CUTLASS_DEVICE
    DefaultEpilogueSoftmax(
        Params const & params,
        int thread_idx,
        MatrixCoord threadblock_offset = MatrixCoord()
    ):
        iterator_(
            typename OutputTileIterator::Param(params.problem_size.column()),
            params.output,
            params.problem_size,
            thread_idx,
            threadblock_offset
        ),
        input_iterator_(
            typename InputTileIterator::Param(params.problem_size.column()),
            params.input,
            params.problem_size,
            thread_idx,
            threadblock_offset
        )
    { }

    CUTLASS_DEVICE
    void operator()(
        ElementAccumulator row_max,
        ElementAccumulator row_sum
    ) {

        typename InputTileIterator::Fragment input_frag;
        AccumulatorFragment compute_frag;
        NumericArrayConverter<ElementAccumulator, Element, kElementsPerAccess> input_converter;
        NumericArrayConverter<Element, ElementAccumulator, kElementsPerAccess> output_converter;
        

        for (int i = 0; i < InputTileIterator::Iterations::kColumn; i ++) {
            input_iterator_.load(input_frag);
            compute_frag = input_converter(input_frag);
            compute_frag = minus_op(compute_frag, row_max);
            compute_frag = exp_op(compute_frag);
            compute_frag = div_op(compute_frag, row_sum);
            iterator_.store(output_converter(compute_frag));
        }
    }

};



template<
    typename ElementAccumulator_,
    typename ElementOutput_,
    int AlignmentOutput_,
    typename ThreadblockShape_,
    typename WarpCount_
>
class DefaultEpilogueLayerNorm {

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

    using Minus = cutlass::minus<AccumulatorFragment>;
    using Exp = cutlass::fast_exp_op<AccumulatorFragment>;
    using Div = cutlass::divides<AccumulatorFragment>;

    //
    // Structures
    //

    struct Arguments {
        //
        // Data members
        //
        Element* input;
        Element* output;
        float eps;
        MatrixCoord problem_size;
    };

    struct Params {
        //
        // Data members
        //
        Element* input;
        Element* output;
        float eps;
        MatrixCoord problem_size;

        //
        // Members
        //
        CUTLASS_HOST_DEVICE
        Params(Arguments const &args):
            input(args.input),
            output(args.output),
            eps(args.eps),
            problem_size(args.problem_size)
        { }
    };

private:

    OutputTileIterator iterator_;
    InputTileIterator input_iterator_;
    float eps;
    Minus minus_op;
    Div div_op;

public:
    /// Constructor
    CUTLASS_DEVICE
    DefaultEpilogueLayerNorm(
        Params const & params,
        int thread_idx,
        MatrixCoord threadblock_offset = MatrixCoord()
    ):
        iterator_(
            typename OutputTileIterator::Param(params.problem_size.column()),
            params.output,
            params.problem_size,
            thread_idx,
            threadblock_offset
        ),
        input_iterator_(
            typename InputTileIterator::Param(params.problem_size.column()),
            params.input,
            params.problem_size,
            thread_idx,
            threadblock_offset
        ), eps(params.eps)
    { }

    CUTLASS_DEVICE
    void operator()(
        ElementAccumulator row_m1,
        ElementAccumulator row_m2
    ) {
        ElementAccumulator row_std = cutlass::sqrt(row_m2 - row_m1 * row_m1 + eps);

        typename InputTileIterator::Fragment input_frag;
        AccumulatorFragment compute_frag;
        NumericArrayConverter<ElementAccumulator, Element, kElementsPerAccess> input_converter;
        NumericArrayConverter<Element, ElementAccumulator, kElementsPerAccess> output_converter;
        

        for (int i = 0; i < InputTileIterator::Iterations::kColumn; i ++) {
            input_iterator_.load(input_frag);
            compute_frag = input_converter(input_frag);
            compute_frag = minus_op(compute_frag, row_m1);
            compute_frag = div_op(compute_frag, row_std);
            iterator_.store(output_converter(compute_frag));
        }
    }

};



/////////////////////////////////////////////////////////////////////////////////////////////////

} // namespace threadblock
} // namespace softmax
} // namespace cutlass

/////////////////////////////////////////////////////////////////////////////////////////////////