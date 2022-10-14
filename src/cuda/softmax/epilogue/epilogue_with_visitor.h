#pragma once

#include "cutlass/cutlass.h"
#include "softmax/epilogue/epilogue.h"


/////////////////////////////////////////////////////////////////////////////////////////////////

namespace cutlass {
namespace softmax {
namespace threadblock {

/////////////////////////////////////////////////////////////////////////////////////////////////


template<
    typename Visitor_,
    typename ElementAccumulator_,
    typename ElementOutput_,
    int AlignmentOutput_,
    typename ThreadblockShape_,
    typename WarpCount_
>
class EpilogueWithVisitor {

public:
    using Visitor = Visitor_;

    using Element = ElementOutput_;
    static const int kElementsPerAccess = AlignmentOutput_;
    using ElementAccumulator = ElementAccumulator_;
    using ThreadblockShape = ThreadblockShape_;
    using WarpCount = WarpCount_;

    using ThreadMap = cutlass::softmax::threadblock::RowThreadMap<
        ThreadblockShape_, WarpCount_, Element, kElementsPerAccess>;
    
    using InputTileIterator = cutlass::softmax::threadblock::RowTileIterator<ThreadMap, Element>;

    using Fragment = typename InputTileIterator::Fragment;

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
        MatrixCoord problem_size;
    };

    struct Params {
        //
        // Data members
        //
        Element* input;
        MatrixCoord problem_size;

        //
        // Members
        //
        CUTLASS_HOST_DEVICE
        Params(Arguments const &args):
            input(args.input),
            problem_size(args.problem_size)
        { }
    };

private:
    InputTileIterator input_iterator_;

    Minus minus_op;
    Exp exp_op;
    Div div_op;

    int row_idx;
    int column_idx;

public:
    /// Constructor
    CUTLASS_DEVICE
    EpilogueWithVisitor(
        Params const & params,
        int thread_idx,
        MatrixCoord threadblock_offset = MatrixCoord()
    ):
        input_iterator_(
            typename InputTileIterator::Params(params.problem_size.column()),
            params.input,
            params.problem_size,
            thread_idx,
            threadblock_offset
        ),
        row_idx(threadblock_offset.row()),
        column_idx((InputTileIterator::ThreadMap::initial_offset(thread_idx) + threadblock_offset).column())
    { }

    CUTLASS_DEVICE
    void operator()(
        Visitor & visitor,
        ElementAccumulator const row_max,
        ElementAccumulator const row_sum
    ) {

        typename InputTileIterator::Fragment input_frag;
        AccumulatorFragment compute_frag;
        NumericArrayConverter<ElementAccumulator, Element, kElementsPerAccess> input_converter;
        NumericArrayConverter<Element, ElementAccumulator, kElementsPerAccess> output_converter;
        
        visitor.begin_epilogue();

        for (int i = 0; i < InputTileIterator::Iterations::kColumn; i ++) {
            input_iterator_.load(input_frag);
            compute_frag = input_converter(input_frag);
            compute_frag = minus_op(compute_frag, row_max);
            compute_frag = exp_op(compute_frag);
            compute_frag = div_op(compute_frag, row_sum);
            visitor.visit(
                row_idx,
                column_idx,
                compute_frag);
            
            column_idx += InputTileIterator::Shape::kColumn;
        }

        visitor.end_epilogue();
    }

};

/////////////////////////////////////////////////////////////////////////////////////////////////

/// Helper to create an EpilogueWithVisitor from an existing epilogue
template <typename Visitor_, typename Existing_>
struct EpilogueWithVisitorFromExistingEpilogue {

    using Epilogue = EpilogueWithVisitor<
        Visitor_,
        typename Existing_::ElementAccumulator,
        typename Existing_::Element,
        Existing_::kElementsPerAccess,
        typename Existing_::ThreadblockShape,
        typename Existing_::WarpCount
    >;
};

/////////////////////////////////////////////////////////////////////////////////////////////////

} // namespace threadblock
} // namespace softmax
} // namespace cutlass

/////////////////////////////////////////////////////////////////////////////////////////////////