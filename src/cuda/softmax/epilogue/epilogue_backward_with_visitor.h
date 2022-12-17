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
class EpilogueBackwardWithVisitor {

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

    using InputFragment = Array<Element, InputTileIterator::Iterations::kColumn * kElementsPerAccess>;

    using Fragment = typename InputTileIterator::Fragment;

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
        MatrixCoord problem_size;
    };

    struct Params {
        //
        // Data members
        //
        Element* o_softmax;
        Element* grad_softmax;
        MatrixCoord problem_size;

        //
        // Members
        //
        CUTLASS_HOST_DEVICE
        Params(Arguments const &args):
            o_softmax(args.o_softmax),
            grad_softmax(args.grad_softmax),
            problem_size(args.problem_size)
        { }
    };

private:
    InputTileIterator o_softmax_iterator_;
    InputTileIterator grad_softmax_iterator_;

    Mult mult_op;
    Sub sub_op;

    int row_idx;
    int column_idx;

public:
    /// Constructor
    CUTLASS_DEVICE
    EpilogueBackwardWithVisitor(
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
        row_idx(threadblock_offset.row()),
        column_idx((InputTileIterator::ThreadMap::initial_offset(thread_idx) + threadblock_offset).column())
    { }

    CUTLASS_DEVICE
    void operator()(
        Visitor & visitor,
        ElementAccumulator const row_sum
     ) {
        typename InputTileIterator::Fragment o_softmax_frag;
        typename InputTileIterator::Fragment grad_softmax_frag;
        AccumulatorFragment compute_frag;

        NumericArrayConverter<ElementAccumulator, Element, kElementsPerAccess> input_converter;
        NumericArrayConverter<Element, ElementAccumulator, kElementsPerAccess> output_converter;

        visitor.begin_epilogue();

        for (int i = 0; i < InputTileIterator::Iterations::kColumn; i ++) {
            o_softmax_iterator_.load(o_softmax_frag);
            grad_softmax_iterator_.load(grad_softmax_frag);
            compute_frag = mult_op(input_converter(o_softmax_frag), sub_op(input_converter(grad_softmax_frag), row_sum));
            visitor.visit(
                row_idx,
                column_idx,
                compute_frag
            );
            column_idx += InputTileIterator::Shape::kColumn;
        }

        visitor.end_epilogue();
    }

    CUTLASS_DEVICE
    void operator()(
        Visitor & visitor,
        InputFragment& o_softmax_buffer,
        InputFragment& grad_softmax_buffer,
        ElementAccumulator const row_sum
    ){
        const typename InputTileIterator::Fragment* o_softmax_frag = reinterpret_cast<const typename InputTileIterator::Fragment*>(&o_softmax_buffer);
        const typename InputTileIterator::Fragment* grad_softmax_frag = reinterpret_cast<const typename InputTileIterator::Fragment*>(&grad_softmax_buffer);

        AccumulatorFragment compute_frag;
        NumericArrayConverter<ElementAccumulator, Element, kElementsPerAccess> input_converter;
        NumericArrayConverter<Element, ElementAccumulator, kElementsPerAccess> output_converter;

        visitor.begin_epilogue();

        for (int i = 0; i < InputTileIterator::Iterations::kColumn; i ++) {
            compute_frag = mult_op(input_converter(*o_softmax_frag), sub_op(input_converter(*grad_softmax_frag), row_sum));
            visitor.visit(
                row_idx,
                column_idx,
                i,
                compute_frag
            );
            column_idx += InputTileIterator::Shape::kColumn;
            o_softmax_frag ++;
            grad_softmax_frag ++;
        }

        visitor.end_epilogue();
    }
};

/////////////////////////////////////////////////////////////////////////////////////////////////

/// Helper to create an EpilogueWithVisitor from an existing epilogue
template <typename Visitor_, typename Existing_>
struct EpilogueBackwardWithVisitorFromExistingEpilogue {

    using Epilogue = EpilogueBackwardWithVisitor<
        Visitor_,
        typename Existing_::ElementAccumulator,
        typename Existing_::Element,
        Existing_::kElementsPerAccess,
        typename Existing_::ThreadblockShape,
        typename Existing_::WarpCount
    >;
};



template<
    typename Visitor_,
    typename ElementAccumulator_,
    typename ElementOutput_,
    int AlignmentOutput_,
    typename ThreadblockShape_,
    typename WarpCount_
>
class LayerNormEpilogueBackwardWithVisitor {

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

    using InputFragment = Array<Element, InputTileIterator::Iterations::kColumn * kElementsPerAccess>;

    using Fragment = typename InputTileIterator::Fragment;

    using AccumulatorFragment = Array<ElementAccumulator, kElementsPerAccess>;

    using Mult = cutlass::multiplies<AccumulatorFragment>;
    using Sub = cutlass::minus<AccumulatorFragment>;
    using Div = cutlass::divides<AccumulatorFragment>;
    using Add = cutlass::plus<AccumulatorFragment>;

    //
    // Structures
    //
    struct Arguments {
        //
        // Data members
        //
        MatrixCoord problem_size;
    };

    struct Params {
        //
        // Data members
        //
        MatrixCoord problem_size;

        //
        // Members
        //
        CUTLASS_HOST_DEVICE
        Params(Arguments const &args):
            problem_size(args.problem_size)
        { }
    };

private:
    Mult mult_op;
    Sub sub_op;
    Div div_op;
    Add add_op;
    ElementAccumulator numel_factor;

    int row_idx;
    int column_idx;

public:
    /// Constructor
    CUTLASS_DEVICE
    LayerNormEpilogueBackwardWithVisitor(
        Params const & params,
        int thread_idx,
        MatrixCoord threadblock_offset = MatrixCoord()
    ):
        row_idx(threadblock_offset.row()),
        column_idx((InputTileIterator::ThreadMap::initial_offset(thread_idx) + threadblock_offset).column()),
        numel_factor(ElementAccumulator(1) / ElementAccumulator(params.problem_size.column()))
    { }

    CUTLASS_DEVICE
    void operator()(
        Visitor & visitor,
        InputFragment& gamma_grad_y_buffer,
        InputFragment& x_hat_buffer,
        ElementAccumulator const row_sum_t1,
        ElementAccumulator const row_sum_t2
    ){
        const typename InputTileIterator::Fragment* gamma_grad_y_frag = reinterpret_cast<const typename InputTileIterator::Fragment*>(&gamma_grad_y_buffer);
        const typename InputTileIterator::Fragment* x_hat_frag = reinterpret_cast<const typename InputTileIterator::Fragment*>(&x_hat_buffer);

        AccumulatorFragment compute_frag;
        NumericArrayConverter<ElementAccumulator, Element, kElementsPerAccess> input_converter;
        NumericArrayConverter<Element, ElementAccumulator, kElementsPerAccess> output_converter;

        visitor.begin_epilogue();

        for (int i = 0; i < InputTileIterator::Iterations::kColumn; i ++) {
            compute_frag = sub_op(
                input_converter(*gamma_grad_y_frag), 
                mult_op(
                    add_op(
                        row_sum_t1,
                        mult_op(
                            input_converter(*x_hat_frag),
                            row_sum_t2
                        )
                    ),
                    numel_factor
                ));
            visitor.visit(
                row_idx,
                column_idx,
                i, 
                compute_frag
            );
            column_idx += InputTileIterator::Shape::kColumn;
            gamma_grad_y_frag ++;
            x_hat_frag ++;
        }

        visitor.end_epilogue();
    }
};

/////////////////////////////////////////////////////////////////////////////////////////////////

/// Helper to create an EpilogueWithVisitor from an existing epilogue
template <typename Visitor_, typename Existing_>
struct LayerNormEpilogueBackwardWithVisitorFromExistingEpilogue {

    using Epilogue = LayerNormEpilogueBackwardWithVisitor<
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