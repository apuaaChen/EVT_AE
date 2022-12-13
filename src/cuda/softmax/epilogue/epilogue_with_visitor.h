#pragma once

#include "cutlass/cutlass.h"
#include "softmax/epilogue/epilogue.h"


namespace cutlass {

CUTLASS_DEVICE
float sqrt(float x) {
    return sqrtf(x);
}

}


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

    using InputFragment = Array<ElementAccumulator, InputTileIterator::Iterations::kColumn * kElementsPerAccess>;

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
        InputFragment& input_buffer,
        ElementAccumulator const row_max,
        ElementAccumulator const row_sum
    ) {

        const AccumulatorFragment* input_frag = reinterpret_cast<const AccumulatorFragment*>(&input_buffer);
        
        AccumulatorFragment compute_frag;
        NumericArrayConverter<ElementAccumulator, Element, kElementsPerAccess> input_converter;
        NumericArrayConverter<Element, ElementAccumulator, kElementsPerAccess> output_converter;
        
        visitor.begin_epilogue();

        #pragma unroll
        for (int i = 0; i < InputTileIterator::Iterations::kColumn; i ++) {
            compute_frag = div_op(*input_frag, row_sum);
            visitor.visit(
                row_idx,
                column_idx,
                compute_frag);
            
            column_idx += InputTileIterator::Shape::kColumn;
            input_frag ++;
        }

        visitor.end_epilogue();
    }

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


template<
    typename Visitor_,
    typename ElementAccumulator_,
    typename ElementOutput_,
    int AlignmentOutput_,
    typename ThreadblockShape_,
    typename WarpCount_
>
class EpilogueWithVisitorLayerNorm {

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

    using InputFragment = Array<ElementAccumulator, InputTileIterator::Iterations::kColumn * kElementsPerAccess>;

    using Fragment = typename InputTileIterator::Fragment;

    using AccumulatorFragment = Array<ElementAccumulator, kElementsPerAccess>;

    using Minus = cutlass::minus<AccumulatorFragment>;
    using Mult = cutlass::multiplies<AccumulatorFragment>;

    //
    // Structures
    //

    struct Arguments {
        //
        // Data members
        //
        Element* input;
        float eps;
        MatrixCoord problem_size;
        ElementAccumulator* mean;
        ElementAccumulator* std_factor;
    };

    struct Params {
        //
        // Data members
        //
        Element* input;
        float eps;
        MatrixCoord problem_size;
        ElementAccumulator* mean;
        ElementAccumulator* std_factor;

        //
        // Members
        //
        CUTLASS_HOST_DEVICE
        Params(Arguments const &args):
            input(args.input),
            eps(args.eps),
            problem_size(args.problem_size),
            mean(args.mean),
            std_factor(args.std_factor)
        { }
    };

private:
    InputTileIterator input_iterator_;
    ElementAccumulator* mean_ptr;
    ElementAccumulator* std_factor_ptr;

    Minus minus_op;
    Mult mult_op;
    float eps;

    int row_idx;
    int column_idx;
    int thread_idx_;

public:
    /// Constructor
    CUTLASS_DEVICE
    EpilogueWithVisitorLayerNorm(
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
        eps(params.eps),
        column_idx((InputTileIterator::ThreadMap::initial_offset(thread_idx) + threadblock_offset).column()),
        thread_idx_(thread_idx),
        mean_ptr(params.mean), std_factor_ptr(params.std_factor)
    { }

    CUTLASS_DEVICE
    void operator()(
        Visitor & visitor,
        InputFragment& input_buffer,
        ElementAccumulator const row_m1,
        ElementAccumulator const row_m2
    ) {
        cutlass::divides<ElementAccumulator> scalar_divide;
        ElementAccumulator row_std = scalar_divide(ElementAccumulator(1), cutlass::sqrt(row_m2 - row_m1 * row_m1 + eps));

        // write mean and std_factor
        if (thread_idx_ == 0){
            *(mean_ptr + row_idx) = row_m1;
            *(std_factor_ptr + row_idx) = row_std;
        }

        const AccumulatorFragment* input_frag = reinterpret_cast<const AccumulatorFragment*>(&input_buffer);
        
        AccumulatorFragment compute_frag;
        NumericArrayConverter<ElementAccumulator, Element, kElementsPerAccess> input_converter;
        NumericArrayConverter<Element, ElementAccumulator, kElementsPerAccess> output_converter;
        
        visitor.begin_epilogue();

        #pragma unroll
        for (int i = 0; i < InputTileIterator::Iterations::kColumn; i ++) {
            compute_frag = mult_op(minus_op(*input_frag, row_m1), row_std);
            visitor.visit(
                row_idx,
                column_idx,
                compute_frag);
            
            column_idx += InputTileIterator::Shape::kColumn;
            input_frag ++;
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


/// Helper to create an EpilogueWithVisitor from an existing epilogue
template <typename Visitor_, typename Existing_>
struct LayerNormEpilogueWithVisitorFromExistingEpilogue {

    using Epilogue = EpilogueWithVisitorLayerNorm<
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