#pragma once

#include "cutlass/cutlass.h"
#include "spmm/epilogue/epilogue.h"

/////////////////////////////////////////////////////////////////////////////////////////////////

namespace cutlass {
namespace spmm {
namespace threadblock {

/////////////////////////////////////////////////////////////////////////////////////////////////

template<
    typename Visitor_,
    typename ElementAccumulator_,
    typename ElementOutput_,
    int ALignmentOutput_,
    typename Accumulator_,
    typename ThreadMap_
>
class EpilogueWithVisitor {
public:
    using Visitor = Visitor_;
    using Element = ElementOutput_;
    static const int kElementsPerAccess = ALignmentOutput_;
    using ThreadMap = ThreadMap_;
    using FragmentAcc = Accumulator_;

    //
    // Structures
    //

    struct Arguments {
        MatrixCoord problem_size;
    };

    struct Params {
        MatrixCoord problem_size;

        CUTLASS_HOST_DEVICE
        Params(Arguments const &args):
            problem_size(args.problem_size)
        { }
    };

private:
    int row_idx_;
    int col_idx_;

public:
    /// Constructor
    CUTLASS_DEVICE
    EpilogueWithVisitor(
        Params const & params,
        int thread_idx,
        int row_idx,
        int col_idx
    ):
        row_idx_(row_idx),
        col_idx_(col_idx) 
    {}

    CUTLASS_DEVICE
    void operator()(
        Visitor & visitor,
        FragmentAcc const &accum
    ) {
        // if (blockIdx.x == 0 && threadIdx.x == 0) {
        //     printf("hhhh222\n");
        // }
        visitor.begin_epilogue();
        visitor.visit(
            row_idx_,
            col_idx_,
            0,
            accum
        );
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
        typename Existing_::FragmentAcc,
        typename Existing_::ThreadMap
    >;
};
    
/////////////////////////////////////////////////////////////////////////////////////////////////

} // namespace threadblock
} // namespace spmm
} // namespace cutlass

/////////////////////////////////////////////////////////////////////////////////////////////////