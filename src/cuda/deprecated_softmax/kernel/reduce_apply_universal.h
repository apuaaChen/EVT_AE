#pragma once
#include "cutlass/cutlass.h"

////////////////////////////////////////////////////////////////////////////////

namespace cutlass {
namespace reduce_apply {
namespace kernel {

////////////////////////////////////////////////////////////////////////////////

template<
    typename Reduction_,
    typename Epilogue_>
struct ReductionApplywithEpilogueVisitor {
public:
    using Reduction = Reduction_;
    using Epilogue = Epilogue_;
    using EpilogueVisitor = typename Epilogue::Visitor;

    using ElementAccumulator = typename Reduction::ElementAccumulator;

    //
    // Structures
    //

    struct Arguments {
        //
        // Data members
        //
        typename Reduction::Arguments reduction_args;
        typename Epilogue::Arguments epilogue_args;
        typename EpilogueVisitor::Arguments epilogue_visitor;
    };

    struct Params {
        //
        // Data members
        //
        typename Reduction::Params reduction_params;
        typename Epilogue::Params epilogue_params;
        typename EpilogueVisitor::Params epilogue_visitor;

        /// Constructs an arguments structure
        Params(
            Arguments const &args
        ):
            reduction_params(args.reduction_args),
            epilogue_params(args.epilogue_args),
            epilogue_visitor(args.epilogue_visitor){ }

    };

    union SharedStorage {
        typename Reduction::SharedStorage reduction_storage;
        typename EpilogueVisitor::SharedStorage visitor;
    };
};



////////////////////////////////////////////////////////////////////////////////

} // namespace kernel
} // namespace softmax
} // namespace cutlass

////////////////////////////////////////////////////////////////////////////////