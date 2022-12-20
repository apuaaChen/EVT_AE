#pragma once
#include "softmax/threadblock/row_tile_iterator.h"
#include <cub/block/block_reduce.cuh>
#include <cub/warp/warp_reduce.cuh>


////////////////////////////////////////////////////////////////////////////////

namespace cutlass {

template <typename T, int N>
CUTLASS_DEVICE
Array<T, N> operator+(Array<T, N> const &a, T const &b) {
    cutlass::plus<Array<T, N>> plus_op_;
    return plus_op_(a, b);
}

template <typename T, int N>
CUTLASS_DEVICE
Array<T, N> operator-(Array<T, N> const &a, T const &b) {
    cutlass::minus<Array<T, N>> minus_op_;
    return minus_op_(a, b);
}

template <typename T, int N>
CUTLASS_DEVICE
Array<T, N> exp(Array<T, N> const &a) {
    cutlass::fast_exp_op<Array<T, N>> exp_op_;
    return exp_op_(a);
}

template <typename T, int N>
CUTLASS_DEVICE
Array<T, N> max(Array<T, N> const &a, Array<T, N> const &b) {
    cutlass::maximum<Array<T, N>> maximum_op_;
    return maximum_op_(a, b);
}

namespace reduce_apply {
namespace threadblock {

////////////////////////////////////////////////////////////////////////////////

template<
    typename ThreadblockShape_,
    typename WarpCount_,
    typename ElementInput_,
    int AlignmentInput_,
    typename ElementAccumulator_>
struct BlockReductionBase {
public:
    using ThreadblockShape = ThreadblockShape_;
    using WarpCount = WarpCount_;
    using ElementInput = ElementInput_;
    using ElementAccumulator = ElementAccumulator_;
    static const int kElementsPerAccess = AlignmentInput_;

    //
    // Iterator
    //
    using ThreadMapInput = cutlass::softmax::threadblock::RowThreadMap<
        ThreadblockShape, WarpCount, ElementInput, AlignmentInput_>;
    using InputIterator = cutlass::softmax::threadblock::RowTileIterator<
        ThreadMapInput, ElementInput>;
    static const int kNumThreads = WarpCount::kCount * 32;

    //
    // Fragments
    //
    using InputFragment = Array<ElementInput, kElementsPerAccess>;
    using AccumulatorFragment = Array<ElementAccumulator, kElementsPerAccess>;
    // Used by cub library
    typedef ElementAccumulator ArrayAcc[kElementsPerAccess];

    //
    // Reduction
    //
    using BlockReduce = cub::BlockReduce<ElementAccumulator, kNumThreads>;


    union SharedStorage {
        typename BlockReduce::TempStorage temp_storage;
        ElementAccumulator broadcast_buffer;
    };

private:

    /// Reduction related members
    BlockReduce block_reduce_;
    ElementAccumulator* broadcast_buffer_ptr_;
    int thread_idx_;

    /// Converters
    NumericArrayConverter<
        ElementAccumulator, ElementInput, kElementsPerAccess> input2acc_;
    NumericArrayConverter<
        ElementInput, ElementAccumulator, kElementsPerAccess> acc2input_;    

public:
    //
    // Constructor
    //
    CUTLASS_DEVICE
    BlockReductionBase (
        int thread_idx,
        SharedStorage & shared_storage
    ):
        block_reduce_(shared_storage.temp_storage),
        broadcast_buffer_ptr_(&shared_storage.broadcast_buffer),
        thread_idx_(thread_idx)
    { }

    //
    // Member functions
    //

    /// Sum reduction
    CUTLASS_DEVICE
    void sum(AccumulatorFragment & partial_sum, ElementAccumulator &row_sum) {
        row_sum = block_reduce_.Sum(
            *(reinterpret_cast<ArrayAcc*>(partial_sum.raw_data()))
        );

        if (thread_idx_ == 0) *(broadcast_buffer_ptr_) = row_sum;
        __syncthreads();
        row_sum = *(broadcast_buffer_ptr_);
    }

    /// Max reduction
    CUTLASS_DEVICE
    void max(AccumulatorFragment & partial_max, ElementAccumulator & row_max) {
        row_max = block_reduce_.Reduce(
            *(reinterpret_cast<ArrayAcc*>(partial_max.raw_data())), cub::Max());

        if (thread_idx_ == 0) *(broadcast_buffer_ptr_) = row_max;
        __syncthreads();
        row_max = *(broadcast_buffer_ptr_);
    }

    /// Data converter
    CUTLASS_DEVICE
    AccumulatorFragment input2acc(InputFragment &input) {
        return input2acc_(input);
    }

    CUTLASS_DEVICE
    InputFragment acc2input(AccumulatorFragment &input) {
        return acc2input_(input);
    }

};



template<
    typename ThreadblockShape_,
    typename WarpCount_,
    typename ElementInput_,
    int AlignmentInput_,
    typename ElementAccumulator_>
struct WarpReductionBase {
public:
    using ThreadblockShape = ThreadblockShape_;
    using WarpCount = WarpCount_;
    using ElementInput = ElementInput_;
    using ElementAccumulator = ElementAccumulator_;
    static const int kElementsPerAccess = AlignmentInput_;

    //
    // Iterator
    //
    using ThreadMapInput = cutlass::softmax::threadblock::RowThreadMap<
        ThreadblockShape, WarpCount, ElementInput, AlignmentInput_>;
    using InputIterator = cutlass::softmax::threadblock::RowTileIterator<
        ThreadMapInput, ElementInput>;
    static const int kNumThreads = WarpCount::kCount * 32;

    //
    // Fragments
    //
    using InputFragment = Array<ElementInput, kElementsPerAccess>;
    using AccumulatorFragment = Array<ElementAccumulator, kElementsPerAccess>;
    // Used by cub library
    typedef ElementAccumulator ArrayAcc[kElementsPerAccess];

    //
    // Reduction
    //
    using WarpReduce = cub::WarpReduce<ElementAccumulator>;


    union SharedStorage {
        typename WarpReduce::TempStorage temp_storage;
    };

private:

    /// Reduction related members
    WarpReduce warp_reduce_;
    int thread_idx_;

    /// Converters
    NumericArrayConverter<
        ElementAccumulator, ElementInput, kElementsPerAccess> input2acc_;
    NumericArrayConverter<
        ElementInput, ElementAccumulator, kElementsPerAccess> acc2input_;    

public:
    //
    // Constructor
    //
    CUTLASS_DEVICE
    WarpReductionBase (
        int thread_idx,
        SharedStorage & shared_storage
    ):
        warp_reduce_(shared_storage.temp_storage),
        thread_idx_(thread_idx)
    { }

    //
    // Member functions
    //

    /// Sum reduction
    CUTLASS_DEVICE
    void sum(AccumulatorFragment & partial_sum, ElementAccumulator &row_sum) {
        /// Get the sum in the array of each thread
        ElementAccumulator row_sum_local = ElementAccumulator(0);
        #pragma unroll
        for (int i = 0; i < kElementsPerAccess; i ++) {
            row_sum_local += *(partial_sum.data() + i);
        }
        row_sum = warp_reduce_.Sum(row_sum_local);

        row_sum = __shfl_sync(0xffffffff, row_sum, 0);
    }

    /// Max reduction
    CUTLASS_DEVICE
    void max(AccumulatorFragment & partial_max, ElementAccumulator & row_max) {
        ElementAccumulator row_max_local = ElementAccumulator(-1e+6);
        #pragma unroll
        for (int i = 0; i < kElementsPerAccess; i ++) {
            row_max_local = *(partial_max.data() + i) < row_max_local ? row_max_local : *(partial_max.data() + i);
        }

        row_max = warp_reduce_.Reduce(row_max_local, cub::Max());

        row_max = __shfl_sync(0xffffffff, row_max, 0);
    }

    /// Data converter
    CUTLASS_DEVICE
    AccumulatorFragment input2acc(InputFragment &input) {
        return input2acc_(input);
    }

    CUTLASS_DEVICE
    InputFragment acc2input(AccumulatorFragment &input) {
        return acc2input_(input);
    }

};


////////////////////////////////////////////////////////////////////////////////

} // namespace threadblock
} // namespace reduce_apply
} // namespace cutlass

////////////////////////////////////////////////////////////////////////////////