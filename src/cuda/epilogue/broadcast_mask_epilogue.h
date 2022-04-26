#include "cutlass/arch/memory_sm75.h"
#include "helper.h"
namespace cutlass {
namespace epilogue {
namespace threadblock {


/// Define the iterator that reads mask from global memory
template <
    typename Element,
    typename ThreadblockTile_Shape_
>
class MaskTileIterator{
public:
    using ThreadblockTile_Shape = ThreadblockTile_Shape_;
    static const int kElementsPerAccess = 2;
    static const int kNumActiveThread = ThreadblockTile_Shape::kN / kElementsPerAccess;

    using TensorCoord = MatrixCoord;

    float* global_ptr;
    int stride;

    CUTLASS_DEVICE
    MaskTileIterator(
        Element* mask_pointer,
        TensorCoord extent,
        int thread_id,
        int batch_id,
        TensorCoord threadblock_offset
    ):
    stride(extent.column())
    {   
        if (thread_id < kNumActiveThread){
            int mask_id = batch_id / extent.row();
            int col_id = (thread_id - thread_id % 8) + (thread_id % 4) * 2 + (thread_id % 8) / 4;
            // TODO: this should be interleaved
            global_ptr = reinterpret_cast<float*>(mask_pointer + mask_id * stride + threadblock_offset.column()) + col_id;
        }
    }

    CUTLASS_DEVICE
    float *get(){
        return global_ptr;
    }
};

/// Define the epilogue that broadcast the 1D mask to fragment
template <
    typename ThreadblockTile_Shape_,
    typename WarpTile_Shape_,
    typename OutputOp_,
    typename Mma_,
    typename MaskTileIterator
>
class BroadcastMaskPrologue{
public:
    using ThreadblockTile_Shape = ThreadblockTile_Shape_;
    using WarpTile_Shape = WarpTile_Shape_;
    using OutputOp = OutputOp_;
    using Mma = Mma_;
    using MaskIterator = MaskTileIterator;

    // number of warps along rows and columns
    using WarpCount = MatrixShape<ThreadblockTile_Shape::kM / WarpTile_Shape::kM,
                                  ThreadblockTile_Shape::kN / WarpTile_Shape::kN>;
    
    // The shape of each fragment
    using FragmentShape = MatrixShape<8, 8>;

    using FragmentCount = MatrixShape<WarpTile_Shape::kM / FragmentShape::kRow,
                                      WarpTile_Shape::kN / FragmentShape::kColumn>;
    
    using ElementOutput = typename OutputOp::ElementOutput;

    using LdsmFootprint = MatrixShape<FragmentShape::kRow, 4 * FragmentShape::kColumn>;

    using ShmemLoadIterations = MatrixShape<1, WarpTile_Shape::kN / LdsmFootprint::kColumn>;


    struct SharedStorage {
        /// Element type of shared memory
        using Element = ElementOutput;

        /// Layout of shared memory allocation
        using Layout = layout::RowMajor;

        /// Tensor reference to shared memory allocation
        using TensorRef = TensorRef<Element, Layout>;

        /// Logical shape of the shared memory tile written to by all warps
        using Shape = MatrixShape<1, ThreadblockTile_Shape::kN>;

        using StorageShape = Shape;

        //
        // Data members
        //

        AlignedBuffer<Element, StorageShape::kCount> storage;

        //
        // Methods
        //

        /// Returns a pointer to the shared memory buffer
        CUTLASS_DEVICE
        Element *data() {
            return storage.data();
        }

        /// Returns a tensor reference to the shared memory buffer
        CUTLASS_DEVICE
        TensorRef reference() {
        return TensorRef(
            storage.data(), 
            Layout::packed({StorageShape::kRow, StorageShape::kColumn}));
        }
    };

    static const int kElementsPerAccess = 2;

    static const int kNumActiveThread = SharedStorage::Shape::kColumn / kElementsPerAccess;

    /// The pointer that loads data from global memory to shared memory
    float* shared_store_ptr;

    /// The pointer that load data from shared memory to global memory
    float4* shared_load_ptr;

    /// Constructor
    CUTLASS_DEVICE
    BroadcastMaskPrologue(
        SharedStorage &shared_storage,
        int thread_id,
        int warp_id,
        int lane_id,
        MaskTileIterator iterator_M
    ){
        if (thread_id < kNumActiveThread){
            shared_store_ptr = reinterpret_cast<float*>(shared_storage.data()) + thread_id;
            *(shared_store_ptr) = *(iterator_M.get());
        }

        // get the load pointer
        int warp_col_id = warp_id / WarpCount::kRow;
        int group_id = lane_id / 8;

        shared_load_ptr = reinterpret_cast<float4*>(shared_storage.data() + warp_col_id * WarpTile_Shape::kN + group_id * FragmentShape::kColumn);

        __syncthreads();
    }

    /// Fill the fragments
    CUTLASS_DEVICE
    void fill(
        typename Mma::FragmentC &accumulators
    ){
        ElementOutput tmp[ShmemLoadIterations::kColumn * 8];
        int* tmp_int = reinterpret_cast<int*>(tmp);
        #pragma unroll
        for (int i = 0; i < ShmemLoadIterations::kColumn; i++){
            unsigned shared_ptr_t = cutlass::arch::cutlass_get_smem_pointer(shared_load_ptr);
            asm volatile ("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0, %1, %2, %3}, [%4];"
                 : "=r"(tmp_int[i * 4]), "=r"(tmp_int[1 + i * 4]), "=r"(tmp_int[2 + i * 4]), "=r"(tmp_int[3 + i * 4]): "r"(shared_ptr_t));
            shared_load_ptr += LdsmFootprint::kColumn / 8;
        }

        float2* accum_ptr = reinterpret_cast<float2*>(&accumulators);

        #pragma unroll
        for (int col = 0; col < FragmentCount::kColumn; col ++){
            float2* accum_ptr_t = accum_ptr;
            (*accum_ptr).x = float(tmp[2 * col]);
            (*accum_ptr).y = float(tmp[2 * col + 1]);
            #pragma unroll
            for (int row = 1; row < FragmentCount::kRow; row ++){
                accum_ptr_t ++;
                *(accum_ptr_t) = *(accum_ptr);
            }
            accum_ptr += FragmentCount::kRow;
        }

    }

};


template <
    typename ThreadblockTile_Shape_,
    typename WarpTile_Shape_,
    typename OutputOp_,
    typename Mma_
>
struct DefaultMaskPrologue {
    using ThreadblockTile_Shape = ThreadblockTile_Shape_;
    using WarpTile_Shape = WarpTile_Shape_;
    using OutputOp = OutputOp_;
    using Element = typename OutputOp::ElementOutput;
    using Mma = Mma_;

    using MaskIterator = MaskTileIterator<Element, ThreadblockTile_Shape>;
    using Prologue = BroadcastMaskPrologue<ThreadblockTile_Shape, WarpTile_Shape, OutputOp, Mma, MaskIterator>;
};

}  // namespace threadblock
}  // namespace epilogue
}  // namespace cutlass