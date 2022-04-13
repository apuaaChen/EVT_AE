namespace cutlass {
namespace epilogue {
namespace threadblock {

template <
    typename Element_,
    typename WarpTile_Shape_,
    typename ThreadMapLoad_,
    typename OutputTileThreadMap
>
class TransposeOutputTileIterator{
public:

    using Element = Element_;
    using WarpTile_Shape = WarpTile_Shape_;
    using ThreadMapLoad = ThreadMapLoad_;
    // using Params = PredicatedTileIteratorParams;
    using TensorCoord = MatrixCoord;
    using Layout = layout::RowMajor;

    /// Uses a non-template class
    struct Params : PredicatedTileIteratorParams {
        using Base = PredicatedTileIteratorParams;

        CUTLASS_HOST_DEVICE
        Params() { }

        CUTLASS_HOST_DEVICE
        Params(Layout const &layout): 
        PredicatedTileIteratorParams(
            layout.stride(0) * 2,
            make_OutputTileThreadMapDesc<OutputTileThreadMap>()
        ) 
        { }

        CUTLASS_HOST_DEVICE
        Params(Base const &base) : 
        Base(base) { }
    };

    Params params_;

    static int const kShmemLoadElementsPerAccess = 8;

    // Pointer to global memory
    float4 *global_ptr;

    int stride;

    CUTLASS_DEVICE
    TransposeOutputTileIterator(
        PredicatedTileIteratorParams const & params,
        Element *pointer,
        TensorCoord extent,
        int thread_idx,
        TensorCoord threadblock_offset
    ):
      params_(params),
      stride(extent.row() / kShmemLoadElementsPerAccess)
    {
        int group_idx = thread_idx / (ThreadMapLoad::kM * ThreadMapLoad::kN);
        int group_lane_idx = thread_idx % (ThreadMapLoad::kM * ThreadMapLoad::kN);
        int sub_warp_idx = group_lane_idx / ThreadMapLoad::kM;
        int sub_lane_idx = group_lane_idx % ThreadMapLoad::kM;
        int global_offset = (sub_warp_idx + group_idx * WarpTile_Shape::kN) * extent.row() + sub_lane_idx * kShmemLoadElementsPerAccess;

        global_ptr = reinterpret_cast<float4 *>(pointer + threadblock_offset.column() * extent.row() + threadblock_offset.row()) + global_offset / kShmemLoadElementsPerAccess;
    }

    CUTLASS_DEVICE
    void store_with_float4_offset(TensorCoord offset, float4 data){
        *(global_ptr + offset.row() * stride + offset.column()) = data;
    }

    CUTLASS_DEVICE
    void add_pointer_offset(TensorCoord offset){
        global_ptr += offset.row() * stride + offset.column();
    }
};


/// Defines the pipelined transpose epilogue of GEMM kernel
template <
    typename ThreadblockTile_Shape_,
    typename WarpTile_Shape_,
    typename OutputOp_,
    typename Mma_,
    typename OutputTileIterator_
>
class PipelinedTransposeEpilogue{
public:
    using ThreadblockTile_Shape = ThreadblockTile_Shape_;
    using WarpTile_Shape = WarpTile_Shape_;
    using OutputOp = OutputOp_;
    using Mma = Mma_;
    using OutputTileIterator = OutputTileIterator_;

    // number of warps along rows and columns
    using WarpCount = MatrixShape<ThreadblockTile_Shape::kM / WarpTile_Shape::kM,
                                  ThreadblockTile_Shape::kN / WarpTile_Shape::kN>;
    
    // The shape of each fragment
    using FragmentShape = MatrixShape<8, 8>;

    // The number of iterations to store data into shared memory
    using ShmStoreIterations = MatrixShape<WarpTile_Shape::kM / FragmentShape::kRow,
                                           WarpTile_Shape::kN / FragmentShape::kColumn>;
    using TensorCoord = MatrixCoord;

public:
    static int const kShmemStoreElementsPerAccess = 2;
    using ShmemStoreAccessType = float;
    using FragmentAccessType = float2;

    // The shared memory
    static int const kSkew = 8;
    static int const kShmStride = ThreadblockTile_Shape::kM + kSkew;

    // Pointer to fragments
    FragmentAccessType *frag_ptr;

    // Pointer to shared memory
    ShmemStoreAccessType *shared_store_ptr;
    float4 *shared_load_ptr;

    // Pointer to global memory
    float4 *global_ptr;

    using ElementOutput = typename OutputOp::ElementOutput;

    static int const kShmemLoadElementsPerAccess = 8;
    using ThreadMapLoad = cutlass::gemm::GemmShape<ThreadblockTile_Shape::kM / kShmemLoadElementsPerAccess,
                                                   WarpCount::kRow * 32 / (ThreadblockTile_Shape::kM/kShmemLoadElementsPerAccess),
                                                   WarpCount::kColumn>;

    using ShmLoadIterations = cutlass::MatrixShape<1, WarpCount::kColumn * FragmentShape::kColumn / ThreadMapLoad::kN / ThreadMapLoad::kK>;

    static int const buffer_stride_float = FragmentShape::kColumn * WarpCount::kColumn * kShmStride / kShmemStoreElementsPerAccess;
    static int const buffer_stride_float4 = buffer_stride_float / 4;

    int quad_idx;


    struct SharedStorage {
        /// Element type of shared memory
        using Element = ElementOutput;

        /// Layout of shared memory allocation
        using Layout = layout::RowMajor;
        
        /// Tensor reference to shared memory allocation
        using TensorRef = TensorRef<Element, Layout>;

        /// Logical shape of the shared memory tile written to by all warps.
        using Shape = MatrixShape<
            ThreadblockTile_Shape::kM,
            FragmentShape::kColumn * WarpCount::kColumn * 2
        >;

        using StorageShape = MatrixShape<
          Shape::kRow + kSkew,
          Shape::kColumn
        >;

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




public:
    /// Constructor
    CUTLASS_DEVICE
    PipelinedTransposeEpilogue(
        SharedStorage &shared_storage,
        int thread_idx,
        int warp_idx,
        int lane_idx
    )
    {
        int col_quad_idx = lane_idx % 8;
        int row_quad_idx = lane_idx / 8;
        int col_major = col_quad_idx % 4;
        int col_minor = col_quad_idx / 4;
        quad_idx = lane_idx / 4;
        int warp_row_idx = warp_idx % WarpCount::kRow;
        int warp_col_idx = warp_idx / WarpCount::kRow;

        int shared_store_offset = (warp_col_idx * FragmentShape::kColumn + col_major * 2 + col_minor) * kShmStride + warp_row_idx * WarpTile_Shape::kM + row_quad_idx * kShmemStoreElementsPerAccess;

        shared_store_ptr = reinterpret_cast<ShmemStoreAccessType *>(shared_storage.data()) + shared_store_offset / kShmemStoreElementsPerAccess;

        int group_lane_idx = thread_idx % (ThreadMapLoad::kM * ThreadMapLoad::kN);
        int group_idx = thread_idx / (ThreadMapLoad::kM * ThreadMapLoad::kN);
        int sub_warp_idx = group_lane_idx / ThreadMapLoad::kM;
        int sub_lane_idx = group_lane_idx % ThreadMapLoad::kM;
        int shared_load_offset = (sub_warp_idx + group_idx * FragmentShape::kColumn) * kShmStride + sub_lane_idx * kShmemLoadElementsPerAccess;

        shared_load_ptr = reinterpret_cast<float4 *>(shared_storage.data()) + shared_load_offset / kShmemLoadElementsPerAccess;
    }

    /// Streams the results to global memory
    CUTLASS_DEVICE
    void operator()(
        OutputOp const &output_op,
        OutputTileIterator iterator_D,
        typename Mma::FragmentC &accumulators,
        OutputTileIterator iterator_D_
    ){
        FragmentAccessType *frag_ptr = reinterpret_cast<FragmentAccessType *>(&accumulators);
        
        ElementOutput tmp_share[1];
        ElementOutput res[2];
        float* res_vec = reinterpret_cast<float* >(res);


        /*
        * Prologue
        */ 
        float2 *frag_ptr_t = frag_ptr;
        float * shared_store_ptr_t = shared_store_ptr;
        #pragma unroll
        for (int row=0; row < ShmStoreIterations::kRow; row ++){
            // Transpose the fragment with shuffle & convert to output type
            if (quad_idx % 2 == 0){
                *tmp_share = ElementOutput((*(frag_ptr_t)).y);
            } else {
                *tmp_share = ElementOutput((*(frag_ptr_t)).x);
            }
            *reinterpret_cast<uint16_t *>(tmp_share) = __shfl_xor_sync(0xffffffff, *reinterpret_cast<uint16_t *>(tmp_share), 4);
            if (quad_idx % 2 == 0){
                res[0] = ElementOutput((*(frag_ptr_t)).x);
                res[1] = *tmp_share;
            } else {
                res[0] = *tmp_share;
                res[1] = ElementOutput((*(frag_ptr_t)).y);
            }
            // Store the results to shared memory
            *(shared_store_ptr_t) = *(res_vec);
            frag_ptr_t += 1;
            shared_store_ptr_t += FragmentShape::kRow / kShmemStoreElementsPerAccess;
        }

        frag_ptr += ShmStoreIterations::kRow;

        /*
        * Main loop
        */

        #pragma unroll
        for (int col=0; col < ShmStoreIterations::kColumn - 1; col ++){
            // To this point, the results have been written into the shared memory
            // Then we load them back to registers
            frag_ptr_t = frag_ptr;
            shared_store_ptr_t = shared_store_ptr + ((col + 1) % 2) * buffer_stride_float;
            #pragma unroll
            for (int row=0; row < ShmStoreIterations::kRow; row ++){
                if (quad_idx % 2 == 0){
                    *tmp_share = ElementOutput((*(frag_ptr_t)).y);
                } else {
                    *tmp_share = ElementOutput((*(frag_ptr_t)).x);
                }
                *reinterpret_cast<uint16_t *>(tmp_share) = __shfl_xor_sync(0xffffffff, *reinterpret_cast<uint16_t *>(tmp_share), 4);
                if (quad_idx % 2 == 0){
                    res[0] = ElementOutput((*(frag_ptr_t)).x);
                    res[1] = *tmp_share;
                } else {
                    res[0] = *tmp_share;
                    res[1] = ElementOutput((*(frag_ptr_t)).y);
                }
                *(shared_store_ptr_t) = *(res_vec);
                frag_ptr_t += 1;
                shared_store_ptr_t += FragmentShape::kRow / kShmemStoreElementsPerAccess;
            }
            if (col == 0){
                __syncthreads();
            }
            
            #pragma unroll
            for (int t=0; t < ShmLoadIterations::kColumn; t ++){
                // *(global_ptr + t * problem_size.m() * ThreadMapLoad::kN / kShmemLoadElementsPerAccess) = *(shared_load_ptr + (col % 2) * buffer_stride_float4 + t * kShmStride * ThreadMapLoad::kN / kShmemLoadElementsPerAccess);
                iterator_D.store_with_float4_offset({t * ThreadMapLoad::kN, 0}, *(shared_load_ptr + (col % 2) * buffer_stride_float4 + t * kShmStride * ThreadMapLoad::kN / kShmemLoadElementsPerAccess));
            }

            iterator_D.add_pointer_offset({kShmemLoadElementsPerAccess, 0});
            // global_ptr += problem_size.m();
            frag_ptr += ShmStoreIterations::kRow;
            
            __syncthreads();
        }

        /*
        * epilogue
        */ 

        #pragma unroll
        for (int t=0; t < ShmLoadIterations::kColumn; t ++){
            iterator_D.store_with_float4_offset({t * ThreadMapLoad::kN, 0}, *(shared_load_ptr + buffer_stride_float4 + t * kShmStride * ThreadMapLoad::kN / kShmemLoadElementsPerAccess));
            // *(global_ptr + t * problem_size.m() * ThreadMapLoad::kN / kShmemLoadElementsPerAccess) = *(shared_load_ptr + buffer_stride_float4 + t * kShmStride * ThreadMapLoad::kN / kShmemLoadElementsPerAccess);
        }


    }
};

template <
    typename ThreadblockTile_Shape_,
    typename WarpTile_Shape_,
    typename OutputOp_,
    typename Mma_
>
struct DefaultTransposeEpilogue {
    using OutputOp = OutputOp_;
    using ThreadblockTile_Shape = ThreadblockTile_Shape_;
    using WarpTile_Shape = WarpTile_Shape_;
    using ElementOutput = typename OutputOp::ElementOutput;
    using Mma = Mma_;

    static int const kShmemLoadElementsPerAccess = 8;

    // number of warps along rows and columns
    using WarpCount = MatrixShape<ThreadblockTile_Shape::kM / WarpTile_Shape::kM,
                                  ThreadblockTile_Shape::kN / WarpTile_Shape::kN>;

    using ThreadMapLoad = cutlass::gemm::GemmShape<ThreadblockTile_Shape::kM / kShmemLoadElementsPerAccess,
                                                   WarpCount::kRow * 32 / (ThreadblockTile_Shape::kM/kShmemLoadElementsPerAccess),
                                                   WarpCount::kColumn>;

    using OutputTileThreadMap = typename cutlass::epilogue::threadblock::DefaultThreadMapTensorOp<
        ThreadblockTile_Shape,
        WarpTile_Shape,
        1,
        ElementOutput,
        8
    >::Type;
    using OutputTileIterator = TransposeOutputTileIterator<ElementOutput, WarpTile_Shape, ThreadMapLoad, OutputTileThreadMap>;

    using Epilogue = PipelinedTransposeEpilogue<ThreadblockTile_Shape, WarpTile_Shape, OutputOp, Mma, OutputTileIterator>;

};

}  // namespace threadblock
}  // namespace epilogue
}  // namespace cutlass