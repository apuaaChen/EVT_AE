namespace cutlass {
namespace epilogue {
namespace threadblock {


/// Defines the iterator that writes metadata to the global memory
template <
    typename ThreadblockTile_Shape_,
    typename WarpTile_Shape_
>
class MetaTileIterator{
public:
    using ThreadblockTile_Shape = ThreadblockTile_Shape_;
    using WarpTile_Shape = WarpTile_Shape_;
    // number of warps along rows and columns
    using WarpCount = MatrixShape<ThreadblockTile_Shape::kM / WarpTile_Shape::kM,
                                  ThreadblockTile_Shape::kN / WarpTile_Shape::kN>;

    using TensorCoord = MatrixCoord;

    using Layout = layout::ColumnMajor;


    int stride;
    int* global_ptr;
    int row_coord;
    int column_coord;

    int row_bound;
    int column_bound;

    struct Params{

        int stride;

        Params() { }

        CUTLASS_HOST_DEVICE
        Params(Layout const &layout):
        stride(layout.stride()[0]){ }
    };

    // TODO: 

    CUTLASS_DEVICE
    MetaTileIterator(
        Params params, 
        uint16_t* meta_pointer,
        TensorCoord extent,
        int thread_id,
        TensorCoord threadblock_offset
    ):
    stride(params.stride),
    row_bound(extent.row()),
    column_bound(extent.column() / 2)
    {
        int warp_id = __shfl_sync(0xffffffff, threadIdx.x / 32, 0);
        int lane_id = threadIdx.x % 32;

        int warp_row_id = warp_id % WarpCount::kRow;
        int warp_col_id = warp_id / WarpCount::kRow;

        column_coord = (threadblock_offset.column() / 32) + warp_col_id * (WarpTile_Shape::kN / 32);
        row_coord = threadblock_offset.row() + warp_row_id * WarpTile_Shape::kM + lane_id;

        // global_ptr = reinterpret_cast<int*>(meta_pointer) + ((threadblock_offset.column() / 32) + warp_col_id * (WarpTile_Shape::kN / 32)) * stride + threadblock_offset.row() + warp_row_id * WarpTile_Shape::kM + lane_id;
        global_ptr = reinterpret_cast<int*>(meta_pointer) + column_coord * stride + row_coord;

    }

    CUTLASS_DEVICE
    void store_with_offset(TensorCoord offset, int data){
        // check predicate
        if ((column_coord + offset.column() < column_bound) && (row_coord + offset.row() < row_bound)){
            *(global_ptr + offset.column() * stride + offset.row()) = data;
        }
    }

    CUTLASS_DEVICE
    int load_with_offset(TensorCoord offset){
       return *(global_ptr + offset.column() * stride + offset.row());
    }

    CUTLASS_DEVICE
    void add_pointer_offset(int64_t offset){
        global_ptr += offset / 2;
    }
};


/// Defines the iterator that writes the nonzeros to the global memory
template <
    typename Element_,
    typename ThreadblockTile_Shape_,
    typename WarpTile_Shape_
>
class NnzTileIterator{
public:
    using ThreadblockTile_Shape = ThreadblockTile_Shape_;
    using WarpTile_Shape = WarpTile_Shape_;
    using Element = Element_;

    using TensorCoord = MatrixCoord;

    // number of warps along rows and columns
    using WarpCount = MatrixShape<ThreadblockTile_Shape::kM / WarpTile_Shape::kM,
                                  ThreadblockTile_Shape::kN / WarpTile_Shape::kN>;
    
    // number of threads available
    static int const kNumThread = WarpCount::kCount * 32;

    // elements per access
    static int const kElementsPerAccess = 8;

    using ThreadMap = MatrixShape<
        kNumThread / (ThreadblockTile_Shape::kN / 2 / kElementsPerAccess),
        ThreadblockTile_Shape::kN / 2 / kElementsPerAccess>;

    using StoreIteration = MatrixShape<
        ThreadblockTile_Shape::kM / ThreadMap::kRow, 1>;

    
    using OutputTileThreadMap = typename cutlass::epilogue::threadblock::DefaultThreadMapTensorOp<
        ThreadblockTile_Shape,
        WarpTile_Shape,
        1,
        Element,
        8
    >::Type;

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




    int stride;
    float4* global_ptr;

    int row_coord;
    int column_coord;

    int row_bound;
    int column_bound;

    CUTLASS_DEVICE
    NnzTileIterator(
        Params params,
        Element* nnz_pointer,
        TensorCoord extent,
        int thread_id,
        TensorCoord threadblock_offset
    ):
    stride(extent.column() / 2),
    row_bound(extent.row()),
    column_bound(extent.column() / 2)
    {
        int lane_id = thread_id % ThreadMap::kColumn;
        int group_id = thread_id / ThreadMap::kColumn;

        row_coord = threadblock_offset.row() + group_id;

        column_coord = (threadblock_offset.column() / 2) + lane_id * kElementsPerAccess;

        // global_ptr = reinterpret_cast<float4 *>(nnz_pointer + (threadblock_offset.row() + group_id) * stride + threadblock_offset.column() / 2) + lane_id;

        global_ptr = reinterpret_cast<float4 *>(nnz_pointer + row_coord * stride + column_coord);
    }

    CUTLASS_DEVICE
    void store(float4 data){
        if ((row_coord < row_bound) && (column_coord < column_bound)){
            *(global_ptr) = data;
        }
    }

    CUTLASS_DEVICE
    void add_pointer_offset(TensorCoord offset){
        row_coord += offset.row();
        column_coord += offset.column() * kElementsPerAccess;
        global_ptr += offset.row() * stride / kElementsPerAccess + offset.column();
    }

    CUTLASS_DEVICE
    void add_pointer_offset(int64_t offset){
        global_ptr += offset / kElementsPerAccess;
    }
};



/// Defines the epilogue that dynamically prunes the output of GEMM kernel to 2:4 sparsity
template <
    typename ThreadblockTile_Shape_,
    typename WarpTile_Shape_,
    typename OutputOp_,
    typename Mma_,
    typename IteratorE_,
    typename NnzTileIterator_,
    bool kShfl=false
>
class DP24Epilogue{
public:
    using ThreadblockTile_Shape = ThreadblockTile_Shape_;
    using WarpTile_Shape = WarpTile_Shape_;
    using OutputOp = OutputOp_;
    using Mma = Mma_;
    using IteratorE = IteratorE_;
    using NnzIterator = NnzTileIterator_;

    using TensorCoord = MatrixCoord;

    // number of warps along rows and columns
    using WarpCount = MatrixShape<ThreadblockTile_Shape::kM / WarpTile_Shape::kM,
                                  ThreadblockTile_Shape::kN / WarpTile_Shape::kN>;
    
    // The shape of each fragment
    using FragmentShape = MatrixShape<8, 8>;

    using FragmentCount = MatrixShape<WarpTile_Shape::kM / FragmentShape::kRow,
                                      WarpTile_Shape::kN / FragmentShape::kColumn>;

    // The shape of metadata generation unit
    using MetaUnitShape = MatrixShape<32, 32>;

    // The number of iterations to produce metadata
    using MetaIterations = cutlass::MatrixShape<WarpTile_Shape::kM / MetaUnitShape::kRow,
                                                WarpTile_Shape::kN / MetaUnitShape::kColumn>;
    

    static int const kSparse = 2;

    static int const kElementsPerElementE = 128 / cutlass::sizeof_bits<half_t>::value;

    using ElementE = uint16_t;

    using LayoutE = layout::ColumnMajor;

    using ElementOutput = typename OutputOp::ElementOutput;

    static int const kSkew = 8;


    struct SharedStorage {
        /// Element type of shared memory
        using Element = ElementOutput;

        /// Layout of shared memory allocation
        using Layout = layout::RowMajor;

        /// Tensor reference to shared memory allocation
        using TensorRef = TensorRef<Element, Layout>;

        /// Logical shape of the shared memory tile written to by all warps
        using Shape = MatrixShape<
            ThreadblockTile_Shape::kM, 
            ThreadblockTile_Shape::kN / 2>;
        
        using StorageShape = MatrixShape<
            Shape::kRow,
            Shape::kColumn + kSkew
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

    /// The pointer that loads data from register to shared memory
    float* shared_load_ptr;

    /// The pointer that accesses the data in shared memory for global store
    float4* shared_store_ptr;

    int warp_row_id;
    int warp_col_id;

    int lane_id;


    /// Constructor
    CUTLASS_DEVICE
    DP24Epilogue(
        SharedStorage &shared_storage,
        int thread_id_,
        int warp_id,
        int lane_id_
    ):
    lane_id(lane_id_)
    {

        warp_row_id = warp_id % WarpCount::kRow;
        warp_col_id = warp_id / WarpCount::kRow;
        int th_group_id = lane_id / 4;
        int group_lane_id = lane_id % 4;

        int shmem_init_offset = (warp_row_id * WarpTile_Shape::kM + th_group_id) * SharedStorage::StorageShape::kColumn + warp_col_id * WarpTile_Shape::kN / 2 + group_lane_id * 2;

        shared_load_ptr = reinterpret_cast<float*>(shared_storage.data()) + shmem_init_offset / 2;

        // set the shared_store_ptr
        int sub_warp_lane_id = thread_id_ % NnzIterator::ThreadMap::kColumn;
        int sub_warp_id = thread_id_ / NnzIterator::ThreadMap::kColumn;

        shared_store_ptr = reinterpret_cast<float4*>(shared_storage.data()) + (sub_warp_id * SharedStorage::StorageShape::kColumn) / 8 + sub_warp_lane_id;
    }


    /// generate the metadata
    CUTLASS_DEVICE
    void get_meta_data(
        typename Mma::FragmentC &accumulators,
        int lane_id,
        IteratorE iterator_E,
        typename OutputOp::Params output_op
    ){
        float2* frag_ptr = reinterpret_cast<float2*>(&accumulators);
        int th_group_id = lane_id % 4;

        #pragma unroll
        for (int m_step = 0; m_step < MetaIterations::kRow; m_step ++){
            #pragma unroll
            for (int n_step = 0; n_step < MetaIterations::kColumn; n_step ++){
                // Each step processes a 32 x 32 Tile
                // Step 1: prun the results in the register file
                int16_t meta[8] = {0};
                // The pruning first occurs in 8 x 16 Tile
                #pragma unroll
                for (int i = 0; i < 2; i ++){
                    #pragma unroll
                    for (int j = 0; j < 2; j++){
                        #pragma unroll
                        for (int k = 0; k < 2; k++){
                            int m_i = m_step * 4 + i * 2 + k;
                            int n_i = n_step * 4 + j * 2;
                            float2* frag1 = frag_ptr + n_i * FragmentCount::kRow + m_i;
                            float2* frag2 = frag_ptr + (n_i + 1) * FragmentCount::kRow + m_i;

                            float data[4] = {(*frag1).x, (*frag1).y, (*frag2).x, (*frag2).y};
                            ElementOutput data_16[4] = {
                                ElementOutput(data[0]), ElementOutput(data[1]),
                                ElementOutput(data[2]), ElementOutput(data[3])
                            };

                            ElementOutput value[2] = {data_16[0], data_16[1]};
                            int16_t meta_bit = 4;
                            float max_val = data[0] + data[1];

                            if (data[0] + data[2] > max_val){
                                meta_bit = 8;
                                value[1] = data_16[2];
                                max_val = data[0] + data[2];
                            }

                            if (data[0] + data[3] > max_val){
                                meta_bit = 12;
                                value[1] = data_16[3];
                                max_val = data[0] + data[3];
                            }

                            if (data[1] + data[2] > max_val){
                                meta_bit = 9;
                                value[0] = data_16[1];
                                value[1] = data_16[2];
                                max_val = data[1] + data[2];
                            }

                            if (data[1] + data[3] > max_val){
                                meta_bit = 13;
                                value[0] = data_16[1];
                                value[1] = data_16[3];
                                max_val = data[1] + data[3];
                            }

                            if (data[2] + data[3] > max_val){
                                meta_bit = 14;
                                value[0] = data_16[2];
                                value[1] = data_16[3];
                            }

                            meta[4 * i + 2 * j + k] = meta_bit << (th_group_id * 4);
                            value[0] *= output_op.alpha;
                            value[1] *= output_op.alpha;
                            // TODO: write the value to shared memory
                            *(shared_load_ptr + ((m_i * FragmentShape::kRow) * SharedStorage::StorageShape::kColumn + n_i * FragmentShape::kColumn / 2) / 2) = *reinterpret_cast<float*>(value);
                        }
                        // Collect the meta dat at the target thread.
                        #pragma unroll
                        for (int k = 0; k < 2; k++){
                            if (i == 0){
                                meta[2 * j + k] |= __shfl_down_sync(0xffffffff, meta[2 * j + k], 2); 
                            }else{
                                meta[2 * j + k + 4] |= __shfl_up_sync(0xffffffff, meta[2 * j + k + 4], 2);
                            }
                            if (j == 0){
                                meta[i * 4 + k] |= __shfl_down_sync(0xffffffff, meta[i * 4 + k], 1);
                            }else{
                                meta[i * 4 + 2 + k] |= __shfl_up_sync(0xffffffff, meta[i * 4 + 2 + k], 1);
                            }
                        }
                        
                    }
                    
                }

                // All the metadat are alreadly collected in meta[8]
                // Step 2: Switch the meta data (Nothing needs to be done)
                // Step 3: vectorize
                int* meta_vec = reinterpret_cast<int *>(meta);
                // Step 4: put the meta data to the first element of the vec
                if (th_group_id == 1) meta_vec[0] = meta_vec[1];
                else if (th_group_id == 2) meta_vec[0] = meta_vec[2];
                else if (th_group_id == 3) meta_vec[0] = meta_vec[3];

                // TODO: write the value to global memory
                iterator_E.store_with_offset({m_step * 32, n_step}, meta_vec[0]);
            }
        }

    }


    /// generate the metadata with target
    CUTLASS_DEVICE
    void get_meta_data(
        typename Mma::FragmentC &accumulators,
        int lane_id,
        IteratorE iterator_E,
        typename OutputOp::Params output_op,
        TensorCoord threadblock_offset,
        int64_t* target,
        int64_t* target_sp
    ){
        float2* frag_ptr = reinterpret_cast<float2*>(&accumulators);
        int th_group_id = lane_id % 4;

        #pragma unroll
        for (int m_step = 0; m_step < MetaIterations::kRow; m_step ++){
            // load the targets
            int64_t targets[4];
            int row_idx = threadblock_offset.row() + warp_row_id * WarpTile_Shape::kM + m_step * 32 + lane_id / 4;
            int row_idx_t = row_idx;
            #pragma unroll
            for (int i = 0; i < 4; i++){
                targets[i] = target[row_idx_t];
                row_idx_t += 8;
            }

            #pragma unroll
            for (int n_step = 0; n_step < MetaIterations::kColumn; n_step ++){
                // Each step processes a 32 x 32 Tile
                // Step 1: prun the results in the register file
                int16_t meta[8] = {0};
                // The pruning first occurs in 8 x 16 Tile
                #pragma unroll
                for (int i = 0; i < 2; i ++){
                    #pragma unroll
                    for (int j = 0; j < 2; j++){
                        #pragma unroll
                        for (int k = 0; k < 2; k++){
                            int m_i = m_step * 4 + i * 2 + k;
                            int n_i = n_step * 4 + j * 2;
                            float2* frag1 = frag_ptr + n_i * FragmentCount::kRow + m_i;
                            float2* frag2 = frag_ptr + (n_i + 1) * FragmentCount::kRow + m_i;

                            float data[4] = {(*frag1).x, (*frag1).y, (*frag2).x, (*frag2).y};
                            ElementOutput data_16[4] = {
                                ElementOutput(data[0]), ElementOutput(data[1]),
                                ElementOutput(data[2]), ElementOutput(data[3])
                            };

                            // check if the target is in the range
                            // get current column
                            int column = threadblock_offset.column() + warp_col_id * WarpTile_Shape::kN + n_i * 8 + th_group_id * 4;
                            int row = i * 2 + k;
                            int column_relative = targets[row] - column;
                            // if the target is in the current pruning range
                            if (column_relative >= 0 && column_relative < 4){

                                if (column_relative == 0){
                                    data[0] = 2147483.0; // just a very large number 
                                }
                                
                                if (column_relative == 1){
                                    data[1] = 2147483.0; // just a very large number 
                                }

                                if (column_relative == 2){
                                    data[2] = 2147483.0; // just a very large number 
                                }

                                if (column_relative == 3){
                                    data[3] = 2147483.0; // just a very large number 
                                }

                            }

                            ElementOutput value[2] = {data_16[0], data_16[1]};
                            float value_float[2] = {data[0], data[1]};
                            
                            int16_t meta_bit = 4;
                            float max_val = data[0] + data[1];

                            if (data[0] + data[2] > max_val){
                                meta_bit = 8;
                                value[1] = data_16[2];
                                value_float[1] = data[2];
                                max_val = data[0] + data[2];
                            }

                            if (data[0] + data[3] > max_val){
                                meta_bit = 12;
                                value[1] = data_16[3];
                                value_float[1] = data[3];
                                max_val = data[0] + data[3];
                            }

                            if (data[1] + data[2] > max_val){
                                meta_bit = 9;
                                value[0] = data_16[1];
                                value[1] = data_16[2];
                                value_float[0] = data[1];
                                value_float[1] = data[2];
                                max_val = data[1] + data[2];
                            }

                            if (data[1] + data[3] > max_val){
                                meta_bit = 13;
                                value[0] = data_16[1];
                                value[1] = data_16[3];
                                value_float[0] = data[1];
                                value_float[1] = data[3];
                                max_val = data[1] + data[3];
                            }

                            if (data[2] + data[3] > max_val){
                                meta_bit = 14;
                                value[0] = data_16[2];
                                value[1] = data_16[3];
                                value_float[0] = data[2];
                                value_float[1] = data[3];
                            }

                            meta[4 * i + 2 * j + k] = meta_bit << (th_group_id * 4);
                            value[0] *= output_op.alpha;
                            value[1] *= output_op.alpha;
                            // TODO: write the value to shared memory
                            *(shared_load_ptr + ((m_i * FragmentShape::kRow) * SharedStorage::StorageShape::kColumn + n_i * FragmentShape::kColumn / 2) / 2) = *reinterpret_cast<float*>(value);

                            if (value_float[0] > 2147480.0){
                                target_sp[row_idx + row * 8] = column / 2;
                            }

                            if (value_float[1] > 2147480.0){
                                target_sp[row_idx + row * 8] = column / 2 + 1;
                            }
                        }
                        // Collect the meta dat at the target thread.
                        #pragma unroll
                        for (int k = 0; k < 2; k++){
                            if (i == 0){
                                meta[2 * j + k] |= __shfl_down_sync(0xffffffff, meta[2 * j + k], 2); 
                            }else{
                                meta[2 * j + k + 4] |= __shfl_up_sync(0xffffffff, meta[2 * j + k + 4], 2);
                            }
                            if (j == 0){
                                meta[i * 4 + k] |= __shfl_down_sync(0xffffffff, meta[i * 4 + k], 1);
                            }else{
                                meta[i * 4 + 2 + k] |= __shfl_up_sync(0xffffffff, meta[i * 4 + 2 + k], 1);
                            }
                        }
                        
                    }
                    
                }

                // All the metadat are alreadly collected in meta[8]
                // Step 2: Switch the meta data (Nothing needs to be done)
                // Step 3: vectorize
                int* meta_vec = reinterpret_cast<int *>(meta);
                // Step 4: put the meta data to the first element of the vec
                if (th_group_id == 1) meta_vec[0] = meta_vec[1];
                else if (th_group_id == 2) meta_vec[0] = meta_vec[2];
                else if (th_group_id == 3) meta_vec[0] = meta_vec[3];

                // TODO: write the value to global memory
                iterator_E.store_with_offset({m_step * 32, n_step}, meta_vec[0]);
            }
        }

    }

    /// prune the result based on the metadata
    CUTLASS_DEVICE
    void pruning(
        typename Mma::FragmentC &accumulators,
        int lane_id,
        IteratorE iteratorE,
        typename OutputOp::Params output_op
    ){
        float2* frag_ptr = reinterpret_cast<float2*>(&accumulators);
        int th_group_id = lane_id % 4;

        #pragma unroll
        for (int m_step = 0; m_step < MetaIterations::kRow; m_step ++){
            #pragma unroll
            for (int n_step = 0; n_step < MetaIterations::kColumn; n_step ++){
                // Each step processes a 32 x 32 Tile
                // Step 1: load the metadata to registers
                int16_t meta[2];
                int* meta_vec = reinterpret_cast<int*>(meta);
                *(meta_vec) = iteratorE.load_with_offset({m_step * 32, n_step});
                // Step 2: prune the results in the register file
                // The pruning first occurs in 8 x 16 Tile
                #pragma unroll
                for (int i = 0; i < 2; i ++){
                    #pragma unroll
                    for (int j = 0; j < 2; j++){
                        #pragma unroll
                        for (int k = 0; k < 2; k++){
                            int m_i = m_step * 4 + i * 2 + k;
                            int n_i = n_step * 4 + j * 2;
                            float2* frag1 = frag_ptr + n_i * FragmentCount::kRow + m_i;
                            float2* frag2 = frag_ptr + (n_i + 1) * FragmentCount::kRow + m_i;

                            // get the elements
                            ElementOutput data_16[4] = {
                                ElementOutput((*frag1).x), ElementOutput((*frag1).y),
                                ElementOutput((*frag2).x), ElementOutput((*frag2).y)
                            };

                            // Perform shfl here
                            if (kShfl){
                                // original: T0{0,1} T1{0,1} T2{0,1} T3{0,1} T0{2,3} T1{2,3} T2{2,3} T3{2,3}
                                float* data_16_vec = reinterpret_cast<float*>(data_16);

                                // step 1: switch T2{0,1} T3{0,1} with T0{2, 3}, T1{2,3}
                                float tmp;
                                int sub_sub_group_id = (lane_id % 4) / 2;
                                // T0{0,1} T1{0,1} tmp(T2{0,1} T3{0,1}) tmp(T0{2,3} T1{2,3}) T2{2,3} T3{2,3}
                                if (sub_sub_group_id == 0) tmp = *(data_16_vec + 1);
                                else tmp = *(data_16_vec);

                                // shfl: T0{0,1} T1{0,1} T0{2,3} T1{2,3} T2{0,1} T3{0,1} T2{2,3} T3{2,3}
                                tmp = __shfl_xor_sync(0xffffffff, tmp, 2);
                                if (sub_sub_group_id == 0) *(data_16_vec + 1)=tmp;
                                else *(data_16_vec)=tmp;

                                // T0{0,1} tmp(T1{0,1}) tmp(T0{2,3}) T1{2,3} T2{0,1} tmp(T3{0,1}) tmp(T2{2,3}) T3{2,3}
                                int sub_sub_sub_group_id = lane_id % 2;
                                if (sub_sub_sub_group_id == 0) tmp = *(data_16_vec + 1);
                                else tmp = *(data_16_vec);

                                // shfl: T0{0,1} T0{2,3} T1{0,1} T1{2,3} T2{0,1} T2{2,3} T3{0,1} T3{2,3}
                                tmp = __shfl_xor_sync(0xffffffff, tmp, 1);
                                if (sub_sub_sub_group_id == 0) *(data_16_vec + 1)=tmp;
                                else *(data_16_vec)=tmp;
                            }

                            // get the meta bits
                            int src_lane = (lane_id / 4) * 4 + j + i * 2;
                            int16_t meta_bit = __shfl_sync(0xffffffff, meta[k], src_lane);
                            meta_bit = (meta_bit >> (th_group_id * 4)) & 0xf;

                            ElementOutput value[2] = {data_16[0], data_16[1]};

                            // get the values
                            if (meta_bit == 8){
                                value[1] = data_16[2];
                            }

                            if (meta_bit == 12){
                                value[1] = data_16[3];
                            }

                            if (meta_bit == 9){
                                value[0] = data_16[1];
                                value[1] = data_16[2];
                            }

                            if (meta_bit == 13){
                                value[0] = data_16[1];
                                value[1] = data_16[3];
                            }

                            if (meta_bit == 14){
                                value[0] = data_16[2];
                                value[1] = data_16[3];
                            }

                            value[0] *= output_op.alpha;
                            value[1] *= output_op.alpha;
                            // TODO: write the value to shared memory
                            *(shared_load_ptr + ((m_i * FragmentShape::kRow) * SharedStorage::StorageShape::kColumn + n_i * FragmentShape::kColumn / 2) / 2) = *reinterpret_cast<float*>(value);
                        }

                    }
                }
            }
        }
    }

    CUTLASS_DEVICE
    void store_nnz(
        NnzIterator iterator_D
    ){
        #pragma unroll
        for (int m=0; m < NnzIterator::StoreIteration::kRow; m++){
            iterator_D.store(*shared_store_ptr);
            shared_store_ptr += NnzIterator::ThreadMap::kRow * SharedStorage::StorageShape::kColumn / 8;
            iterator_D.add_pointer_offset({NnzIterator::ThreadMap::kRow, 0});
        }
    }

    CUTLASS_DEVICE
    void operator()(
        typename OutputOp::Params output_op,
        IteratorE iterator_E,
        NnzIterator iterator_D,
        typename Mma::FragmentC &accumulators,
        IteratorE iterator_E_,
        NnzIterator iterator_D_
    ){
        get_meta_data(accumulators, lane_id, iterator_E, output_op);
        __syncthreads();
        store_nnz(iterator_D);
    }
};

template <
    typename ThreadblockTile_Shape_,
    typename WarpTile_Shape_,
    typename OutputOp_,
    typename Mma_,
    bool kShfl=false
>
struct DefaultSddmmEpilogue {
    using OutputOp = OutputOp_;
    using ThreadblockTile_Shape = ThreadblockTile_Shape_;
    using WarpTile_Shape = WarpTile_Shape_;
    using ElementOutput = typename OutputOp::ElementOutput;
    using Mma = Mma_;

    using IteratorE = MetaTileIterator<ThreadblockTile_Shape, WarpTile_Shape>;
    using NnzIterator = NnzTileIterator<ElementOutput, ThreadblockTile_Shape, WarpTile_Shape>;
    using Epilogue = DP24Epilogue<ThreadblockTile_Shape, WarpTile_Shape, OutputOp, Mma, IteratorE, NnzIterator, kShfl>;
};

}  // namespace threadblock
}  // namespace epilogue
}  // namespace cutlass