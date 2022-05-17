namespace cutlass {
namespace gemm {
namespace threadblock {

/// Structure to reduce the sparse operand along the dense axis
template <
    typename Threadblock_Shape_,
    typename WarpTile_Shape_,
    typename Instruction_Shape_,
    typename Element,
    typename WarpTransformedFragment_
>
class OperandReduceOp{
public:
    using Threadblock_Shape = Threadblock_Shape_;
    using WarpTile_Shape = WarpTile_Shape_;
    using Instruction_Shape = Instruction_Shape_;

    using WarpCount = MatrixShape<Threadblock_Shape::kM / WarpTile_Shape::kM,
                                  Threadblock_Shape::kN / WarpTile_Shape::kN>;

    static const int kBufferSize = WarpTile_Shape::kM / 8;

    // static const int kInnerReduceSize = Instruction_Shape::kM;
    
    using OuterReduceIteration = MatrixShape<WarpTile_Shape::kM / Instruction_Shape::kK, 1>;
    
    // declare the input array
    using ReduceBuffer = Array<float, kBufferSize>;
    using WarpTransformedFragment = WarpTransformedFragment_;
    
    
    bool active;
    int warp_id;
    int lane_id;

    CUTLASS_DEVICE
    OperandReduceOp(
        int lane_id_,
        int warp_id_,
        GemmCoord threadblock_tile_offset
    ):
    warp_id(warp_id_),
    lane_id(lane_id_)
    {
        active = (warp_id < WarpCount::kRow) && (threadblock_tile_offset.n() == 0);
    }

    CUTLASS_DEVICE
    void clear_buffer(
        ReduceBuffer &buffer
    ){
        if (active){
            float* buffer_float = reinterpret_cast<float*>(&buffer);
            #pragma unroll
            for (int i=0; i < kBufferSize; i++){
                *(buffer_float + i) = 0.0f;
            }
        }
    }

    CUTLASS_DEVICE
    void get_operands(
        WarpTransformedFragment* frag_A_,
        ReduceBuffer &buffer
    ){
        if(active){
            Element* frag_A = reinterpret_cast<Element*>(frag_A_);
            float* buffer_float = reinterpret_cast<float*>(&buffer);

            #pragma unroll
            for (int i=0; i < OuterReduceIteration::kRow; i++){
                // This contains a 32x32 block

                Element psum = Element(0.0f);
                psum += *(frag_A) + *(frag_A + 1);
                psum += *(frag_A + 4) + *(frag_A + 5);
                psum += *(frag_A + 16) + *(frag_A + 17);
                psum += *(frag_A + 20) + *(frag_A + 21);
                *(buffer_float) += float(psum);

                psum = Element(0.0f);
                psum += *(frag_A + 2) + *(frag_A + 3);
                psum += *(frag_A + 6) + *(frag_A + 7);
                psum += *(frag_A + 18) + *(frag_A + 19);
                psum += *(frag_A + 22) + *(frag_A + 23);
                *(buffer_float + 1) += float(psum);

                psum = Element(0.0f);
                psum += *(frag_A + 8) + *(frag_A + 9);
                psum += *(frag_A + 12) + *(frag_A + 13);
                psum += *(frag_A + 24) + *(frag_A + 25);
                psum += *(frag_A + 28) + *(frag_A + 29);
                *(buffer_float + 2) += float(psum);

                psum = Element(0.0f);
                psum += *(frag_A + 10) + *(frag_A + 11);
                psum += *(frag_A + 14) + *(frag_A + 15);
                psum += *(frag_A + 26) + *(frag_A + 27);
                psum += *(frag_A + 30) + *(frag_A + 31);
                *(buffer_float + 3) += float(psum);

                buffer_float += 4;
                frag_A += 32;
            }
        }
    }

    CUTLASS_DEVICE
    void write_results(
        ReduceBuffer &buffer,
        GemmCoord problem_size,
        GemmCoord threadblock_tile_offset,
        Element* output_reduce,
        int batch_idx
    ){
        if (active){
            float* buffer_float = reinterpret_cast<float*>(&buffer);

            #pragma unroll
            for (int i=0; i < kBufferSize; i ++){
                buffer_float[i] += __shfl_down_sync(0xffffffff, buffer_float[i], 2);
                buffer_float[i] += __shfl_down_sync(0xffffffff, buffer_float[i], 1);
            }

            // compute the output pointer
            Element* output_ptr = 
                output_reduce + problem_size.m() * batch_idx + threadblock_tile_offset.m() * Threadblock_Shape::kM + 
                (warp_id % WarpCount::kRow) * WarpTile_Shape::kM + lane_id / 4;
            
            if (lane_id % 4 == 0){
                #pragma unroll
                for (int i=0; i < kBufferSize; i++){
                    (*output_ptr) = Element(*(buffer_float + i));
                    output_ptr += 8;
                }
            }

        }
    }

};

}  // namespace threadblock
}  // namespace gemm
}  // namespace cutlass