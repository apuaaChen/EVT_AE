#include <cuda.h>
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <iostream>
#include "helper.h"
#include <stdio.h>
#include <vector>
#include "cuda_bf16.h"


__device__ __forceinline__ int reorder_index(int row, int col, int m){
    int dest_row = (row / 32) * 32 + (row % 8) * 4 + (row % 32) / 8;
    int dest_col = col;

    if (((dest_row % 2) == 0) && ((dest_col % 2) == 1)){
        ++dest_row;
        --dest_col;
    }else if (((dest_row % 2) == 1) && ((dest_col % 2) == 0)){
        --dest_row;
        ++dest_col;
    }

    int column_major = dest_col / 2;
    int column_minor = dest_col % 2;
    int idx = column_major * 2 * m + dest_row * 2 + column_minor;
    return idx;
}

template <typename Scalar16>
__device__ void Dense2Sparse_(
    int m, int k,
    const Scalar16* __restrict__ dense_matrix,
    Scalar16* __restrict__ sparse_matrix,
    Scalar16* __restrict__ uncompressed_matrix,
    int16_t * __restrict__ metadata,
    int16_t * __restrict__ metadata_reorder)
{
    // One thread block per row
    int m_index = blockIdx.x;

    const Scalar16* dense_matrix_t = dense_matrix + m_index * k;
    Scalar16* uncompressed_matrix_t = uncompressed_matrix + m_index * k;
    Scalar16* sparse_matrix_t = sparse_matrix + m_index * (k/2);
    int16_t* metadata_t = metadata + m_index * (k / 16);

    // Each thread processes 16 elements in each iteration
    for(int i = threadIdx.x; i < k / 16; i += blockDim.x){
        int16_t meta = 0;
        // Loop through the threadgroup
        #pragma unroll
        for (int j = 0; j < 4; j++){
            Scalar16 data[4];
            float data_float[4];

            #pragma unroll
            for (int v = 0; v < 4; v++){
                data[v] = __ldg(dense_matrix_t + 16 * i + 4 * j + v);
                data_float[v] = __bfloat162float(data[v]);
            }
            
            // TTFF
            Scalar16 value_sp[2] = {data[0], data[1]};
            Scalar16 value_uncompressed[4] = {data[0], data[1], __float2bfloat16(0.0f), __float2bfloat16(0.0f)};
            int16_t meta_bit = 4;
            float max_val = data_float[0] + data_float[1];

            // TFTF
            if (data_float[0] + data_float[2] > max_val){
                meta_bit = 8;
                value_sp[0] = data[0];
                value_sp[1] = data[2];
                value_uncompressed[0] = data[0];
                value_uncompressed[1] = __float2bfloat16(0.0f);
                value_uncompressed[2] = data[2];
                value_uncompressed[3] = __float2bfloat16(0.0f);
                max_val = data_float[0] + data_float[2];
            }

            // TFFT
            if (data_float[0] + data_float[3] > max_val){
                meta_bit = 12;
                value_sp[0] = data[0];
                value_sp[1] = data[3];
                value_uncompressed[0] = data[0];
                value_uncompressed[1] = __float2bfloat16(0.0f);
                value_uncompressed[2] = __float2bfloat16(0.0f);
                value_uncompressed[3] = data[3];
                max_val = data_float[0] + data_float[3];
            }

            // FTTF
            if (data_float[1] + data_float[2] > max_val){
                meta_bit = 9;
                value_sp[0] = data[1];
                value_sp[1] = data[2];
                value_uncompressed[0] = __float2bfloat16(0.0f);
                value_uncompressed[1] = data[1];
                value_uncompressed[2] = data[2];
                value_uncompressed[3] = __float2bfloat16(0.0f);
                max_val = data_float[1] + data_float[2];
            }

            // FTFT
            if (data_float[1] + data_float[3] > max_val){
                meta_bit = 13;
                value_sp[0] = data[1];
                value_sp[1] = data[3];
                value_uncompressed[0] = __float2bfloat16(0.0f);
                value_uncompressed[1] = data[1];
                value_uncompressed[2] = __float2bfloat16(0.0f);
                value_uncompressed[3] = data[3];
                max_val = data_float[1] + data_float[3];
            }

            // FFTT
            if (data_float[2] + data_float[3] > max_val){
                meta_bit = 14;
                value_sp[0] = data[2];
                value_sp[1] = data[3];
                value_uncompressed[0] = __float2bfloat16(0.0f);
                value_uncompressed[1] = __float2bfloat16(0.0f);
                value_uncompressed[2] = data[2];
                value_uncompressed[3] = data[3];

            }

            meta |= meta_bit << (j * 4);
            #pragma unroll
            for (int v = 0; v < 4; v++){
                *(uncompressed_matrix_t + 16 * i + 4 * j + v) = value_uncompressed[v];
            }
            *(sparse_matrix_t + 8 * i + 2 * j) = value_sp[0];
            *(sparse_matrix_t + 8 * i + 2 * j + 1) = value_sp[1];
        }
        *(metadata_t + i) = meta;

        int idx = reorder_index(m_index, i, m);
        *(metadata_reorder + idx) = meta;
    }
}


template <typename Scalar>
__global__ void Dense2Sparse(
    int m, int k,
    const Scalar* __restrict__ dense_matrix,
    Scalar* __restrict__ sparse_matrix,
    Scalar* __restrict__ uncompressed_matrix,
    int16_t * __restrict__ metadata,
    int16_t * __restrict__ metadata_reorder)
{
    Dense2Sparse_<Scalar>(m, k, dense_matrix, sparse_matrix, uncompressed_matrix, metadata, metadata_reorder);
}


std::vector<torch::Tensor> batched_dense2sparse_cuda(
    torch::Tensor dense_matrix)
{
    // Get problem size
    int m = dense_matrix.size(-2);
    int k = dense_matrix.size(-1);
    int batch_size = dense_matrix.numel() / (m * k);

    int meta_ratio;
    if (dense_matrix.dtype() == torch::kBFloat16){
        meta_ratio = 16;
    }
    else{
        meta_ratio = 8;
    }

    // Initiate output matrices
    auto options_val = torch::TensorOptions().dtype(dense_matrix.dtype()).device(dense_matrix.device());
    auto options_meta = torch::TensorOptions().dtype(torch::kInt16).device(dense_matrix.device());

    torch::Tensor sparse_matrix;
    torch::Tensor uncompressed_matrix;
    torch::Tensor metadata;
    torch::Tensor metadata_reorder;

    // For batched implementation
    if (batch_size > 1){
        sparse_matrix = torch::empty({batch_size, m, k/2}, options_val);
        uncompressed_matrix = torch::empty({batch_size, m, k}, options_val);
        metadata = torch::empty({batch_size, m, k/meta_ratio}, options_meta);
        metadata_reorder = torch::empty({batch_size, m, k/meta_ratio}, options_meta);
    }
    // For single Matrix
    else{
        sparse_matrix = torch::empty({m, k/2}, options_val);
        uncompressed_matrix = torch::empty({m, k}, options_val);
        metadata = torch::empty({m, k/meta_ratio}, options_meta);
        metadata_reorder = torch::empty({m, k/meta_ratio}, options_meta);
    }

    // Get grid size and block size
    dim3 grid;
    grid.x = m;
    grid.z = batch_size;

    dim3 block;
    block.x = 128;

    // Launch kernels
    // if (dense_matrix.dtype() == torch::kBFloat16){
    Dense2Sparse<nv_bfloat16><<<grid, block>>>(
        m, k, 
        (nv_bfloat16*)dense_matrix.data_ptr(), 
        (nv_bfloat16*)sparse_matrix.data_ptr(), 
        (nv_bfloat16*)uncompressed_matrix.data_ptr(), 
        metadata.data<int16_t>(), metadata_reorder.data<int16_t>());        
    // }
    // else{
    //     Dense2Sparse<float><<<grid, block>>>(
    //         m, k, dense_matrix.data<float>(), sparse_matrix.data<float>(), 
    //         uncompressed_matrix.data<float>(), metadata.data<int16_t>(), metadata_reorder.data<int16_t>());
    // }

    return {sparse_matrix, uncompressed_matrix, metadata_reorder};
}
