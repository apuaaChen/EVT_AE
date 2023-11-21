-------------------------------------------------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  
                                                   Name    Self CPU %      Self CPU   CPU total %     CPU total  CPU time avg     Self CUDA   Self CUDA %    CUDA total  CUDA time avg    # of Calls  
-------------------------------------------------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  
void at::native::vectorized_elementwise_kernel<4, at...         0.00%       0.000us         0.00%       0.000us       0.000us     197.422ms        25.74%     197.422ms       1.234ms           160  
void at::native::elementwise_kernel<128, 4, at::nati...         0.00%       0.000us         0.00%       0.000us       0.000us     114.129ms        14.88%     114.129ms       1.902ms            60  
void at::native::vectorized_elementwise_kernel<4, at...         0.00%       0.000us         0.00%       0.000us       0.000us      59.192ms         7.72%      59.192ms       1.973ms            30  
void at::native::vectorized_elementwise_kernel<4, at...         0.00%       0.000us         0.00%       0.000us       0.000us      39.642ms         5.17%      39.642ms       1.982ms            20  
sm80_xmma_gemm_f16f16_f16f32_f32_tn_n_tilesize160x12...         0.00%       0.000us         0.00%       0.000us       0.000us      32.989ms         4.30%      32.989ms       3.299ms            10  
void at::native::elementwise_kernel<128, 4, at::nati...         0.00%       0.000us         0.00%       0.000us       0.000us      31.362ms         4.09%      31.362ms       3.136ms            10  
void at::native::vectorized_elementwise_kernel<4, at...         0.00%       0.000us         0.00%       0.000us       0.000us      30.231ms         3.94%      30.231ms       3.023ms            10  
void cudnn::ops::nchwToNhwcKernel<__half, __half, fl...         0.00%       0.000us         0.00%       0.000us       0.000us      29.622ms         3.86%      29.622ms     246.850us           120  
ampere_fp16_s16816gemm_fp16_256x128_ldg8_f2f_stages_...         0.00%       0.000us         0.00%       0.000us       0.000us      28.723ms         3.74%      28.723ms       2.872ms            10  
                         Memcpy DtoD (Device -> Device)         0.00%       0.000us         0.00%       0.000us       0.000us      25.539ms         3.33%      25.539ms       1.277ms            20  
ampere_fp16_s16816gemm_fp16_128x128_ldg8_f2f_stages_...         0.00%       0.000us         0.00%       0.000us       0.000us      25.359ms         3.31%      25.359ms       2.536ms            10  
void at::native::vectorized_elementwise_kernel<4, at...         0.00%       0.000us         0.00%       0.000us       0.000us      20.606ms         2.69%      20.606ms     412.120us            50  
void at::native::vectorized_elementwise_kernel<4, at...         0.00%       0.000us         0.00%       0.000us       0.000us      20.479ms         2.67%      20.479ms       2.048ms            10  
void at::native::vectorized_elementwise_kernel<4, at...         0.00%       0.000us         0.00%       0.000us       0.000us      20.170ms         2.63%      20.170ms       2.017ms            10  
void at::native::vectorized_elementwise_kernel<4, at...         0.00%       0.000us         0.00%       0.000us       0.000us      15.656ms         2.04%      15.656ms     104.373us           150  
void at::native::reduce_kernel<512, 1, at::native::R...         0.00%       0.000us         0.00%       0.000us       0.000us      11.023ms         1.44%      11.023ms     275.575us            40  
void at::native::reduce_kernel<128, 4, at::native::R...         0.00%       0.000us         0.00%       0.000us       0.000us       9.782ms         1.28%       9.782ms     489.100us            20  
sm80_xmma_wgrad_implicit_gemm_indexed_f16f16_f16f32_...         0.00%       0.000us         0.00%       0.000us       0.000us       8.774ms         1.14%       8.774ms     877.400us            10  
sm80_xmma_fprop_implicit_gemm_indexed_f16f16_f16f32_...         0.00%       0.000us         0.00%       0.000us       0.000us       7.118ms         0.93%       7.118ms     355.900us            20  
void at::native::(anonymous namespace)::fused_dropou...         0.00%       0.000us         0.00%       0.000us       0.000us       6.730ms         0.88%       6.730ms     336.500us            20  
void at::native::(anonymous namespace)::max_pool_bac...         0.00%       0.000us         0.00%       0.000us       0.000us       5.692ms         0.74%       5.692ms     189.733us            30  
sm80_xmma_wgrad_implicit_gemm_indexed_f16f16_f16f32_...         0.00%       0.000us         0.00%       0.000us       0.000us       5.305ms         0.69%       5.305ms     530.500us            10  
void at::native::(anonymous namespace)::max_pool_for...         0.00%       0.000us         0.00%       0.000us       0.000us       4.290ms         0.56%       4.290ms     143.000us            30  
_ZN19cutlass_cudnn_train6KernelINS_4conv6kernel23Imp...         0.00%       0.000us         0.00%       0.000us       0.000us       3.450ms         0.45%       3.450ms     345.000us            10  
ampere_fp16_s16816gemm_fp16_128x64_ldg8_f2f_stages_3...         0.00%       0.000us         0.00%       0.000us       0.000us       3.192ms         0.42%       3.192ms     319.200us            10  
sm80_xmma_fprop_implicit_gemm_f16f16_f16f32_f32_nhwc...         0.00%       0.000us         0.00%       0.000us       0.000us       3.094ms         0.40%       3.094ms     309.400us            10  
void at::native::elementwise_kernel<128, 4, at::nati...         0.00%       0.000us         0.00%       0.000us       0.000us       1.408ms         0.18%       1.408ms      46.933us            30  
ampere_fp16_s16816gemm_fp16_128x128_ldg8_relu_f2f_st...         0.00%       0.000us         0.00%       0.000us       0.000us       1.215ms         0.16%       1.215ms     121.500us            10  
ampere_fp16_s16816gemm_fp16_128x128_ldg8_f2f_stages_...         0.00%       0.000us         0.00%       0.000us       0.000us       1.199ms         0.16%       1.199ms     119.900us            10  
void at::native::(anonymous namespace)::CatArrayBatc...         0.00%       0.000us         0.00%       0.000us       0.000us       1.180ms         0.15%       1.180ms     118.000us            10  
void at::native::elementwise_kernel<128, 4, at::nati...         0.00%       0.000us         0.00%       0.000us       0.000us       1.082ms         0.14%       1.082ms      36.067us            30  
void cudnn::ops::nhwcToNchwKernel<__half, __half, fl...         0.00%       0.000us         0.00%       0.000us       0.000us     789.000us         0.10%     789.000us      13.150us            60  
                                       Memset (Unknown)         0.00%       0.000us         0.00%       0.000us       0.000us     166.000us         0.02%     166.000us       1.844us            90  
                                                memset8         0.00%       0.000us         0.00%       0.000us       0.000us     106.000us         0.01%     106.000us       5.300us            20  
void at::native::vectorized_elementwise_kernel<4, at...         0.00%       0.000us         0.00%       0.000us       0.000us      86.000us         0.01%      86.000us       8.600us            10  
void cutlass_cudnn_train::Kernel<cutlass_cudnn_train...         0.00%       0.000us         0.00%       0.000us       0.000us      70.000us         0.01%      70.000us       7.000us            10  
void at::native::vectorized_elementwise_kernel<4, at...         0.00%       0.000us         0.00%       0.000us       0.000us      44.000us         0.01%      44.000us       4.400us            10  
void at::native::vectorized_elementwise_kernel<4, at...         0.00%       0.000us         0.00%       0.000us       0.000us      23.000us         0.00%      23.000us       1.150us            20  
void at::native::vectorized_elementwise_kernel<4, at...         0.00%       0.000us         0.00%       0.000us       0.000us      20.000us         0.00%      20.000us       1.000us            20  
                                               memset32         0.00%       0.000us         0.00%       0.000us       0.000us      20.000us         0.00%      20.000us       2.000us            10  
                                       cudaLaunchKernel         0.08%     586.000us         0.08%     586.000us       4.883us       0.000us         0.00%       0.000us       0.000us           120  
                                        cudaMemcpyAsync         0.03%     195.000us         0.03%     195.000us       9.750us       0.000us         0.00%       0.000us       0.000us            20  
                                  cudaStreamIsCapturing         0.00%       2.000us         0.00%       2.000us       0.100us       0.000us         0.00%       0.000us       0.000us            20  
                                        cudaGraphLaunch         0.18%       1.412ms         0.18%       1.412ms     141.200us       0.000us         0.00%       0.000us       0.000us            10  
                                   cudaDriverGetVersion         0.00%       0.000us         0.00%       0.000us       0.000us       0.000us         0.00%       0.000us       0.000us            10  
                                  cudaDeviceSynchronize        99.71%     763.869ms        99.71%     763.869ms     763.869ms       0.000us         0.00%       0.000us       0.000us             1  
-------------------------------------------------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  
Self CPU time total: 766.064ms
Self CUDA time total: 766.979ms