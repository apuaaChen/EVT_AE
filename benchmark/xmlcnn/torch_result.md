-------------------------------------------------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  
                                                   Name    Self CPU %      Self CPU   CPU total %     CPU total  CPU time avg     Self CUDA   Self CUDA %    CUDA total  CUDA time avg    # of Calls  
-------------------------------------------------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  
void at::native::vectorized_elementwise_kernel<4, at...         0.00%       0.000us         0.00%       0.000us       0.000us     197.418ms        25.69%     197.418ms       1.234ms           160  
void at::native::elementwise_kernel<128, 4, at::nati...         0.00%       0.000us         0.00%       0.000us       0.000us     114.672ms        14.92%     114.672ms       1.911ms            60  
void at::native::vectorized_elementwise_kernel<4, at...         0.00%       0.000us         0.00%       0.000us       0.000us      59.204ms         7.70%      59.204ms       1.973ms            30  
void at::native::vectorized_elementwise_kernel<4, at...         0.00%       0.000us         0.00%       0.000us       0.000us      39.647ms         5.16%      39.647ms       1.982ms            20  
sm80_xmma_gemm_f16f16_f16f32_f32_tn_n_tilesize160x12...         0.00%       0.000us         0.00%       0.000us       0.000us      33.190ms         4.32%      33.190ms       3.319ms            10  
void at::native::elementwise_kernel<128, 4, at::nati...         0.00%       0.000us         0.00%       0.000us       0.000us      31.403ms         4.09%      31.403ms       3.140ms            10  
void at::native::vectorized_elementwise_kernel<4, at...         0.00%       0.000us         0.00%       0.000us       0.000us      30.259ms         3.94%      30.259ms       3.026ms            10  
void cudnn::ops::nchwToNhwcKernel<__half, __half, fl...         0.00%       0.000us         0.00%       0.000us       0.000us      29.644ms         3.86%      29.644ms     247.033us           120  
ampere_fp16_s16816gemm_fp16_256x128_ldg8_f2f_stages_...         0.00%       0.000us         0.00%       0.000us       0.000us      29.089ms         3.79%      29.089ms       2.909ms            10  
                         Memcpy DtoD (Device -> Device)         0.00%       0.000us         0.00%       0.000us       0.000us      25.536ms         3.32%      25.536ms       1.277ms            20  
ampere_fp16_s16816gemm_fp16_128x128_ldg8_f2f_stages_...         0.00%       0.000us         0.00%       0.000us       0.000us      25.509ms         3.32%      25.509ms       2.551ms            10  
void at::native::vectorized_elementwise_kernel<4, at...         0.00%       0.000us         0.00%       0.000us       0.000us      20.608ms         2.68%      20.608ms     412.160us            50  
void at::native::vectorized_elementwise_kernel<4, at...         0.00%       0.000us         0.00%       0.000us       0.000us      20.486ms         2.67%      20.486ms       2.049ms            10  
void at::native::vectorized_elementwise_kernel<4, at...         0.00%       0.000us         0.00%       0.000us       0.000us      20.178ms         2.63%      20.178ms       2.018ms            10  
void at::native::vectorized_elementwise_kernel<4, at...         0.00%       0.000us         0.00%       0.000us       0.000us      15.671ms         2.04%      15.671ms     104.473us           150  
void at::native::reduce_kernel<512, 1, at::native::R...         0.00%       0.000us         0.00%       0.000us       0.000us      11.023ms         1.43%      11.023ms     275.575us            40  
void at::native::reduce_kernel<128, 4, at::native::R...         0.00%       0.000us         0.00%       0.000us       0.000us       9.775ms         1.27%       9.775ms     488.750us            20  
sm80_xmma_wgrad_implicit_gemm_indexed_f16f16_f16f32_...         0.00%       0.000us         0.00%       0.000us       0.000us       8.807ms         1.15%       8.807ms     880.700us            10  
sm80_xmma_fprop_implicit_gemm_indexed_f16f16_f16f32_...         0.00%       0.000us         0.00%       0.000us       0.000us       7.130ms         0.93%       7.130ms     356.500us            20  
void at::native::(anonymous namespace)::fused_dropou...         0.00%       0.000us         0.00%       0.000us       0.000us       6.738ms         0.88%       6.738ms     336.900us            20  
void at::native::(anonymous namespace)::max_pool_bac...         0.00%       0.000us         0.00%       0.000us       0.000us       5.568ms         0.72%       5.568ms     185.600us            30  
sm80_xmma_wgrad_implicit_gemm_indexed_f16f16_f16f32_...         0.00%       0.000us         0.00%       0.000us       0.000us       5.329ms         0.69%       5.329ms     532.900us            10  
void at::native::(anonymous namespace)::max_pool_for...         0.00%       0.000us         0.00%       0.000us       0.000us       4.319ms         0.56%       4.319ms     143.967us            30  
_ZN19cutlass_cudnn_train6KernelINS_4conv6kernel23Imp...         0.00%       0.000us         0.00%       0.000us       0.000us       3.463ms         0.45%       3.463ms     346.300us            10  
ampere_fp16_s16816gemm_fp16_128x64_ldg8_f2f_stages_3...         0.00%       0.000us         0.00%       0.000us       0.000us       3.196ms         0.42%       3.196ms     319.600us            10  
sm80_xmma_fprop_implicit_gemm_f16f16_f16f32_f32_nhwc...         0.00%       0.000us         0.00%       0.000us       0.000us       3.094ms         0.40%       3.094ms     309.400us            10  
void at::native::elementwise_kernel<128, 4, at::nati...         0.00%       0.000us         0.00%       0.000us       0.000us       1.407ms         0.18%       1.407ms      46.900us            30  
ampere_fp16_s16816gemm_fp16_128x128_ldg8_relu_f2f_st...         0.00%       0.000us         0.00%       0.000us       0.000us       1.216ms         0.16%       1.216ms     121.600us            10  
ampere_fp16_s16816gemm_fp16_128x128_ldg8_f2f_stages_...         0.00%       0.000us         0.00%       0.000us       0.000us       1.205ms         0.16%       1.205ms     120.500us            10  
void at::native::(anonymous namespace)::CatArrayBatc...         0.00%       0.000us         0.00%       0.000us       0.000us       1.191ms         0.15%       1.191ms     119.100us            10  
void at::native::elementwise_kernel<128, 4, at::nati...         0.00%       0.000us         0.00%       0.000us       0.000us       1.081ms         0.14%       1.081ms      36.033us            30  
void cudnn::ops::nhwcToNchwKernel<__half, __half, fl...         0.00%       0.000us         0.00%       0.000us       0.000us     815.000us         0.11%     815.000us      13.583us            60  
                                       Memset (Unknown)         0.00%       0.000us         0.00%       0.000us       0.000us     171.000us         0.02%     171.000us       1.900us            90  
                                                memset8         0.00%       0.000us         0.00%       0.000us       0.000us     110.000us         0.01%     110.000us       5.500us            20  
void at::native::vectorized_elementwise_kernel<4, at...         0.00%       0.000us         0.00%       0.000us       0.000us      88.000us         0.01%      88.000us       8.800us            10  
void cutlass_cudnn_train::Kernel<cutlass_cudnn_train...         0.00%       0.000us         0.00%       0.000us       0.000us      70.000us         0.01%      70.000us       7.000us            10  
void at::native::vectorized_elementwise_kernel<4, at...         0.00%       0.000us         0.00%       0.000us       0.000us      44.000us         0.01%      44.000us       4.400us            10  
void at::native::vectorized_elementwise_kernel<4, at...         0.00%       0.000us         0.00%       0.000us       0.000us      22.000us         0.00%      22.000us       1.100us            20  
void at::native::vectorized_elementwise_kernel<4, at...         0.00%       0.000us         0.00%       0.000us       0.000us      20.000us         0.00%      20.000us       1.000us            20  
                                               memset32         0.00%       0.000us         0.00%       0.000us       0.000us      20.000us         0.00%      20.000us       2.000us            10  
                                       cudaLaunchKernel         0.08%     587.000us         0.08%     587.000us       4.892us       0.000us         0.00%       0.000us       0.000us           120  
                                        cudaMemcpyAsync         0.03%     205.000us         0.03%     205.000us      10.250us       0.000us         0.00%       0.000us       0.000us            20  
                                  cudaStreamIsCapturing         0.00%       2.000us         0.00%       2.000us       0.100us       0.000us         0.00%       0.000us       0.000us            20  
                                        cudaGraphLaunch         0.19%       1.460ms         0.19%       1.460ms     146.000us       0.000us         0.00%       0.000us       0.000us            10  
                                   cudaDriverGetVersion         0.00%       0.000us         0.00%       0.000us       0.000us       0.000us         0.00%       0.000us       0.000us            10  
                                  cudaDeviceSynchronize        99.71%     765.183ms        99.71%     765.183ms     765.183ms       0.000us         0.00%       0.000us       0.000us             1  
-------------------------------------------------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  
Self CPU time total: 767.437ms
Self CUDA time total: 768.416ms