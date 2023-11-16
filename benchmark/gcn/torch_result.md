-------------------------------------------------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  
                                                   Name    Self CPU %      Self CPU   CPU total %     CPU total  CPU time avg     Self CUDA   Self CUDA %    CUDA total  CUDA time avg    # of Calls  
-------------------------------------------------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  
void at::native::(anonymous namespace)::nll_loss_for...         0.00%       0.000us         0.00%       0.000us       0.000us     248.703ms        41.56%     248.703ms      24.870ms            10  
void at::native::(anonymous namespace)::nll_loss_bac...         0.00%       0.000us         0.00%       0.000us       0.000us     128.858ms        21.53%     128.858ms      12.886ms            10  
void cusparse::csrmm_alg2_kernel<cusparse::CsrMMPoli...         0.00%       0.000us         0.00%       0.000us       0.000us      71.653ms        11.97%      71.653ms       1.194ms            60  
void (anonymous namespace)::softmax_warp_backward<fl...         0.00%       0.000us         0.00%       0.000us       0.000us      22.753ms         3.80%      22.753ms       2.275ms            10  
void (anonymous namespace)::softmax_warp_forward<flo...         0.00%       0.000us         0.00%       0.000us       0.000us      14.982ms         2.50%      14.982ms       1.498ms            10  
void at::native::unrolled_elementwise_kernel<at::nat...         0.00%       0.000us         0.00%       0.000us       0.000us      13.441ms         2.25%      13.441ms       1.344ms            10  
void at::native::unrolled_elementwise_kernel<at::nat...         0.00%       0.000us         0.00%       0.000us       0.000us      13.284ms         2.22%      13.284ms       1.328ms            10  
void at::native::vectorized_elementwise_kernel<4, at...         0.00%       0.000us         0.00%       0.000us       0.000us      12.404ms         2.07%      12.404ms     103.367us           120  
                                          memcpy32_post         0.00%       0.000us         0.00%       0.000us       0.000us       8.615ms         1.44%       8.615ms     143.583us            60  
void at::native::reduce_kernel<128, 4, at::native::R...         0.00%       0.000us         0.00%       0.000us       0.000us       7.863ms         1.31%       7.863ms     262.100us            30  
void at::native::vectorized_elementwise_kernel<4, at...         0.00%       0.000us         0.00%       0.000us       0.000us       6.631ms         1.11%       6.631ms     331.550us            20  
ampere_fp16_s1688gemm_fp16_128x128_ldg8_relu_f2f_sta...         0.00%       0.000us         0.00%       0.000us       0.000us       6.573ms         1.10%       6.573ms     657.300us            10  
void cutlass::Kernel<cutlass_80_tensorop_f16_s16816g...         0.00%       0.000us         0.00%       0.000us       0.000us       6.187ms         1.03%       6.187ms     309.350us            20  
ampere_fp16_s16816gemm_fp16_64x128_ldg8_f2f_stages_6...         0.00%       0.000us         0.00%       0.000us       0.000us       4.629ms         0.77%       4.629ms     462.900us            10  
void cusparse::matrix_scalar_multiply_kernel<cuspars...         0.00%       0.000us         0.00%       0.000us       0.000us       4.625ms         0.77%       4.625ms      77.083us            60  
void at::native::vectorized_elementwise_kernel<4, at...         0.00%       0.000us         0.00%       0.000us       0.000us       4.222ms         0.71%       4.222ms      35.183us           120  
void at::native::vectorized_elementwise_kernel<4, at...         0.00%       0.000us         0.00%       0.000us       0.000us       4.197ms         0.70%       4.197ms     209.850us            20  
void at::native::(anonymous namespace)::fused_dropou...         0.00%       0.000us         0.00%       0.000us       0.000us       4.039ms         0.67%       4.039ms     201.950us            20  
void cutlass::Kernel<cutlass_80_tensorop_f16_s16816g...         0.00%       0.000us         0.00%       0.000us       0.000us       3.592ms         0.60%       3.592ms     179.600us            20  
void at::native::vectorized_elementwise_kernel<4, at...         0.00%       0.000us         0.00%       0.000us       0.000us       3.348ms         0.56%       3.348ms     167.400us            20  
void at::native::vectorized_elementwise_kernel<4, at...         0.00%       0.000us         0.00%       0.000us       0.000us       2.689ms         0.45%       2.689ms     134.450us            20  
ampere_fp16_s16816gemm_fp16_128x64_ldg8_f2f_stages_6...         0.00%       0.000us         0.00%       0.000us       0.000us       2.133ms         0.36%       2.133ms     213.300us            10  
ampere_fp16_s16816gemm_fp16_64x64_sliced1x2_ldg8_f2f...         0.00%       0.000us         0.00%       0.000us       0.000us       1.515ms         0.25%       1.515ms     151.500us            10  
void cusparse::csrmm_alg2_partition_kernel<128, long...         0.00%       0.000us         0.00%       0.000us       0.000us       1.205ms         0.20%       1.205ms      20.083us            60  
void splitKreduce_kernel<32, 16, int, __half, __half...         0.00%       0.000us         0.00%       0.000us       0.000us     162.000us         0.03%     162.000us       5.400us            30  
                                       Memset (Unknown)         0.00%       0.000us         0.00%       0.000us       0.000us      41.000us         0.01%      41.000us       1.367us            30  
void at::native::vectorized_elementwise_kernel<4, at...         0.00%       0.000us         0.00%       0.000us       0.000us      22.000us         0.00%      22.000us       1.100us            20  
void at::native::vectorized_elementwise_kernel<4, at...         0.00%       0.000us         0.00%       0.000us       0.000us      20.000us         0.00%      20.000us       1.000us            20  
                                       cudaLaunchKernel         0.07%     403.000us         0.07%     403.000us       5.037us       0.000us         0.00%       0.000us       0.000us            80  
                                  cudaStreamIsCapturing         0.00%       2.000us         0.00%       2.000us       0.100us       0.000us         0.00%       0.000us       0.000us            20  
                                        cudaGraphLaunch         0.16%     983.000us         0.16%     983.000us      98.300us       0.000us         0.00%       0.000us       0.000us            10  
                                   cudaDriverGetVersion         0.00%       0.000us         0.00%       0.000us       0.000us       0.000us         0.00%       0.000us       0.000us            10  
                                  cudaDeviceSynchronize        99.77%     596.109ms        99.77%     596.109ms     596.109ms       0.000us         0.00%       0.000us       0.000us             1  
-------------------------------------------------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  
Self CPU time total: 597.497ms
Self CUDA time total: 598.386ms