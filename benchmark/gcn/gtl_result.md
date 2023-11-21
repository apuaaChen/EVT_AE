-------------------------------------------------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  
                                                   Name    Self CPU %      Self CPU   CPU total %     CPU total  CPU time avg     Self CUDA   Self CUDA %    CUDA total  CUDA time avg    # of Calls  
-------------------------------------------------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  
                                            spmm_kernel         0.00%       0.000us         0.00%       0.000us       0.000us      70.454ms        62.82%      70.454ms       1.174ms            60  
cutlass_tensorop_f16_s16816gemm_f16_64x64_64x5_nt_al...         0.00%       0.000us         0.00%       0.000us       0.000us       8.682ms         7.74%       8.682ms     289.400us            30  
                                         softmax_kernel         0.00%       0.000us         0.00%       0.000us       0.000us       8.585ms         7.65%       8.585ms     858.500us            10  
void at::native::reduce_kernel<128, 4, at::native::R...         0.00%       0.000us         0.00%       0.000us       0.000us       7.827ms         6.98%       7.827ms     260.900us            30  
cutlass_tensorop_f16_s16816gemm_f16_128x128_32x3_tn_...         0.00%       0.000us         0.00%       0.000us       0.000us       4.703ms         4.19%       4.703ms     470.300us            10  
cutlass_tensorop_f16_s16816gemm_f16_64x64_64x5_tt_al...         0.00%       0.000us         0.00%       0.000us       0.000us       4.574ms         4.08%       4.574ms     457.400us            10  
cutlass_tensorop_f16_s16816gemm_f16_256x64_32x3_tt_a...         0.00%       0.000us         0.00%       0.000us       0.000us       2.366ms         2.11%       2.366ms     236.600us            10  
cutlass_tensorop_f16_s16816gemm_f16_256x64_64x3_tn_a...         0.00%       0.000us         0.00%       0.000us       0.000us       2.076ms         1.85%       2.076ms     207.600us            10  
cutlass_tensorop_f16_s16816gemm_f16_256x64_32x3_tn_a...         0.00%       0.000us         0.00%       0.000us       0.000us       1.330ms         1.19%       1.330ms     133.000us            10  
void at::native::vectorized_elementwise_kernel<4, at...         0.00%       0.000us         0.00%       0.000us       0.000us       1.306ms         1.16%       1.306ms      43.533us            30  
void at::native::vectorized_elementwise_kernel<4, at...         0.00%       0.000us         0.00%       0.000us       0.000us      79.000us         0.07%      79.000us       1.129us            70  
void at::native::vectorized_elementwise_kernel<4, at...         0.00%       0.000us         0.00%       0.000us       0.000us      74.000us         0.07%      74.000us       1.233us            60  
                                       Memset (Unknown)         0.00%       0.000us         0.00%       0.000us       0.000us      48.000us         0.04%      48.000us       1.600us            30  
void at::native::vectorized_elementwise_kernel<4, at...         0.00%       0.000us         0.00%       0.000us       0.000us      30.000us         0.03%      30.000us       1.000us            30  
void at::native::vectorized_elementwise_kernel<4, at...         0.00%       0.000us         0.00%       0.000us       0.000us      20.000us         0.02%      20.000us       1.000us            20  
                                       cudaLaunchKernel         0.35%     389.000us         0.35%     389.000us       4.862us       0.000us         0.00%       0.000us       0.000us            80  
                                  cudaStreamIsCapturing         0.00%       2.000us         0.00%       2.000us       0.100us       0.000us         0.00%       0.000us       0.000us            20  
                                        cudaGraphLaunch         0.41%     458.000us         0.41%     458.000us      45.800us       0.000us         0.00%       0.000us       0.000us            10  
                                   cudaDriverGetVersion         0.00%       0.000us         0.00%       0.000us       0.000us       0.000us         0.00%       0.000us       0.000us            10  
                                  cudaDeviceSynchronize        99.24%     110.988ms        99.24%     110.988ms     110.988ms       0.000us         0.00%       0.000us       0.000us             1  
-------------------------------------------------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  
Self CPU time total: 111.837ms
Self CUDA time total: 112.154ms