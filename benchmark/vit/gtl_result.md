-------------------------------------------------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  
                                                   Name    Self CPU %      Self CPU   CPU total %     CPU total  CPU time avg     Self CUDA   Self CUDA %    CUDA total  CUDA time avg    # of Calls  
-------------------------------------------------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  
cutlass_tensorop_f16_s16816gemm_f16_128x128_32x5_tt_...         0.00%       0.000us         0.00%       0.000us       0.000us     175.597ms        11.48%     175.597ms     487.769us           360  
cutlass_tensorop_f16_s16816gemm_f16_128x128_32x5_tn_...         0.00%       0.000us         0.00%       0.000us       0.000us     165.086ms        10.79%     165.086ms     687.858us           240  
cutlass_tensorop_f16_s16816gemm_f16_128x128_32x5_nt_...         0.00%       0.000us         0.00%       0.000us       0.000us     136.877ms         8.95%     136.877ms     228.128us           600  
cutlass_tensorop_f16_s16816gemm_f16_128x128_32x4_tt_...         0.00%       0.000us         0.00%       0.000us       0.000us     127.551ms         8.34%     127.551ms     531.462us           240  
void at::native::elementwise_kernel<128, 4, at::nati...         0.00%       0.000us         0.00%       0.000us       0.000us     124.657ms         8.15%     124.657ms     445.204us           280  
cutlass_tensorop_f16_s16816gemm_f16_128x64_32x6_nt_a...         0.00%       0.000us         0.00%       0.000us       0.000us     120.193ms         7.86%     120.193ms     500.804us           240  
cutlass_tensorop_f16_s16816gemm_f16_128x128_32x3_tt_...         0.00%       0.000us         0.00%       0.000us       0.000us     102.229ms         6.68%     102.229ms     283.969us           360  
cutlass_tensorop_f16_s16816gemm_f16_256x128_32x3_tn_...         0.00%       0.000us         0.00%       0.000us       0.000us      80.878ms         5.29%      80.878ms     673.983us           120  
                                         softmax_kernel         0.00%       0.000us         0.00%       0.000us       0.000us      75.376ms         4.93%      75.376ms     579.815us           130  
cutlass_tensorop_f16_s16816gemm_f16_256x128_32x3_nt_...         0.00%       0.000us         0.00%       0.000us       0.000us      65.599ms         4.29%      65.599ms     546.658us           120  
cutlass_tensorop_f16_s16816gemm_f16_128x128_32x3_tn_...         0.00%       0.000us         0.00%       0.000us       0.000us      63.988ms         4.18%      63.988ms     177.744us           360  
cutlass_tensorop_f16_s16816gemm_f16_128x64_64x3_tt_a...         0.00%       0.000us         0.00%       0.000us       0.000us      60.143ms         3.93%      60.143ms     501.192us           120  
                                softmax_backward_kernel         0.00%       0.000us         0.00%       0.000us       0.000us      56.012ms         3.66%      56.012ms     466.767us           120  
                              layernorm_backward_kernel         0.00%       0.000us         0.00%       0.000us       0.000us      48.365ms         3.16%      48.365ms     193.460us           250  
cutlass_tensorop_f16_s16816gemm_f16_128x64_32x6_tt_a...         0.00%       0.000us         0.00%       0.000us       0.000us      30.959ms         2.02%      30.959ms     257.992us           120  
void at::native::reduce_kernel<128, 4, at::native::R...         0.00%       0.000us         0.00%       0.000us       0.000us      20.080ms         1.31%      20.080ms      74.370us           270  
                                       layernorm_kernel         0.00%       0.000us         0.00%       0.000us       0.000us      17.625ms         1.15%      17.625ms      70.500us           250  
void at::native::elementwise_kernel<128, 4, at::nati...         0.00%       0.000us         0.00%       0.000us       0.000us      15.052ms         0.98%      15.052ms     107.514us           140  
void at::native::elementwise_kernel<128, 4, at::nati...         0.00%       0.000us         0.00%       0.000us       0.000us      14.879ms         0.97%      14.879ms     114.454us           130  
void at::native::vectorized_elementwise_kernel<4, at...         0.00%       0.000us         0.00%       0.000us       0.000us       5.999ms         0.39%       5.999ms       3.015us          1990  
void at::native::vectorized_elementwise_kernel<4, at...         0.00%       0.000us         0.00%       0.000us       0.000us       4.214ms         0.28%       4.214ms       1.538us          2740  
void at::native::unrolled_elementwise_kernel<at::nat...         0.00%       0.000us         0.00%       0.000us       0.000us       3.133ms         0.20%       3.133ms     156.650us            20  
void at::native::vectorized_elementwise_kernel<4, at...         0.00%       0.000us         0.00%       0.000us       0.000us       2.181ms         0.14%       2.181ms       1.000us          2180  
                                              memcpy128         0.00%       0.000us         0.00%       0.000us       0.000us       2.101ms         0.14%       2.101ms      70.033us            30  
cutlass_tensorop_f16_s16816gemm_f16_256x128_32x3_tn_...         0.00%       0.000us         0.00%       0.000us       0.000us       1.949ms         0.13%       1.949ms     194.900us            10  
void at::native::(anonymous namespace)::CatArrayBatc...         0.00%       0.000us         0.00%       0.000us       0.000us       1.574ms         0.10%       1.574ms     157.400us            10  
void at::native::vectorized_elementwise_kernel<4, at...         0.00%       0.000us         0.00%       0.000us       0.000us       1.499ms         0.10%       1.499ms      37.475us            40  
cutlass_tensorop_f16_s16816gemm_f16_128x128_32x3_nt_...         0.00%       0.000us         0.00%       0.000us       0.000us       1.416ms         0.09%       1.416ms     141.600us            10  
void at::native::unrolled_elementwise_kernel<at::nat...         0.00%       0.000us         0.00%       0.000us       0.000us     822.000us         0.05%     822.000us       3.288us           250  
void at::native::(anonymous namespace)::distribution...         0.00%       0.000us         0.00%       0.000us       0.000us     744.000us         0.05%     744.000us      74.400us            10  
                         Memcpy DtoD (Device -> Device)         0.00%       0.000us         0.00%       0.000us       0.000us     642.000us         0.04%     642.000us      32.100us            20  
void at::native::vectorized_elementwise_kernel<4, at...         0.00%       0.000us         0.00%       0.000us       0.000us     620.000us         0.04%     620.000us      62.000us            10  
void at::native::reduce_kernel<128, 4, at::native::R...         0.00%       0.000us         0.00%       0.000us       0.000us     498.000us         0.03%     498.000us      49.800us            10  
                                       Memset (Unknown)         0.00%       0.000us         0.00%       0.000us       0.000us     430.000us         0.03%     430.000us       1.720us           250  
void at::native::vectorized_elementwise_kernel<4, at...         0.00%       0.000us         0.00%       0.000us       0.000us     250.000us         0.02%     250.000us       1.000us           250  
void at::native::vectorized_elementwise_kernel<4, at...         0.00%       0.000us         0.00%       0.000us       0.000us     240.000us         0.02%     240.000us       2.000us           120  
cutlass_tensorop_f16_s16816gemm_f16_64x64_64x5_tt_al...         0.00%       0.000us         0.00%       0.000us       0.000us     100.000us         0.01%     100.000us      10.000us            10  
cutlass_tensorop_f16_s16816gemm_f16_64x64_64x5_tn_al...         0.00%       0.000us         0.00%       0.000us       0.000us      70.000us         0.00%      70.000us       7.000us            10  
cutlass_tensorop_f16_s16816gemm_f16_64x128_32x6_nt_a...         0.00%       0.000us         0.00%       0.000us       0.000us      50.000us         0.00%      50.000us       5.000us            10  
void at::native::vectorized_elementwise_kernel<4, at...         0.00%       0.000us         0.00%       0.000us       0.000us      20.000us         0.00%      20.000us       1.000us            20  
                                       cudaLaunchKernel        58.63%     890.489ms        58.63%     890.489ms     440.836us       0.000us         0.00%       0.000us       0.000us          2020  
                                        cudaMemcpyAsync         0.01%     179.000us         0.01%     179.000us       8.950us       0.000us         0.00%       0.000us       0.000us            20  
                                  cudaStreamIsCapturing         0.00%       2.000us         0.00%       2.000us       0.100us       0.000us         0.00%       0.000us       0.000us            20  
                                        cudaGraphLaunch         0.91%      13.799ms         0.91%      13.799ms       1.380ms       0.000us         0.00%       0.000us       0.000us            10  
                                   cudaDriverGetVersion         0.00%       0.000us         0.00%       0.000us       0.000us       0.000us         0.00%       0.000us       0.000us            10  
                                  cudaDeviceSynchronize        40.45%     614.363ms        40.45%     614.363ms     614.363ms       0.000us         0.00%       0.000us       0.000us             1  
-------------------------------------------------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  
Self CPU time total: 1.519s
Self CUDA time total: 1.530s