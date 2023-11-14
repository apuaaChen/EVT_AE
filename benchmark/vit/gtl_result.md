-------------------------------------------------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  
                                                   Name    Self CPU %      Self CPU   CPU total %     CPU total  CPU time avg     Self CUDA   Self CUDA %    CUDA total  CUDA time avg    # of Calls  
-------------------------------------------------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  
cutlass_tensorop_f16_s16816gemm_f16_128x128_32x5_tt_...         0.00%       0.000us         0.00%       0.000us       0.000us     176.062ms        11.49%     176.062ms     489.061us           360  
cutlass_tensorop_f16_s16816gemm_f16_128x128_32x5_tn_...         0.00%       0.000us         0.00%       0.000us       0.000us     166.053ms        10.84%     166.053ms     691.888us           240  
cutlass_tensorop_f16_s16816gemm_f16_128x128_32x5_nt_...         0.00%       0.000us         0.00%       0.000us       0.000us     136.925ms         8.94%     136.925ms     228.208us           600  
cutlass_tensorop_f16_s16816gemm_f16_128x128_32x4_tt_...         0.00%       0.000us         0.00%       0.000us       0.000us     127.379ms         8.32%     127.379ms     530.746us           240  
void at::native::elementwise_kernel<128, 4, at::nati...         0.00%       0.000us         0.00%       0.000us       0.000us     124.911ms         8.16%     124.911ms     446.111us           280  
cutlass_tensorop_f16_s16816gemm_f16_128x64_32x6_nt_a...         0.00%       0.000us         0.00%       0.000us       0.000us     120.496ms         7.87%     120.496ms     502.067us           240  
cutlass_tensorop_f16_s16816gemm_f16_128x128_32x3_tt_...         0.00%       0.000us         0.00%       0.000us       0.000us     102.170ms         6.67%     102.170ms     283.806us           360  
cutlass_tensorop_f16_s16816gemm_f16_256x128_32x3_tn_...         0.00%       0.000us         0.00%       0.000us       0.000us      81.074ms         5.29%      81.074ms     675.617us           120  
                                         softmax_kernel         0.00%       0.000us         0.00%       0.000us       0.000us      75.397ms         4.92%      75.397ms     579.977us           130  
cutlass_tensorop_f16_s16816gemm_f16_256x128_32x3_nt_...         0.00%       0.000us         0.00%       0.000us       0.000us      65.614ms         4.28%      65.614ms     546.783us           120  
cutlass_tensorop_f16_s16816gemm_f16_128x128_32x3_tn_...         0.00%       0.000us         0.00%       0.000us       0.000us      64.233ms         4.19%      64.233ms     178.425us           360  
cutlass_tensorop_f16_s16816gemm_f16_128x64_64x3_tt_a...         0.00%       0.000us         0.00%       0.000us       0.000us      60.462ms         3.95%      60.462ms     503.850us           120  
                                softmax_backward_kernel         0.00%       0.000us         0.00%       0.000us       0.000us      56.034ms         3.66%      56.034ms     466.950us           120  
                              layernorm_backward_kernel         0.00%       0.000us         0.00%       0.000us       0.000us      48.364ms         3.16%      48.364ms     193.456us           250  
cutlass_tensorop_f16_s16816gemm_f16_128x64_32x6_tt_a...         0.00%       0.000us         0.00%       0.000us       0.000us      31.004ms         2.02%      31.004ms     258.367us           120  
void at::native::reduce_kernel<128, 4, at::native::R...         0.00%       0.000us         0.00%       0.000us       0.000us      20.066ms         1.31%      20.066ms      74.319us           270  
                                       layernorm_kernel         0.00%       0.000us         0.00%       0.000us       0.000us      17.647ms         1.15%      17.647ms      70.588us           250  
void at::native::elementwise_kernel<128, 4, at::nati...         0.00%       0.000us         0.00%       0.000us       0.000us      15.089ms         0.99%      15.089ms     107.779us           140  
void at::native::elementwise_kernel<128, 4, at::nati...         0.00%       0.000us         0.00%       0.000us       0.000us      14.919ms         0.97%      14.919ms     114.762us           130  
void at::native::vectorized_elementwise_kernel<4, at...         0.00%       0.000us         0.00%       0.000us       0.000us       6.015ms         0.39%       6.015ms       3.023us          1990  
void at::native::vectorized_elementwise_kernel<4, at...         0.00%       0.000us         0.00%       0.000us       0.000us       4.221ms         0.28%       4.221ms       1.541us          2740  
void at::native::unrolled_elementwise_kernel<at::nat...         0.00%       0.000us         0.00%       0.000us       0.000us       3.138ms         0.20%       3.138ms     156.900us            20  
void at::native::vectorized_elementwise_kernel<4, at...         0.00%       0.000us         0.00%       0.000us       0.000us       2.180ms         0.14%       2.180ms       1.000us          2180  
cutlass_tensorop_f16_s16816gemm_f16_256x128_32x3_tn_...         0.00%       0.000us         0.00%       0.000us       0.000us       1.962ms         0.13%       1.962ms     196.200us            10  
void at::native::(anonymous namespace)::CatArrayBatc...         0.00%       0.000us         0.00%       0.000us       0.000us       1.578ms         0.10%       1.578ms     157.800us            10  
void at::native::vectorized_elementwise_kernel<4, at...         0.00%       0.000us         0.00%       0.000us       0.000us       1.454ms         0.09%       1.454ms      36.350us            40  
cutlass_tensorop_f16_s16816gemm_f16_128x128_32x3_nt_...         0.00%       0.000us         0.00%       0.000us       0.000us       1.422ms         0.09%       1.422ms     142.200us            10  
                                              memcpy128         0.00%       0.000us         0.00%       0.000us       0.000us       1.337ms         0.09%       1.337ms      66.850us            20  
void at::native::unrolled_elementwise_kernel<at::nat...         0.00%       0.000us         0.00%       0.000us       0.000us     824.000us         0.05%     824.000us       3.296us           250  
void at::native::(anonymous namespace)::distribution...         0.00%       0.000us         0.00%       0.000us       0.000us     747.000us         0.05%     747.000us      74.700us            10  
                         Memcpy DtoD (Device -> Device)         0.00%       0.000us         0.00%       0.000us       0.000us     644.000us         0.04%     644.000us      32.200us            20  
void at::native::vectorized_elementwise_kernel<4, at...         0.00%       0.000us         0.00%       0.000us       0.000us     621.000us         0.04%     621.000us      62.100us            10  
void at::native::reduce_kernel<128, 4, at::native::R...         0.00%       0.000us         0.00%       0.000us       0.000us     491.000us         0.03%     491.000us      49.100us            10  
                                       Memset (Unknown)         0.00%       0.000us         0.00%       0.000us       0.000us     442.000us         0.03%     442.000us       1.768us           250  
void at::native::vectorized_elementwise_kernel<4, at...         0.00%       0.000us         0.00%       0.000us       0.000us     250.000us         0.02%     250.000us       1.000us           250  
void at::native::vectorized_elementwise_kernel<4, at...         0.00%       0.000us         0.00%       0.000us       0.000us     240.000us         0.02%     240.000us       2.000us           120  
cutlass_tensorop_f16_s16816gemm_f16_64x64_64x5_tt_al...         0.00%       0.000us         0.00%       0.000us       0.000us     100.000us         0.01%     100.000us      10.000us            10  
cutlass_tensorop_f16_s16816gemm_f16_64x64_64x5_tn_al...         0.00%       0.000us         0.00%       0.000us       0.000us      70.000us         0.00%      70.000us       7.000us            10  
cutlass_tensorop_f16_s16816gemm_f16_64x128_32x6_nt_a...         0.00%       0.000us         0.00%       0.000us       0.000us      50.000us         0.00%      50.000us       5.000us            10  
void at::native::vectorized_elementwise_kernel<4, at...         0.00%       0.000us         0.00%       0.000us       0.000us      20.000us         0.00%      20.000us       1.000us            20  
                                       cudaLaunchKernel        58.95%     896.853ms        58.95%     896.853ms     443.987us       0.000us         0.00%       0.000us       0.000us          2020  
                                        cudaMemcpyAsync         0.01%     164.000us         0.01%     164.000us       8.200us       0.000us         0.00%       0.000us       0.000us            20  
                                  cudaStreamIsCapturing         0.00%       2.000us         0.00%       2.000us       0.100us       0.000us         0.00%       0.000us       0.000us            20  
                                        cudaGraphLaunch         0.67%      10.247ms         0.67%      10.247ms       1.025ms       0.000us         0.00%       0.000us       0.000us            10  
                                   cudaDriverGetVersion         0.00%       0.000us         0.00%       0.000us       0.000us       0.000us         0.00%       0.000us       0.000us            10  
                                  cudaDeviceSynchronize        40.36%     614.093ms        40.36%     614.093ms     614.093ms       0.000us         0.00%       0.000us       0.000us             1  
-------------------------------------------------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  
Self CPU time total: 1.521s
Self CUDA time total: 1.532s