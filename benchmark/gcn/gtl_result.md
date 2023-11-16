-------------------------------------------------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  
                                                   Name    Self CPU %      Self CPU   CPU total %     CPU total  CPU time avg     Self CUDA   Self CUDA %    CUDA total  CUDA time avg    # of Calls  
-------------------------------------------------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  
                                            spmm_kernel         0.00%       0.000us         0.00%       0.000us       0.000us      71.163ms        63.12%      71.163ms       1.186ms            60  
                                         softmax_kernel         0.00%       0.000us         0.00%       0.000us       0.000us       8.604ms         7.63%       8.604ms     860.400us            10  
cutlass_tensorop_f16_s16816gemm_f16_64x64_64x5_nt_al...         0.00%       0.000us         0.00%       0.000us       0.000us       8.587ms         7.62%       8.587ms     286.233us            30  
void at::native::reduce_kernel<128, 4, at::native::R...         0.00%       0.000us         0.00%       0.000us       0.000us       7.951ms         7.05%       7.951ms     265.033us            30  
cutlass_tensorop_f16_s16816gemm_f16_128x128_32x3_tn_...         0.00%       0.000us         0.00%       0.000us       0.000us       4.768ms         4.23%       4.768ms     476.800us            10  
cutlass_tensorop_f16_s16816gemm_f16_64x64_64x5_tt_al...         0.00%       0.000us         0.00%       0.000us       0.000us       4.577ms         4.06%       4.577ms     457.700us            10  
cutlass_tensorop_f16_s16816gemm_f16_256x64_32x3_tt_a...         0.00%       0.000us         0.00%       0.000us       0.000us       2.126ms         1.89%       2.126ms     212.600us            10  
cutlass_tensorop_f16_s16816gemm_f16_256x64_64x3_tn_a...         0.00%       0.000us         0.00%       0.000us       0.000us       2.078ms         1.84%       2.078ms     207.800us            10  
cutlass_tensorop_f16_s16816gemm_f16_256x64_32x3_tn_a...         0.00%       0.000us         0.00%       0.000us       0.000us       1.330ms         1.18%       1.330ms     133.000us            10  
void at::native::vectorized_elementwise_kernel<4, at...         0.00%       0.000us         0.00%       0.000us       0.000us       1.309ms         1.16%       1.309ms      43.633us            30  
void at::native::vectorized_elementwise_kernel<4, at...         0.00%       0.000us         0.00%       0.000us       0.000us      80.000us         0.07%      80.000us       1.143us            70  
void at::native::vectorized_elementwise_kernel<4, at...         0.00%       0.000us         0.00%       0.000us       0.000us      78.000us         0.07%      78.000us       1.300us            60  
                                       Memset (Unknown)         0.00%       0.000us         0.00%       0.000us       0.000us      48.000us         0.04%      48.000us       1.600us            30  
void at::native::vectorized_elementwise_kernel<4, at...         0.00%       0.000us         0.00%       0.000us       0.000us      30.000us         0.03%      30.000us       1.000us            30  
void at::native::vectorized_elementwise_kernel<4, at...         0.00%       0.000us         0.00%       0.000us       0.000us      20.000us         0.02%      20.000us       1.000us            20  
                                       cudaLaunchKernel         0.35%     395.000us         0.35%     395.000us       4.938us       0.000us         0.00%       0.000us       0.000us            80  
                                  cudaStreamIsCapturing         0.00%       2.000us         0.00%       2.000us       0.100us       0.000us         0.00%       0.000us       0.000us            20  
                                        cudaGraphLaunch         0.43%     485.000us         0.43%     485.000us      48.500us       0.000us         0.00%       0.000us       0.000us            10  
                                   cudaDriverGetVersion         0.00%       0.000us         0.00%       0.000us       0.000us       0.000us         0.00%       0.000us       0.000us            10  
                                  cudaDeviceSynchronize        99.21%     111.332ms        99.21%     111.332ms     111.332ms       0.000us         0.00%       0.000us       0.000us             1  
-------------------------------------------------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  
Self CPU time total: 112.214ms
Self CUDA time total: 112.749ms