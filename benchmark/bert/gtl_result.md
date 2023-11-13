-------------------------------------------------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  
                                                   Name    Self CPU %      Self CPU   CPU total %     CPU total  CPU time avg     Self CUDA   Self CUDA %    CUDA total  CUDA time avg    # of Calls  
-------------------------------------------------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  
cutlass_tensorop_f16_s16816gemm_f16_128x128_32x4_nt_...         0.00%       0.000us         0.00%       0.000us       0.000us      37.381ms        14.89%      37.381ms     257.800us           145  
cutlass_tensorop_f16_s16816gemm_f16_128x256_32x3_tn_...         0.00%       0.000us         0.00%       0.000us       0.000us      24.587ms         9.79%      24.587ms     256.115us            96  
cutlass_tensorop_f16_s16816gemm_f16_128x256_64x3_tn_...         0.00%       0.000us         0.00%       0.000us       0.000us      23.170ms         9.23%      23.170ms     472.857us            49  
cutlass_tensorop_f16_s16816gemm_f16_64x256_32x4_tt_a...         0.00%       0.000us         0.00%       0.000us       0.000us      21.649ms         8.62%      21.649ms     432.980us            50  
                                         softmax_kernel         0.00%       0.000us         0.00%       0.000us       0.000us      20.123ms         8.02%      20.123ms     773.962us            26  
cutlass_tensorop_f16_s16816gemm_f16_128x256_32x3_tt_...         0.00%       0.000us         0.00%       0.000us       0.000us      17.799ms         7.09%      17.799ms     711.960us            25  
cutlass_tensorop_f16_s16816gemm_f16_128x128_32x5_tt_...         0.00%       0.000us         0.00%       0.000us       0.000us      15.985ms         6.37%      15.985ms     340.106us            47  
                                softmax_backward_kernel         0.00%       0.000us         0.00%       0.000us       0.000us      13.711ms         5.46%      13.711ms     571.292us            24  
cutlass_tensorop_f16_s16816gemm_f16_128x256_64x3_tt_...         0.00%       0.000us         0.00%       0.000us       0.000us      12.866ms         5.13%      12.866ms     178.694us            72  
cutlass_tensorop_f16_s16816gemm_f16_128x64_64x3_tt_a...         0.00%       0.000us         0.00%       0.000us       0.000us      12.594ms         5.02%      12.594ms     262.375us            48  
cutlass_tensorop_f16_s16816gemm_f16_256x64_32x4_nt_a...         0.00%       0.000us         0.00%       0.000us       0.000us      12.430ms         4.95%      12.430ms     258.958us            48  
void at::native::elementwise_kernel<128, 4, at::nati...         0.00%       0.000us         0.00%       0.000us       0.000us      10.056ms         4.01%      10.056ms     182.836us            55  
                              layernorm_backward_kernel         0.00%       0.000us         0.00%       0.000us       0.000us       5.019ms         2.00%       5.019ms     102.429us            49  
cutlass_tensorop_f16_s16816gemm_f16_128x128_32x3_tn_...         0.00%       0.000us         0.00%       0.000us       0.000us       4.054ms         1.61%       4.054ms       4.054ms             1  
void at::native::reduce_kernel<128, 4, at::native::R...         0.00%       0.000us         0.00%       0.000us       0.000us       3.634ms         1.45%       3.634ms      71.255us            51  
cutlass_tensorop_f16_s16816gemm_f16_128x256_32x3_nt_...         0.00%       0.000us         0.00%       0.000us       0.000us       3.530ms         1.41%       3.530ms       3.530ms             1  
void at::native::vectorized_elementwise_kernel<4, at...         0.00%       0.000us         0.00%       0.000us       0.000us       2.379ms         0.95%       2.379ms       5.918us           402  
                                       layernorm_kernel         0.00%       0.000us         0.00%       0.000us       0.000us       2.221ms         0.88%       2.221ms      45.327us            49  
void at::native::elementwise_kernel<128, 4, at::nati...         0.00%       0.000us         0.00%       0.000us       0.000us       1.794ms         0.71%       1.794ms      78.000us            23  
void at::native::elementwise_kernel<128, 4, at::nati...         0.00%       0.000us         0.00%       0.000us       0.000us       1.772ms         0.71%       1.772ms      77.043us            23  
void at::native::vectorized_elementwise_kernel<4, at...         0.00%       0.000us         0.00%       0.000us       0.000us       1.228ms         0.49%       1.228ms       2.241us           548  
void at::native::(anonymous namespace)::indexSelectL...         0.00%       0.000us         0.00%       0.000us       0.000us     547.000us         0.22%     547.000us     182.333us             3  
void at::native::(anonymous namespace)::sum_and_scat...         0.00%       0.000us         0.00%       0.000us       0.000us     318.000us         0.13%     318.000us     106.000us             3  
void at::native::(anonymous namespace)::compute_grad...         0.00%       0.000us         0.00%       0.000us       0.000us     285.000us         0.11%     285.000us      95.000us             3  
void at_cuda_detail::cub::DeviceRadixSortOnesweepKer...         0.00%       0.000us         0.00%       0.000us       0.000us     231.000us         0.09%     231.000us       9.625us            24  
void at::native::unrolled_elementwise_kernel<at::nat...         0.00%       0.000us         0.00%       0.000us       0.000us     192.000us         0.08%     192.000us       3.840us            50  
void at::native::elementwise_kernel<128, 4, at::nati...         0.00%       0.000us         0.00%       0.000us       0.000us     164.000us         0.07%     164.000us     164.000us             1  
void at::native::vectorized_elementwise_kernel<4, at...         0.00%       0.000us         0.00%       0.000us       0.000us     148.000us         0.06%     148.000us       1.000us           148  
void at::native::(anonymous namespace)::GammaBetaBac...         0.00%       0.000us         0.00%       0.000us       0.000us     120.000us         0.05%     120.000us     120.000us             1  
void at::native::elementwise_kernel<128, 4, at::nati...         0.00%       0.000us         0.00%       0.000us       0.000us     118.000us         0.05%     118.000us     118.000us             1  
void at::native::elementwise_kernel<128, 2, at::nati...         0.00%       0.000us         0.00%       0.000us       0.000us     104.000us         0.04%     104.000us     104.000us             1  
                                       Memset (Unknown)         0.00%       0.000us         0.00%       0.000us       0.000us      92.000us         0.04%      92.000us       1.769us            52  
void at::native::reduce_kernel<128, 4, at::native::R...         0.00%       0.000us         0.00%       0.000us       0.000us      87.000us         0.03%      87.000us      87.000us             1  
void at::native::(anonymous namespace)::layer_norm_g...         0.00%       0.000us         0.00%       0.000us       0.000us      86.000us         0.03%      86.000us      86.000us             1  
void at::native::(anonymous namespace)::vectorized_l...         0.00%       0.000us         0.00%       0.000us       0.000us      75.000us         0.03%      75.000us      75.000us             1  
void at::native::(anonymous namespace)::fused_dropou...         0.00%       0.000us         0.00%       0.000us       0.000us      71.000us         0.03%      71.000us      71.000us             1  
void at::native::vectorized_elementwise_kernel<4, at...         0.00%       0.000us         0.00%       0.000us       0.000us      65.000us         0.03%      65.000us      65.000us             1  
                         Memcpy DtoD (Device -> Device)         0.00%       0.000us         0.00%       0.000us       0.000us      59.000us         0.02%      59.000us      11.800us             5  
                                              memcpy128         0.00%       0.000us         0.00%       0.000us       0.000us      55.000us         0.02%      55.000us      55.000us             1  
void at::native::vectorized_elementwise_kernel<4, at...         0.00%       0.000us         0.00%       0.000us       0.000us      49.000us         0.02%      49.000us       1.000us            49  
                                        Memset (Device)         0.00%       0.000us         0.00%       0.000us       0.000us      36.000us         0.01%      36.000us       1.200us            30  
void at_cuda_detail::cub::DeviceUniqueByKeySweepKern...         0.00%       0.000us         0.00%       0.000us       0.000us      16.000us         0.01%      16.000us       5.333us             3  
cutlass_tensorop_f16_s16816gemm_f16_64x64_64x5_tn_al...         0.00%       0.000us         0.00%       0.000us       0.000us      15.000us         0.01%      15.000us       7.500us             2  
void at_cuda_detail::cub::DeviceRadixSortHistogramKe...         0.00%       0.000us         0.00%       0.000us       0.000us      15.000us         0.01%      15.000us       5.000us             3  
void at::native::(anonymous namespace)::krn_partial_...         0.00%       0.000us         0.00%       0.000us       0.000us      15.000us         0.01%      15.000us       5.000us             3  
cutlass_tensorop_f16_s16816gemm_f16_64x64_64x5_tt_al...         0.00%       0.000us         0.00%       0.000us       0.000us       8.000us         0.00%       8.000us       8.000us             1  
void at_cuda_detail::cub::DeviceScanKernel<at_cuda_d...         0.00%       0.000us         0.00%       0.000us       0.000us       8.000us         0.00%       8.000us       2.667us             3  
void (anonymous namespace)::elementwise_kernel_with_...         0.00%       0.000us         0.00%       0.000us       0.000us       7.000us         0.00%       7.000us       1.750us             4  
void at::native::elementwise_kernel<128, 2, at::nati...         0.00%       0.000us         0.00%       0.000us       0.000us       7.000us         0.00%       7.000us       3.500us             2  
cutlass_tensorop_f16_s16816gemm_f16_64x128_64x3_tt_a...         0.00%       0.000us         0.00%       0.000us       0.000us       7.000us         0.00%       7.000us       7.000us             1  
void at_cuda_detail::cub::DeviceRadixSortExclusiveSu...         0.00%       0.000us         0.00%       0.000us       0.000us       6.000us         0.00%       6.000us       2.000us             3  
void at::native::reduce_kernel<512, 1, at::native::R...         0.00%       0.000us         0.00%       0.000us       0.000us       5.000us         0.00%       5.000us       5.000us             1  
cutlass_tensorop_f16_s16816gemm_f16_256x64_32x3_nt_a...         0.00%       0.000us         0.00%       0.000us       0.000us       5.000us         0.00%       5.000us       5.000us             1  
void at::native::(anonymous namespace)::krn_partials...         0.00%       0.000us         0.00%       0.000us       0.000us       5.000us         0.00%       5.000us       1.667us             3  
void at::native::unrolled_elementwise_kernel<at::nat...         0.00%       0.000us         0.00%       0.000us       0.000us       4.000us         0.00%       4.000us       2.000us             2  
void at::native::vectorized_elementwise_kernel<4, at...         0.00%       0.000us         0.00%       0.000us       0.000us       4.000us         0.00%       4.000us       2.000us             2  
void at::native::reduce_kernel<256, 2, at::native::R...         0.00%       0.000us         0.00%       0.000us       0.000us       4.000us         0.00%       4.000us       4.000us             1  
cutlass_tensorop_f16_s16816gemm_f16_64x128_64x3_nt_a...         0.00%       0.000us         0.00%       0.000us       0.000us       4.000us         0.00%       4.000us       4.000us             1  
void at_cuda_detail::cub::DeviceCompactInitKernel<at...         0.00%       0.000us         0.00%       0.000us       0.000us       4.000us         0.00%       4.000us       1.333us             3  
void at::native::vectorized_elementwise_kernel<4, at...         0.00%       0.000us         0.00%       0.000us       0.000us       3.000us         0.00%       3.000us       1.500us             2  
void at::native::unrolled_elementwise_kernel<at::nat...         0.00%       0.000us         0.00%       0.000us       0.000us       3.000us         0.00%       3.000us       3.000us             1  
void at_cuda_detail::cub::DeviceScanInitKernel<at_cu...         0.00%       0.000us         0.00%       0.000us       0.000us       3.000us         0.00%       3.000us       1.000us             3  
void at::native::(anonymous namespace)::compute_num_...         0.00%       0.000us         0.00%       0.000us       0.000us       3.000us         0.00%       3.000us       1.000us             3  
void at::native::vectorized_elementwise_kernel<4, at...         0.00%       0.000us         0.00%       0.000us       0.000us       2.000us         0.00%       2.000us       2.000us             1  
void at::native::unrolled_elementwise_kernel<at::nat...         0.00%       0.000us         0.00%       0.000us       0.000us       2.000us         0.00%       2.000us       2.000us             1  
void at::native::vectorized_elementwise_kernel<4, at...         0.00%       0.000us         0.00%       0.000us       0.000us       1.000us         0.00%       1.000us       1.000us             1  
                                       cudaLaunchKernel         1.00%       2.499ms         1.00%       2.499ms       5.195us       0.000us         0.00%       0.000us       0.000us           481  
                                        cudaMemcpyAsync         0.02%      62.000us         0.02%      62.000us      12.400us       0.000us         0.00%       0.000us       0.000us             5  
                                         cudaEventQuery         0.00%      11.000us         0.00%      11.000us      11.000us       0.000us         0.00%       0.000us       0.000us             1  
                                  cudaStreamIsCapturing         0.00%       2.000us         0.00%       2.000us       0.667us       0.000us         0.00%       0.000us       0.000us             3  
                                        cudaGraphLaunch         0.89%       2.228ms         0.89%       2.228ms       2.228ms       0.000us         0.00%       0.000us       0.000us             1  
                                   cudaDriverGetVersion         0.00%       0.000us         0.00%       0.000us       0.000us       0.000us         0.00%       0.000us       0.000us             1  
                                        cudaMemsetAsync         0.11%     277.000us         0.11%     277.000us       9.233us       0.000us         0.00%       0.000us       0.000us            30  
                                 cudaDeviceGetAttribute         0.00%       3.000us         0.00%       3.000us       0.200us       0.000us         0.00%       0.000us       0.000us            15  
cudaOccupancyMaxActiveBlocksPerMultiprocessorWithFla...         0.00%       9.000us         0.00%       9.000us       1.500us       0.000us         0.00%       0.000us       0.000us             6  
                                    cudaPeekAtLastError         0.00%       0.000us         0.00%       0.000us       0.000us       0.000us         0.00%       0.000us       0.000us            51  
                                    cudaStreamWaitEvent         0.00%       5.000us         0.00%       5.000us       2.500us       0.000us         0.00%       0.000us       0.000us             2  
                                  cudaDeviceSynchronize        97.97%     245.968ms        97.97%     245.968ms     245.968ms       0.000us         0.00%       0.000us       0.000us             1  
-------------------------------------------------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  
Self CPU time total: 251.064ms
Self CUDA time total: 251.040ms