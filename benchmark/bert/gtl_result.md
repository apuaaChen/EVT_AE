-------------------------------------------------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  
                                                   Name    Self CPU %      Self CPU   CPU total %     CPU total  CPU time avg     Self CUDA   Self CUDA %    CUDA total  CUDA time avg    # of Calls  
-------------------------------------------------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  
cutlass_tensorop_f16_s16816gemm_f16_128x128_32x5_tt_...         0.00%       0.000us         0.00%       0.000us       0.000us     493.673ms        19.92%     493.673ms     411.394us          1200  
cutlass_tensorop_f16_s16816gemm_f16_128x128_32x3_tn_...         0.00%       0.000us         0.00%       0.000us       0.000us     324.700ms        13.10%     324.700ms     268.347us          1210  
cutlass_tensorop_f16_s16816gemm_f16_128x128_32x4_nt_...         0.00%       0.000us         0.00%       0.000us       0.000us     235.897ms         9.52%     235.897ms     491.452us           480  
                                         softmax_kernel         0.00%       0.000us         0.00%       0.000us       0.000us     201.146ms         8.12%     201.146ms     773.638us           260  
cutlass_tensorop_f16_s16816gemm_f16_128x256_32x3_tn_...         0.00%       0.000us         0.00%       0.000us       0.000us     180.303ms         7.28%     180.303ms     721.212us           250  
cutlass_tensorop_f16_s16816gemm_f16_128x128_32x3_tt_...         0.00%       0.000us         0.00%       0.000us       0.000us     138.671ms         5.60%     138.671ms     192.599us           720  
                                softmax_backward_kernel         0.00%       0.000us         0.00%       0.000us       0.000us     137.273ms         5.54%     137.273ms     571.971us           240  
cutlass_tensorop_f16_s16816gemm_f16_128x128_32x5_nt_...         0.00%       0.000us         0.00%       0.000us       0.000us     130.749ms         5.28%     130.749ms     134.793us           970  
cutlass_tensorop_f16_s16816gemm_f16_256x64_32x4_nt_a...         0.00%       0.000us         0.00%       0.000us       0.000us     123.888ms         5.00%     123.888ms     258.100us           480  
void at::native::elementwise_kernel<128, 4, at::nati...         0.00%       0.000us         0.00%       0.000us       0.000us      95.778ms         3.87%      95.778ms     177.367us           540  
cutlass_tensorop_f16_s16816gemm_f16_128x64_64x3_tt_a...         0.00%       0.000us         0.00%       0.000us       0.000us      63.427ms         2.56%      63.427ms     264.279us           240  
cutlass_tensorop_f16_s16816gemm_f16_64x64_64x5_tt_al...         0.00%       0.000us         0.00%       0.000us       0.000us      62.552ms         2.52%      62.552ms     250.208us           250  
                              layernorm_backward_kernel         0.00%       0.000us         0.00%       0.000us       0.000us      50.551ms         2.04%      50.551ms     103.165us           490  
cutlass_tensorop_f16_s16816gemm_f16_128x256_64x3_tt_...         0.00%       0.000us         0.00%       0.000us       0.000us      37.164ms         1.50%      37.164ms       3.716ms            10  
void at::native::reduce_kernel<128, 4, at::native::R...         0.00%       0.000us         0.00%       0.000us       0.000us      36.335ms         1.47%      36.335ms      71.245us           510  
cutlass_tensorop_f16_s16816gemm_f16_128x256_64x3_nt_...         0.00%       0.000us         0.00%       0.000us       0.000us      35.522ms         1.43%      35.522ms       3.552ms            10  
void at::native::vectorized_elementwise_kernel<4, at...         0.00%       0.000us         0.00%       0.000us       0.000us      23.613ms         0.95%      23.613ms       5.874us          4020  
                                       layernorm_kernel         0.00%       0.000us         0.00%       0.000us       0.000us      22.846ms         0.92%      22.846ms      46.624us           490  
void at::native::elementwise_kernel<128, 4, at::nati...         0.00%       0.000us         0.00%       0.000us       0.000us      17.879ms         0.72%      17.879ms      77.735us           230  
void at::native::elementwise_kernel<128, 4, at::nati...         0.00%       0.000us         0.00%       0.000us       0.000us      17.632ms         0.71%      17.632ms      76.661us           230  
void at::native::vectorized_elementwise_kernel<4, at...         0.00%       0.000us         0.00%       0.000us       0.000us      11.945ms         0.48%      11.945ms       2.180us          5480  
void at::native::(anonymous namespace)::indexSelectL...         0.00%       0.000us         0.00%       0.000us       0.000us       5.510ms         0.22%       5.510ms     183.667us            30  
void at::native::vectorized_elementwise_kernel<4, at...         0.00%       0.000us         0.00%       0.000us       0.000us       3.900ms         0.16%       3.900ms       1.000us          3900  
void at::native::(anonymous namespace)::sum_and_scat...         0.00%       0.000us         0.00%       0.000us       0.000us       3.145ms         0.13%       3.145ms     104.833us            30  
cutlass_tensorop_f16_s16816gemm_f16_64x256_32x4_tt_a...         0.00%       0.000us         0.00%       0.000us       0.000us       2.858ms         0.12%       2.858ms     285.800us            10  
void at::native::(anonymous namespace)::compute_grad...         0.00%       0.000us         0.00%       0.000us       0.000us       2.824ms         0.11%       2.824ms      94.133us            30  
void at_cuda_detail::cub::DeviceRadixSortOnesweepKer...         0.00%       0.000us         0.00%       0.000us       0.000us       2.302ms         0.09%       2.302ms       9.592us           240  
void at::native::unrolled_elementwise_kernel<at::nat...         0.00%       0.000us         0.00%       0.000us       0.000us       1.924ms         0.08%       1.924ms       3.848us           500  
                                              memcpy128         0.00%       0.000us         0.00%       0.000us       0.000us       1.576ms         0.06%       1.576ms      52.533us            30  
void at::native::unrolled_elementwise_kernel<at::nat...         0.00%       0.000us         0.00%       0.000us       0.000us       1.329ms         0.05%       1.329ms      66.450us            20  
void at::native::(anonymous namespace)::GammaBetaBac...         0.00%       0.000us         0.00%       0.000us       0.000us       1.186ms         0.05%       1.186ms     118.600us            10  
void at::native::elementwise_kernel<128, 4, at::nati...         0.00%       0.000us         0.00%       0.000us       0.000us       1.183ms         0.05%       1.183ms     118.300us            10  
void at::native::elementwise_kernel<128, 2, at::nati...         0.00%       0.000us         0.00%       0.000us       0.000us       1.050ms         0.04%       1.050ms     105.000us            10  
                                       Memset (Unknown)         0.00%       0.000us         0.00%       0.000us       0.000us     919.000us         0.04%     919.000us       1.767us           520  
void at::native::(anonymous namespace)::layer_norm_g...         0.00%       0.000us         0.00%       0.000us       0.000us     854.000us         0.03%     854.000us      85.400us            10  
void at::native::reduce_kernel<128, 4, at::native::R...         0.00%       0.000us         0.00%       0.000us       0.000us     825.000us         0.03%     825.000us      82.500us            10  
void at::native::(anonymous namespace)::vectorized_l...         0.00%       0.000us         0.00%       0.000us       0.000us     752.000us         0.03%     752.000us      75.200us            10  
void at::native::(anonymous namespace)::fused_dropou...         0.00%       0.000us         0.00%       0.000us       0.000us     713.000us         0.03%     713.000us      71.300us            10  
void at::native::vectorized_elementwise_kernel<4, at...         0.00%       0.000us         0.00%       0.000us       0.000us     663.000us         0.03%     663.000us      66.300us            10  
                         Memcpy DtoD (Device -> Device)         0.00%       0.000us         0.00%       0.000us       0.000us     600.000us         0.02%     600.000us      12.000us            50  
void at::native::vectorized_elementwise_kernel<4, at...         0.00%       0.000us         0.00%       0.000us       0.000us     490.000us         0.02%     490.000us       1.000us           490  
                                        Memset (Device)         0.00%       0.000us         0.00%       0.000us       0.000us     355.000us         0.01%     355.000us       1.183us           300  
cutlass_tensorop_f16_s16816gemm_f16_64x64_64x5_tn_al...         0.00%       0.000us         0.00%       0.000us       0.000us     154.000us         0.01%     154.000us       7.700us            20  
void at_cuda_detail::cub::DeviceRadixSortHistogramKe...         0.00%       0.000us         0.00%       0.000us       0.000us     150.000us         0.01%     150.000us       5.000us            30  
void at::native::(anonymous namespace)::krn_partial_...         0.00%       0.000us         0.00%       0.000us       0.000us     140.000us         0.01%     140.000us       4.667us            30  
void at_cuda_detail::cub::DeviceUniqueByKeySweepKern...         0.00%       0.000us         0.00%       0.000us       0.000us     138.000us         0.01%     138.000us       4.600us            30  
void at_cuda_detail::cub::DeviceScanKernel<at_cuda_d...         0.00%       0.000us         0.00%       0.000us       0.000us      80.000us         0.00%      80.000us       2.667us            30  
void (anonymous namespace)::elementwise_kernel_with_...         0.00%       0.000us         0.00%       0.000us       0.000us      70.000us         0.00%      70.000us       1.750us            40  
void at::native::elementwise_kernel<128, 2, at::nati...         0.00%       0.000us         0.00%       0.000us       0.000us      70.000us         0.00%      70.000us       3.500us            20  
cutlass_tensorop_f16_s16816gemm_f16_64x128_64x3_tt_a...         0.00%       0.000us         0.00%       0.000us       0.000us      70.000us         0.00%      70.000us       7.000us            10  
void at_cuda_detail::cub::DeviceRadixSortExclusiveSu...         0.00%       0.000us         0.00%       0.000us       0.000us      60.000us         0.00%      60.000us       2.000us            30  
void at::native::(anonymous namespace)::krn_partials...         0.00%       0.000us         0.00%       0.000us       0.000us      55.000us         0.00%      55.000us       1.833us            30  
void at::native::reduce_kernel<512, 1, at::native::R...         0.00%       0.000us         0.00%       0.000us       0.000us      50.000us         0.00%      50.000us       5.000us            10  
cutlass_tensorop_f16_s16816gemm_f16_256x64_64x4_nt_a...         0.00%       0.000us         0.00%       0.000us       0.000us      50.000us         0.00%      50.000us       5.000us            10  
void at::native::unrolled_elementwise_kernel<at::nat...         0.00%       0.000us         0.00%       0.000us       0.000us      40.000us         0.00%      40.000us       2.000us            20  
void at::native::vectorized_elementwise_kernel<4, at...         0.00%       0.000us         0.00%       0.000us       0.000us      40.000us         0.00%      40.000us       2.000us            20  
void at::native::reduce_kernel<256, 2, at::native::R...         0.00%       0.000us         0.00%       0.000us       0.000us      40.000us         0.00%      40.000us       4.000us            10  
cutlass_tensorop_f16_s16816gemm_f16_64x128_64x3_nt_a...         0.00%       0.000us         0.00%       0.000us       0.000us      40.000us         0.00%      40.000us       4.000us            10  
void at_cuda_detail::cub::DeviceCompactInitKernel<at...         0.00%       0.000us         0.00%       0.000us       0.000us      40.000us         0.00%      40.000us       1.333us            30  
void at::native::vectorized_elementwise_kernel<4, at...         0.00%       0.000us         0.00%       0.000us       0.000us      30.000us         0.00%      30.000us       1.500us            20  
void at::native::unrolled_elementwise_kernel<at::nat...         0.00%       0.000us         0.00%       0.000us       0.000us      30.000us         0.00%      30.000us       3.000us            10  
void at_cuda_detail::cub::DeviceScanInitKernel<at_cu...         0.00%       0.000us         0.00%       0.000us       0.000us      30.000us         0.00%      30.000us       1.000us            30  
void at::native::(anonymous namespace)::compute_num_...         0.00%       0.000us         0.00%       0.000us       0.000us      30.000us         0.00%      30.000us       1.000us            30  
void at::native::vectorized_elementwise_kernel<4, at...         0.00%       0.000us         0.00%       0.000us       0.000us      20.000us         0.00%      20.000us       2.000us            10  
void at::native::vectorized_elementwise_kernel<4, at...         0.00%       0.000us         0.00%       0.000us       0.000us      10.000us         0.00%      10.000us       1.000us            10  
                                       cudaLaunchKernel        78.68%        1.914s        78.68%        1.914s     397.948us       0.000us         0.00%       0.000us       0.000us          4810  
                                        cudaMemcpyAsync         0.02%     399.000us         0.02%     399.000us       7.980us       0.000us         0.00%       0.000us       0.000us            50  
                                         cudaEventQuery         0.02%     580.000us         0.02%     580.000us       1.021us       0.000us         0.00%       0.000us       0.000us           568  
                                  cudaStreamIsCapturing         0.00%       9.000us         0.00%       9.000us       0.300us       0.000us         0.00%       0.000us       0.000us            30  
                                        cudaGraphLaunch         0.81%      19.793ms         0.81%      19.793ms       1.979ms       0.000us         0.00%       0.000us       0.000us            10  
                                   cudaDriverGetVersion         0.00%       0.000us         0.00%       0.000us       0.000us       0.000us         0.00%       0.000us       0.000us            10  
                                        cudaMemsetAsync         0.12%       2.909ms         0.12%       2.909ms       9.697us       0.000us         0.00%       0.000us       0.000us           300  
                                 cudaDeviceGetAttribute         0.00%      36.000us         0.00%      36.000us       0.240us       0.000us         0.00%       0.000us       0.000us           150  
cudaOccupancyMaxActiveBlocksPerMultiprocessorWithFla...         0.01%     254.000us         0.01%     254.000us       4.233us       0.000us         0.00%       0.000us       0.000us            60  
                                    cudaPeekAtLastError         0.00%       0.000us         0.00%       0.000us       0.000us       0.000us         0.00%       0.000us       0.000us           510  
                                    cudaStreamWaitEvent         0.00%      42.000us         0.00%      42.000us       2.100us       0.000us         0.00%       0.000us       0.000us            20  
                                  cudaDeviceSynchronize        20.33%     494.524ms        20.33%     494.524ms     494.524ms       0.000us         0.00%       0.000us       0.000us             1  
-------------------------------------------------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  
Self CPU time total: 2.433s
Self CUDA time total: 2.478s