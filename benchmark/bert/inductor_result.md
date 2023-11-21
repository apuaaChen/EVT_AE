-------------------------------------------------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  
                                                   Name    Self CPU %      Self CPU   CPU total %     CPU total  CPU time avg     Self CUDA   Self CUDA %    CUDA total  CUDA time avg    # of Calls  
-------------------------------------------------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  
                            triton__0d1d2d3d4d5d6d78d9d         0.00%       0.000us         0.00%       0.000us       0.000us     370.387ms        12.59%     370.387ms       1.543ms           240  
                                         triton__0d1d2d         0.00%       0.000us         0.00%       0.000us       0.000us     301.002ms        10.23%     301.002ms     172.001us          1750  
ampere_fp16_s16816gemm_fp16_128x128_ldg8_f2f_stages_...         0.00%       0.000us         0.00%       0.000us       0.000us     270.412ms         9.19%     270.412ms     223.481us          1210  
void cutlass::Kernel<cutlass_80_tensorop_f16_s16816g...         0.00%       0.000us         0.00%       0.000us       0.000us     264.772ms         9.00%     264.772ms     551.608us           480  
void cutlass::Kernel<cutlass_80_tensorop_f16_s16816g...         0.00%       0.000us         0.00%       0.000us       0.000us     257.388ms         8.75%     257.388ms     212.717us          1210  
ampere_fp16_s16816gemm_fp16_128x128_ldg8_relu_f2f_st...         0.00%       0.000us         0.00%       0.000us       0.000us     166.901ms         5.67%     166.901ms     177.554us           940  
                                   triton__0d1d2d3d4d5d         0.00%       0.000us         0.00%       0.000us       0.000us     165.175ms         5.61%     165.175ms     660.700us           250  
                                       triton__0d1d2d3d         0.00%       0.000us         0.00%       0.000us       0.000us     152.503ms         5.18%     152.503ms      41.554us          3670  
void cutlass::Kernel<cutlass_80_tensorop_f16_s16816g...         0.00%       0.000us         0.00%       0.000us       0.000us     128.231ms         4.36%     128.231ms     534.296us           240  
void cutlass::Kernel<cutlass_80_tensorop_f16_s16816g...         0.00%       0.000us         0.00%       0.000us       0.000us     122.790ms         4.17%     122.790ms     511.625us           240  
ampere_fp16_s16816gemm_fp16_64x128_sliced1x2_ldg8_f2...         0.00%       0.000us         0.00%       0.000us       0.000us     122.536ms         4.16%     122.536ms     260.715us           470  
                        triton__0d1d2d3d4d5d6d7d8d9d10d         0.00%       0.000us         0.00%       0.000us       0.000us      80.592ms         2.74%      80.592ms     171.472us           470  
                      triton__0d1d2d3d4d5d6d7d8d910d11d         0.00%       0.000us         0.00%       0.000us       0.000us      62.435ms         2.12%      62.435ms     130.073us           480  
void cutlass::Kernel<cutlass_80_tensorop_f16_s16816g...         0.00%       0.000us         0.00%       0.000us       0.000us      60.413ms         2.05%      60.413ms     251.721us           240  
void cutlass::Kernel<cutlass_80_tensorop_f16_s16816g...         0.00%       0.000us         0.00%       0.000us       0.000us      58.046ms         1.97%      58.046ms     241.858us           240  
                             triton__0d1d2d3d4d5d6d7d8d         0.00%       0.000us         0.00%       0.000us       0.000us      42.475ms         1.44%      42.475ms     151.696us           280  
                  triton__0d1d2d3d4d5d6d7d8d9d10d11d12d         0.00%       0.000us         0.00%       0.000us       0.000us      40.336ms         1.37%      40.336ms     175.374us           230  
ampere_fp16_s16816gemm_fp16_128x128_ldg8_f2f_stages_...         0.00%       0.000us         0.00%       0.000us       0.000us      39.997ms         1.36%      39.997ms       4.000ms            10  
ampere_fp16_s16816gemm_fp16_256x128_ldg8_f2f_stages_...         0.00%       0.000us         0.00%       0.000us       0.000us      37.452ms         1.27%      37.452ms       3.745ms            10  
ampere_fp16_s16816gemm_fp16_256x128_ldg8_f2f_stages_...         0.00%       0.000us         0.00%       0.000us       0.000us      36.805ms         1.25%      36.805ms       3.680ms            10  
                                     triton__0d1d2d3d4d         0.00%       0.000us         0.00%       0.000us       0.000us      36.462ms         1.24%      36.462ms     607.700us            60  
                                 triton__0d1d2d3d4d5d6d         0.00%       0.000us         0.00%       0.000us       0.000us      29.005ms         0.99%      29.005ms       2.901ms            10  
                                        triton__0d1d2d3         0.00%       0.000us         0.00%       0.000us       0.000us      25.923ms         0.88%      25.923ms      54.006us           480  
void at::native::vectorized_elementwise_kernel<4, at...         0.00%       0.000us         0.00%       0.000us       0.000us      24.942ms         0.85%      24.942ms       6.220us          4010  
void at::native::vectorized_elementwise_kernel<4, at...         0.00%       0.000us         0.00%       0.000us       0.000us      10.250ms         0.35%      10.250ms       2.556us          4010  
                                           triton__0d1d         0.00%       0.000us         0.00%       0.000us       0.000us       6.466ms         0.22%       6.466ms     215.533us            30  
void at::native::(anonymous namespace)::indexSelectL...         0.00%       0.000us         0.00%       0.000us       0.000us       5.467ms         0.19%       5.467ms     182.233us            30  
                                        Memset (Device)         0.00%       0.000us         0.00%       0.000us       0.000us       3.489ms         0.12%       3.489ms       1.897us          1839  
void at::native::(anonymous namespace)::sum_and_scat...         0.00%       0.000us         0.00%       0.000us       0.000us       3.161ms         0.11%       3.161ms     105.367us            30  
                                       Memset (Unknown)         0.00%       0.000us         0.00%       0.000us       0.000us       3.099ms         0.11%       3.099ms       2.294us          1351  
void at::native::(anonymous namespace)::compute_grad...         0.00%       0.000us         0.00%       0.000us       0.000us       2.826ms         0.10%       2.826ms      94.200us            30  
void at_cuda_detail::cub::DeviceRadixSortOnesweepKer...         0.00%       0.000us         0.00%       0.000us       0.000us       2.305ms         0.08%       2.305ms       9.604us           240  
                     triton__0d1d2d3d4d5d6d7d8d9d10d11d         0.00%       0.000us         0.00%       0.000us       0.000us       1.185ms         0.04%       1.185ms     118.500us            10  
void at::native::(anonymous namespace)::GammaBetaBac...         0.00%       0.000us         0.00%       0.000us       0.000us       1.184ms         0.04%       1.184ms     118.400us            10  
void at::native::elementwise_kernel<128, 4, at::nati...         0.00%       0.000us         0.00%       0.000us       0.000us     912.000us         0.03%     912.000us      91.200us            10  
void at::native::(anonymous namespace)::layer_norm_g...         0.00%       0.000us         0.00%       0.000us       0.000us     846.000us         0.03%     846.000us      84.600us            10  
                           triton__0d1d2d3d4d5d6d7d8d9d         0.00%       0.000us         0.00%       0.000us       0.000us     833.000us         0.03%     833.000us      83.300us            10  
void at::native::(anonymous namespace)::vectorized_l...         0.00%       0.000us         0.00%       0.000us       0.000us     751.000us         0.03%     751.000us      75.100us            10  
void at::native::(anonymous namespace)::fused_dropou...         0.00%       0.000us         0.00%       0.000us       0.000us     716.000us         0.02%     716.000us      71.600us            10  
                         Memcpy DtoD (Device -> Device)         0.00%       0.000us         0.00%       0.000us       0.000us     592.000us         0.02%     592.000us      11.840us            50  
void at_cuda_detail::cub::DeviceUniqueByKeySweepKern...         0.00%       0.000us         0.00%       0.000us       0.000us     157.000us         0.01%     157.000us       5.233us            30  
void at_cuda_detail::cub::DeviceRadixSortHistogramKe...         0.00%       0.000us         0.00%       0.000us       0.000us     150.000us         0.01%     150.000us       5.000us            30  
void at::native::(anonymous namespace)::krn_partial_...         0.00%       0.000us         0.00%       0.000us       0.000us     140.000us         0.00%     140.000us       4.667us            30  
void at_cuda_detail::cub::DeviceScanKernel<at_cuda_d...         0.00%       0.000us         0.00%       0.000us       0.000us      80.000us         0.00%      80.000us       2.667us            30  
ampere_fp16_s16816gemm_fp16_64x64_sliced1x2_ldg8_f2f...         0.00%       0.000us         0.00%       0.000us       0.000us      78.000us         0.00%      78.000us       7.800us            10  
void (anonymous namespace)::elementwise_kernel_with_...         0.00%       0.000us         0.00%       0.000us       0.000us      70.000us         0.00%      70.000us       1.750us            40  
void at::native::elementwise_kernel<128, 2, at::nati...         0.00%       0.000us         0.00%       0.000us       0.000us      70.000us         0.00%      70.000us       3.500us            20  
sm80_xmma_gemm_f16f16_f16f32_f32_tn_n_tilesize32x32x...         0.00%       0.000us         0.00%       0.000us       0.000us      70.000us         0.00%      70.000us       7.000us            10  
void at::native::reduce_kernel<512, 1, at::native::R...         0.00%       0.000us         0.00%       0.000us       0.000us      60.000us         0.00%      60.000us       6.000us            10  
void at_cuda_detail::cub::DeviceRadixSortExclusiveSu...         0.00%       0.000us         0.00%       0.000us       0.000us      60.000us         0.00%      60.000us       2.000us            30  
void at::native::(anonymous namespace)::krn_partials...         0.00%       0.000us         0.00%       0.000us       0.000us      60.000us         0.00%      60.000us       2.000us            30  
void at::native::unrolled_elementwise_kernel<at::nat...         0.00%       0.000us         0.00%       0.000us       0.000us      46.000us         0.00%      46.000us       2.300us            20  
void at::native::vectorized_elementwise_kernel<4, at...         0.00%       0.000us         0.00%       0.000us       0.000us      40.000us         0.00%      40.000us       1.000us            40  
void at::native::unrolled_elementwise_kernel<at::nat...         0.00%       0.000us         0.00%       0.000us       0.000us      40.000us         0.00%      40.000us       4.000us            10  
void at_cuda_detail::cub::DeviceCompactInitKernel<at...         0.00%       0.000us         0.00%       0.000us       0.000us      40.000us         0.00%      40.000us       1.333us            30  
                                        triton__0d1d23d         0.00%       0.000us         0.00%       0.000us       0.000us      30.000us         0.00%      30.000us       1.500us            20  
void at_cuda_detail::cub::DeviceScanInitKernel<at_cu...         0.00%       0.000us         0.00%       0.000us       0.000us      30.000us         0.00%      30.000us       1.000us            30  
void at::native::(anonymous namespace)::compute_num_...         0.00%       0.000us         0.00%       0.000us       0.000us      30.000us         0.00%      30.000us       1.000us            30  
void at::native::vectorized_elementwise_kernel<4, at...         0.00%       0.000us         0.00%       0.000us       0.000us      29.000us         0.00%      29.000us       2.900us            10  
void at::native::(anonymous namespace)::distribution...         0.00%       0.000us         0.00%       0.000us       0.000us      20.000us         0.00%      20.000us       2.000us            10  
                                  triton__0d1d2d3d4d56d         0.00%       0.000us         0.00%       0.000us       0.000us      20.000us         0.00%      20.000us       2.000us            10  
void at::native::vectorized_elementwise_kernel<4, at...         0.00%       0.000us         0.00%       0.000us       0.000us      20.000us         0.00%      20.000us       1.000us            20  
void at::native::unrolled_elementwise_kernel<at::nat...         0.00%       0.000us         0.00%       0.000us       0.000us      20.000us         0.00%      20.000us       2.000us            10  
void at::native::unrolled_elementwise_kernel<at::nat...         0.00%       0.000us         0.00%       0.000us       0.000us      19.000us         0.00%      19.000us       1.900us            10  
                                         triton__0d1d23         0.00%       0.000us         0.00%       0.000us       0.000us      10.000us         0.00%      10.000us       1.000us            10  
void at::native::vectorized_elementwise_kernel<4, at...         0.00%       0.000us         0.00%       0.000us       0.000us      10.000us         0.00%      10.000us       1.000us            10  
void at::native::vectorized_elementwise_kernel<4, at...         0.00%       0.000us         0.00%       0.000us       0.000us      10.000us         0.00%      10.000us       1.000us            10  
                                       cudaLaunchKernel        67.37%        1.910s        67.37%        1.910s     215.065us       0.000us         0.00%       0.000us       0.000us          8880  
                                  cudaStreamIsCapturing         0.00%      12.000us         0.00%      12.000us       0.240us       0.000us         0.00%       0.000us       0.000us            50  
                                        cudaMemcpyAsync         0.64%      18.233ms         0.64%      18.233ms     364.660us       0.000us         0.00%       0.000us       0.000us            50  
                                        cudaGraphLaunch        17.80%     504.453ms        17.80%     504.453ms      25.223ms       0.000us         0.00%       0.000us       0.000us            20  
                                   cudaDriverGetVersion         0.00%       0.000us         0.00%       0.000us       0.000us       0.000us         0.00%       0.000us       0.000us            20  
                                        cudaMemsetAsync         4.90%     138.997ms         4.90%     138.997ms     463.323us       0.000us         0.00%       0.000us       0.000us           300  
                                 cudaDeviceGetAttribute         0.00%      13.000us         0.00%      13.000us       0.087us       0.000us         0.00%       0.000us       0.000us           150  
cudaOccupancyMaxActiveBlocksPerMultiprocessorWithFla...         0.00%      23.000us         0.00%      23.000us       0.383us       0.000us         0.00%       0.000us       0.000us            60  
                                    cudaPeekAtLastError         0.00%       0.000us         0.00%       0.000us       0.000us       0.000us         0.00%       0.000us       0.000us           510  
                                  cudaDeviceSynchronize         9.28%     263.200ms         9.28%     263.200ms     263.200ms       0.000us         0.00%       0.000us       0.000us             1  
-------------------------------------------------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  
Self CPU time total: 2.835s
Self CUDA time total: 2.942s