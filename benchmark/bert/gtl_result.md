-------------------------------------------------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  
                                                   Name    Self CPU %      Self CPU   CPU total %     CPU total  CPU time avg     Self CUDA   Self CUDA %    CUDA total  CUDA time avg    # of Calls  
-------------------------------------------------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  
cutlass_tensorop_f16_s16816gemm_f16_128x128_32x5_tt_...         0.00%       0.000us         0.00%       0.000us       0.000us     394.129ms        15.82%     394.129ms     539.903us           730  
cutlass_tensorop_f16_s16816gemm_f16_128x128_32x3_tn_...         0.00%       0.000us         0.00%       0.000us       0.000us     324.535ms        13.03%     324.535ms     268.211us          1210  
cutlass_tensorop_f16_s16816gemm_f16_128x128_32x4_nt_...         0.00%       0.000us         0.00%       0.000us       0.000us     235.284ms         9.45%     235.284ms     490.175us           480  
                                         softmax_kernel         0.00%       0.000us         0.00%       0.000us       0.000us     200.980ms         8.07%     200.980ms     773.000us           260  
cutlass_tensorop_f16_s16816gemm_f16_128x256_32x3_tn_...         0.00%       0.000us         0.00%       0.000us       0.000us     180.293ms         7.24%     180.293ms     721.172us           250  
cutlass_tensorop_f16_s16816gemm_f16_128x128_32x3_tt_...         0.00%       0.000us         0.00%       0.000us       0.000us     174.246ms         7.00%     174.246ms     181.506us           960  
                                softmax_backward_kernel         0.00%       0.000us         0.00%       0.000us       0.000us     137.348ms         5.51%     137.348ms     572.283us           240  
cutlass_tensorop_f16_s16816gemm_f16_128x128_32x5_nt_...         0.00%       0.000us         0.00%       0.000us       0.000us     131.760ms         5.29%     131.760ms     135.835us           970  
cutlass_tensorop_f16_s16816gemm_f16_256x64_32x4_nt_a...         0.00%       0.000us         0.00%       0.000us       0.000us     124.247ms         4.99%     124.247ms     258.848us           480  
cutlass_tensorop_f16_s16816gemm_f16_128x256_32x3_tt_...         0.00%       0.000us         0.00%       0.000us       0.000us     110.773ms         4.45%     110.773ms     481.622us           230  
void at::native::elementwise_kernel<128, 4, at::nati...         0.00%       0.000us         0.00%       0.000us       0.000us      96.615ms         3.88%      96.615ms     175.664us           550  
cutlass_tensorop_f16_s16816gemm_f16_128x64_64x3_tt_a...         0.00%       0.000us         0.00%       0.000us       0.000us      63.219ms         2.54%      63.219ms     263.413us           240  
cutlass_tensorop_f16_s16816gemm_f16_64x64_64x5_tt_al...         0.00%       0.000us         0.00%       0.000us       0.000us      62.495ms         2.51%      62.495ms     249.980us           250  
                              layernorm_backward_kernel         0.00%       0.000us         0.00%       0.000us       0.000us      49.781ms         2.00%      49.781ms     101.594us           490  
cutlass_tensorop_f16_s16816gemm_f16_128x256_64x3_tt_...         0.00%       0.000us         0.00%       0.000us       0.000us      37.180ms         1.49%      37.180ms       3.718ms            10  
void at::native::reduce_kernel<128, 4, at::native::R...         0.00%       0.000us         0.00%       0.000us       0.000us      35.968ms         1.44%      35.968ms      70.525us           510  
cutlass_tensorop_f16_s16816gemm_f16_128x256_64x3_nt_...         0.00%       0.000us         0.00%       0.000us       0.000us      35.499ms         1.43%      35.499ms       3.550ms            10  
void at::native::vectorized_elementwise_kernel<4, at...         0.00%       0.000us         0.00%       0.000us       0.000us      23.612ms         0.95%      23.612ms       5.874us          4020  
                                       layernorm_kernel         0.00%       0.000us         0.00%       0.000us       0.000us      23.243ms         0.93%      23.243ms      47.435us           490  
void at::native::vectorized_elementwise_kernel<4, at...         0.00%       0.000us         0.00%       0.000us       0.000us      11.817ms         0.47%      11.817ms       2.156us          5480  
void at::native::(anonymous namespace)::indexSelectL...         0.00%       0.000us         0.00%       0.000us       0.000us       5.485ms         0.22%       5.485ms     182.833us            30  
void at::native::vectorized_elementwise_kernel<4, at...         0.00%       0.000us         0.00%       0.000us       0.000us       3.670ms         0.15%       3.670ms       1.000us          3670  
void at::native::(anonymous namespace)::sum_and_scat...         0.00%       0.000us         0.00%       0.000us       0.000us       3.100ms         0.12%       3.100ms     103.333us            30  
cutlass_tensorop_f16_s16816gemm_f16_64x256_32x4_tt_a...         0.00%       0.000us         0.00%       0.000us       0.000us       2.809ms         0.11%       2.809ms     280.900us            10  
void at::native::(anonymous namespace)::compute_grad...         0.00%       0.000us         0.00%       0.000us       0.000us       2.806ms         0.11%       2.806ms      93.533us            30  
void at_cuda_detail::cub::DeviceRadixSortOnesweepKer...         0.00%       0.000us         0.00%       0.000us       0.000us       2.297ms         0.09%       2.297ms       9.571us           240  
                         Memcpy DtoD (Device -> Device)         0.00%       0.000us         0.00%       0.000us       0.000us       2.164ms         0.09%       2.164ms      27.050us            80  
void at::native::unrolled_elementwise_kernel<at::nat...         0.00%       0.000us         0.00%       0.000us       0.000us       1.886ms         0.08%       1.886ms       3.772us           500  
void at::native::unrolled_elementwise_kernel<at::nat...         0.00%       0.000us         0.00%       0.000us       0.000us       1.332ms         0.05%       1.332ms      66.600us            20  
void at::native::(anonymous namespace)::GammaBetaBac...         0.00%       0.000us         0.00%       0.000us       0.000us       1.180ms         0.05%       1.180ms     118.000us            10  
void at::native::elementwise_kernel<128, 4, at::nati...         0.00%       0.000us         0.00%       0.000us       0.000us       1.165ms         0.05%       1.165ms     116.500us            10  
void at::native::elementwise_kernel<128, 2, at::nati...         0.00%       0.000us         0.00%       0.000us       0.000us       1.050ms         0.04%       1.050ms     105.000us            10  
                                       Memset (Unknown)         0.00%       0.000us         0.00%       0.000us       0.000us     929.000us         0.04%     929.000us       1.787us           520  
void at::native::(anonymous namespace)::layer_norm_g...         0.00%       0.000us         0.00%       0.000us       0.000us     864.000us         0.03%     864.000us      86.400us            10  
void at::native::reduce_kernel<128, 4, at::native::R...         0.00%       0.000us         0.00%       0.000us       0.000us     834.000us         0.03%     834.000us      83.400us            10  
void at::native::elementwise_kernel<128, 4, at::nati...         0.00%       0.000us         0.00%       0.000us       0.000us     778.000us         0.03%     778.000us      77.800us            10  
void at::native::elementwise_kernel<128, 4, at::nati...         0.00%       0.000us         0.00%       0.000us       0.000us     767.000us         0.03%     767.000us      76.700us            10  
void at::native::(anonymous namespace)::vectorized_l...         0.00%       0.000us         0.00%       0.000us       0.000us     751.000us         0.03%     751.000us      75.100us            10  
void at::native::(anonymous namespace)::fused_dropou...         0.00%       0.000us         0.00%       0.000us       0.000us     714.000us         0.03%     714.000us      71.400us            10  
void at::native::vectorized_elementwise_kernel<4, at...         0.00%       0.000us         0.00%       0.000us       0.000us     664.000us         0.03%     664.000us      66.400us            10  
void at::native::vectorized_elementwise_kernel<4, at...         0.00%       0.000us         0.00%       0.000us       0.000us     490.000us         0.02%     490.000us       1.000us           490  
                                        Memset (Device)         0.00%       0.000us         0.00%       0.000us       0.000us     357.000us         0.01%     357.000us       1.190us           300  
cutlass_tensorop_f16_s16816gemm_f16_64x64_64x5_tn_al...         0.00%       0.000us         0.00%       0.000us       0.000us     158.000us         0.01%     158.000us       7.900us            20  
void at_cuda_detail::cub::DeviceRadixSortHistogramKe...         0.00%       0.000us         0.00%       0.000us       0.000us     150.000us         0.01%     150.000us       5.000us            30  
void at_cuda_detail::cub::DeviceUniqueByKeySweepKern...         0.00%       0.000us         0.00%       0.000us       0.000us     143.000us         0.01%     143.000us       4.767us            30  
void at::native::(anonymous namespace)::krn_partial_...         0.00%       0.000us         0.00%       0.000us       0.000us     140.000us         0.01%     140.000us       4.667us            30  
void at_cuda_detail::cub::DeviceScanKernel<at_cuda_d...         0.00%       0.000us         0.00%       0.000us       0.000us      80.000us         0.00%      80.000us       2.667us            30  
void (anonymous namespace)::elementwise_kernel_with_...         0.00%       0.000us         0.00%       0.000us       0.000us      70.000us         0.00%      70.000us       1.750us            40  
void at::native::elementwise_kernel<128, 2, at::nati...         0.00%       0.000us         0.00%       0.000us       0.000us      70.000us         0.00%      70.000us       3.500us            20  
cutlass_tensorop_f16_s16816gemm_f16_64x128_64x3_tt_a...         0.00%       0.000us         0.00%       0.000us       0.000us      70.000us         0.00%      70.000us       7.000us            10  
void at_cuda_detail::cub::DeviceRadixSortExclusiveSu...         0.00%       0.000us         0.00%       0.000us       0.000us      60.000us         0.00%      60.000us       2.000us            30  
void at::native::(anonymous namespace)::krn_partials...         0.00%       0.000us         0.00%       0.000us       0.000us      54.000us         0.00%      54.000us       1.800us            30  
void at::native::reduce_kernel<512, 1, at::native::R...         0.00%       0.000us         0.00%       0.000us       0.000us      50.000us         0.00%      50.000us       5.000us            10  
cutlass_tensorop_f16_s16816gemm_f16_256x64_64x4_nt_a...         0.00%       0.000us         0.00%       0.000us       0.000us      50.000us         0.00%      50.000us       5.000us            10  
void at::native::unrolled_elementwise_kernel<at::nat...         0.00%       0.000us         0.00%       0.000us       0.000us      40.000us         0.00%      40.000us       2.000us            20  
void at::native::vectorized_elementwise_kernel<4, at...         0.00%       0.000us         0.00%       0.000us       0.000us      40.000us         0.00%      40.000us       2.000us            20  
void at::native::reduce_kernel<256, 2, at::native::R...         0.00%       0.000us         0.00%       0.000us       0.000us      40.000us         0.00%      40.000us       4.000us            10  
cutlass_tensorop_f16_s16816gemm_f16_64x128_64x3_nt_a...         0.00%       0.000us         0.00%       0.000us       0.000us      40.000us         0.00%      40.000us       4.000us            10  
void at_cuda_detail::cub::DeviceCompactInitKernel<at...         0.00%       0.000us         0.00%       0.000us       0.000us      36.000us         0.00%      36.000us       1.200us            30  
void at::native::unrolled_elementwise_kernel<at::nat...         0.00%       0.000us         0.00%       0.000us       0.000us      30.000us         0.00%      30.000us       3.000us            10  
void at_cuda_detail::cub::DeviceScanInitKernel<at_cu...         0.00%       0.000us         0.00%       0.000us       0.000us      30.000us         0.00%      30.000us       1.000us            30  
void at::native::(anonymous namespace)::compute_num_...         0.00%       0.000us         0.00%       0.000us       0.000us      30.000us         0.00%      30.000us       1.000us            30  
void at::native::vectorized_elementwise_kernel<4, at...         0.00%       0.000us         0.00%       0.000us       0.000us      26.000us         0.00%      26.000us       1.300us            20  
void at::native::vectorized_elementwise_kernel<4, at...         0.00%       0.000us         0.00%       0.000us       0.000us      20.000us         0.00%      20.000us       2.000us            10  
void at::native::vectorized_elementwise_kernel<4, at...         0.00%       0.000us         0.00%       0.000us       0.000us      10.000us         0.00%      10.000us       1.000us            10  
                                       cudaLaunchKernel        82.27%        2.417s        82.27%        2.417s     502.444us       0.000us         0.00%       0.000us       0.000us          4810  
                                        cudaMemcpyAsync         0.01%     382.000us         0.01%     382.000us       7.640us       0.000us         0.00%       0.000us       0.000us            50  
                                         cudaEventQuery         0.02%     629.000us         0.02%     629.000us       0.998us       0.000us         0.00%       0.000us       0.000us           630  
                                  cudaStreamIsCapturing         0.00%      10.000us         0.00%      10.000us       0.333us       0.000us         0.00%       0.000us       0.000us            30  
                                        cudaGraphLaunch         0.73%      21.419ms         0.73%      21.419ms       2.142ms       0.000us         0.00%       0.000us       0.000us            10  
                                   cudaDriverGetVersion         0.00%       0.000us         0.00%       0.000us       0.000us       0.000us         0.00%       0.000us       0.000us            10  
                                        cudaMemsetAsync         0.09%       2.620ms         0.09%       2.620ms       8.733us       0.000us         0.00%       0.000us       0.000us           300  
                                 cudaDeviceGetAttribute         0.00%      34.000us         0.00%      34.000us       0.227us       0.000us         0.00%       0.000us       0.000us           150  
cudaOccupancyMaxActiveBlocksPerMultiprocessorWithFla...         0.00%      88.000us         0.00%      88.000us       1.467us       0.000us         0.00%       0.000us       0.000us            60  
                                    cudaPeekAtLastError         0.00%       0.000us         0.00%       0.000us       0.000us       0.000us         0.00%       0.000us       0.000us           510  
                                    cudaStreamWaitEvent         0.00%      45.000us         0.00%      45.000us       2.250us       0.000us         0.00%       0.000us       0.000us            20  
                                  cudaDeviceSynchronize        16.87%     495.520ms        16.87%     495.520ms     495.520ms       0.000us         0.00%       0.000us       0.000us             1  
-------------------------------------------------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  
Self CPU time total: 2.938s
Self CUDA time total: 2.491s