-------------------------------------------------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  
                                                   Name    Self CPU %      Self CPU   CPU total %     CPU total  CPU time avg     Self CUDA   Self CUDA %    CUDA total  CUDA time avg    # of Calls  
-------------------------------------------------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  
cutlass_tensorop_f16_s16816gemm_f16_128x128_32x5_tt_...         0.00%       0.000us         0.00%       0.000us       0.000us     443.832ms        17.98%     443.832ms     462.325us           960  
cutlass_tensorop_f16_s16816gemm_f16_128x128_32x3_tn_...         0.00%       0.000us         0.00%       0.000us       0.000us     330.543ms        13.39%     330.543ms     273.176us          1210  
cutlass_tensorop_f16_s16816gemm_f16_128x128_32x4_nt_...         0.00%       0.000us         0.00%       0.000us       0.000us     238.836ms         9.68%     238.836ms     497.575us           480  
                                         softmax_kernel         0.00%       0.000us         0.00%       0.000us       0.000us     201.250ms         8.15%     201.250ms     774.038us           260  
cutlass_tensorop_f16_s16816gemm_f16_128x256_32x3_tn_...         0.00%       0.000us         0.00%       0.000us       0.000us     183.007ms         7.41%     183.007ms     732.028us           250  
cutlass_tensorop_f16_s16816gemm_f16_128x128_32x3_tt_...         0.00%       0.000us         0.00%       0.000us       0.000us     142.476ms         5.77%     142.476ms     195.173us           730  
                                softmax_backward_kernel         0.00%       0.000us         0.00%       0.000us       0.000us     136.891ms         5.55%     136.891ms     570.379us           240  
cutlass_tensorop_f16_s16816gemm_f16_128x128_32x5_nt_...         0.00%       0.000us         0.00%       0.000us       0.000us     133.670ms         5.41%     133.670ms     137.804us           970  
cutlass_tensorop_f16_s16816gemm_f16_256x64_32x4_nt_a...         0.00%       0.000us         0.00%       0.000us       0.000us     124.541ms         5.05%     124.541ms     259.460us           480  
void at::native::elementwise_kernel<128, 4, at::nati...         0.00%       0.000us         0.00%       0.000us       0.000us      96.751ms         3.92%      96.751ms     182.549us           530  
cutlass_tensorop_f16_s16816gemm_f16_128x64_64x3_tt_a...         0.00%       0.000us         0.00%       0.000us       0.000us      63.813ms         2.59%      63.813ms     265.887us           240  
cutlass_tensorop_f16_s16816gemm_f16_64x64_64x5_tt_al...         0.00%       0.000us         0.00%       0.000us       0.000us      62.954ms         2.55%      62.954ms     251.816us           250  
cutlass_tensorop_f16_s16816gemm_f16_128x256_32x3_tt_...         0.00%       0.000us         0.00%       0.000us       0.000us      55.609ms         2.25%      55.609ms     241.778us           230  
                              layernorm_backward_kernel         0.00%       0.000us         0.00%       0.000us       0.000us      50.070ms         2.03%      50.070ms     102.184us           490  
cutlass_tensorop_f16_s16816gemm_f16_128x256_64x3_tt_...         0.00%       0.000us         0.00%       0.000us       0.000us      37.903ms         1.54%      37.903ms       3.790ms            10  
cutlass_tensorop_f16_s16816gemm_f16_128x256_64x3_nt_...         0.00%       0.000us         0.00%       0.000us       0.000us      36.201ms         1.47%      36.201ms       3.620ms            10  
void at::native::reduce_kernel<128, 4, at::native::R...         0.00%       0.000us         0.00%       0.000us       0.000us      35.912ms         1.45%      35.912ms      71.824us           500  
                                       layernorm_kernel         0.00%       0.000us         0.00%       0.000us       0.000us      23.302ms         0.94%      23.302ms      47.555us           490  
void at::native::vectorized_elementwise_kernel<4, at...         0.00%       0.000us         0.00%       0.000us       0.000us      23.117ms         0.94%      23.117ms       5.765us          4010  
void at::native::vectorized_elementwise_kernel<4, at...         0.00%       0.000us         0.00%       0.000us       0.000us      12.325ms         0.50%      12.325ms       2.249us          5480  
void at::native::(anonymous namespace)::indexSelectL...         0.00%       0.000us         0.00%       0.000us       0.000us       5.629ms         0.23%       5.629ms     187.633us            30  
void at::native::vectorized_elementwise_kernel<4, at...         0.00%       0.000us         0.00%       0.000us       0.000us       3.670ms         0.15%       3.670ms       1.000us          3670  
void at::native::(anonymous namespace)::sum_and_scat...         0.00%       0.000us         0.00%       0.000us       0.000us       3.240ms         0.13%       3.240ms     108.000us            30  
void at::native::(anonymous namespace)::compute_grad...         0.00%       0.000us         0.00%       0.000us       0.000us       2.889ms         0.12%       2.889ms      96.300us            30  
cutlass_tensorop_f16_s16816gemm_f16_64x256_32x4_tt_a...         0.00%       0.000us         0.00%       0.000us       0.000us       2.850ms         0.12%       2.850ms     285.000us            10  
                                              memcpy128         0.00%       0.000us         0.00%       0.000us       0.000us       2.675ms         0.11%       2.675ms      53.500us            50  
void at_cuda_detail::cub::DeviceRadixSortOnesweepKer...         0.00%       0.000us         0.00%       0.000us       0.000us       2.318ms         0.09%       2.318ms       9.658us           240  
void at::native::unrolled_elementwise_kernel<at::nat...         0.00%       0.000us         0.00%       0.000us       0.000us       1.916ms         0.08%       1.916ms       3.832us           500  
void at::native::(anonymous namespace)::GammaBetaBac...         0.00%       0.000us         0.00%       0.000us       0.000us       1.207ms         0.05%       1.207ms     120.700us            10  
                                       Memset (Unknown)         0.00%       0.000us         0.00%       0.000us       0.000us     879.000us         0.04%     879.000us       1.758us           500  
void at::native::(anonymous namespace)::layer_norm_g...         0.00%       0.000us         0.00%       0.000us       0.000us     868.000us         0.04%     868.000us      86.800us            10  
CudaCodeGen::kernel3(CudaCodeGen::Tensor<CudaCodeGen...         0.00%       0.000us         0.00%       0.000us       0.000us     826.000us         0.03%     826.000us      82.600us            10  
void at::native::(anonymous namespace)::vectorized_l...         0.00%       0.000us         0.00%       0.000us       0.000us     769.000us         0.03%     769.000us      76.900us            10  
void at::native::(anonymous namespace)::fused_dropou...         0.00%       0.000us         0.00%       0.000us       0.000us     718.000us         0.03%     718.000us      71.800us            10  
void at::native::vectorized_elementwise_kernel<4, at...         0.00%       0.000us         0.00%       0.000us       0.000us     661.000us         0.03%     661.000us      66.100us            10  
CudaCodeGen::kernel2(CudaCodeGen::Tensor<CudaCodeGen...         0.00%       0.000us         0.00%       0.000us       0.000us     651.000us         0.03%     651.000us      65.100us            10  
                         Memcpy DtoD (Device -> Device)         0.00%       0.000us         0.00%       0.000us       0.000us     604.000us         0.02%     604.000us      12.080us            50  
void at::native::vectorized_elementwise_kernel<4, at...         0.00%       0.000us         0.00%       0.000us       0.000us     490.000us         0.02%     490.000us       1.000us           490  
void at::native::vectorized_elementwise_kernel<4, at...         0.00%       0.000us         0.00%       0.000us       0.000us     480.000us         0.02%     480.000us       2.000us           240  
CudaCodeGen::kernel1(CudaCodeGen::Tensor<CudaCodeGen...         0.00%       0.000us         0.00%       0.000us       0.000us     413.000us         0.02%     413.000us      41.300us            10  
                                        Memset (Device)         0.00%       0.000us         0.00%       0.000us       0.000us     351.000us         0.01%     351.000us       1.170us           300  
cutlass_tensorop_f16_s16816gemm_f16_64x64_64x5_tn_al...         0.00%       0.000us         0.00%       0.000us       0.000us     157.000us         0.01%     157.000us       7.850us            20  
void at_cuda_detail::cub::DeviceUniqueByKeySweepKern...         0.00%       0.000us         0.00%       0.000us       0.000us     157.000us         0.01%     157.000us       5.233us            30  
void at::native::(anonymous namespace)::krn_partial_...         0.00%       0.000us         0.00%       0.000us       0.000us     157.000us         0.01%     157.000us       5.233us            30  
void at_cuda_detail::cub::DeviceRadixSortHistogramKe...         0.00%       0.000us         0.00%       0.000us       0.000us     150.000us         0.01%     150.000us       5.000us            30  
void at_cuda_detail::cub::DeviceScanKernel<at_cuda_d...         0.00%       0.000us         0.00%       0.000us       0.000us      80.000us         0.00%      80.000us       2.667us            30  
cutlass_tensorop_f16_s16816gemm_f16_64x128_64x3_tt_a...         0.00%       0.000us         0.00%       0.000us       0.000us      72.000us         0.00%      72.000us       7.200us            10  
void (anonymous namespace)::elementwise_kernel_with_...         0.00%       0.000us         0.00%       0.000us       0.000us      70.000us         0.00%      70.000us       1.750us            40  
void at::native::elementwise_kernel<128, 2, at::nati...         0.00%       0.000us         0.00%       0.000us       0.000us      70.000us         0.00%      70.000us       3.500us            20  
void at_cuda_detail::cub::DeviceRadixSortExclusiveSu...         0.00%       0.000us         0.00%       0.000us       0.000us      60.000us         0.00%      60.000us       2.000us            30  
void at::native::(anonymous namespace)::krn_partials...         0.00%       0.000us         0.00%       0.000us       0.000us      56.000us         0.00%      56.000us       1.867us            30  
void at::native::reduce_kernel<512, 1, at::native::R...         0.00%       0.000us         0.00%       0.000us       0.000us      50.000us         0.00%      50.000us       5.000us            10  
cutlass_tensorop_f16_s16816gemm_f16_256x64_64x4_nt_a...         0.00%       0.000us         0.00%       0.000us       0.000us      50.000us         0.00%      50.000us       5.000us            10  
void at::native::vectorized_elementwise_kernel<4, at...         0.00%       0.000us         0.00%       0.000us       0.000us      46.000us         0.00%      46.000us       1.150us            40  
void at::native::unrolled_elementwise_kernel<at::nat...         0.00%       0.000us         0.00%       0.000us       0.000us      40.000us         0.00%      40.000us       2.000us            20  
void at::native::reduce_kernel<256, 2, at::native::R...         0.00%       0.000us         0.00%       0.000us       0.000us      40.000us         0.00%      40.000us       4.000us            10  
cutlass_tensorop_f16_s16816gemm_f16_64x128_64x3_nt_a...         0.00%       0.000us         0.00%       0.000us       0.000us      40.000us         0.00%      40.000us       4.000us            10  
void at_cuda_detail::cub::DeviceCompactInitKernel<at...         0.00%       0.000us         0.00%       0.000us       0.000us      40.000us         0.00%      40.000us       1.333us            30  
void at::native::unrolled_elementwise_kernel<at::nat...         0.00%       0.000us         0.00%       0.000us       0.000us      30.000us         0.00%      30.000us       3.000us            10  
void at_cuda_detail::cub::DeviceScanInitKernel<at_cu...         0.00%       0.000us         0.00%       0.000us       0.000us      30.000us         0.00%      30.000us       1.000us            30  
void at::native::(anonymous namespace)::compute_num_...         0.00%       0.000us         0.00%       0.000us       0.000us      30.000us         0.00%      30.000us       1.000us            30  
void at::native::vectorized_elementwise_kernel<4, at...         0.00%       0.000us         0.00%       0.000us       0.000us      20.000us         0.00%      20.000us       2.000us            10  
void at::native::unrolled_elementwise_kernel<at::nat...         0.00%       0.000us         0.00%       0.000us       0.000us      20.000us         0.00%      20.000us       2.000us            10  
void at::native::vectorized_elementwise_kernel<4, at...         0.00%       0.000us         0.00%       0.000us       0.000us      10.000us         0.00%      10.000us       1.000us            10  
                                       cudaLaunchKernel        78.74%        1.912s        78.74%        1.912s     397.573us       0.000us         0.00%       0.000us       0.000us          4810  
                                        cudaMemcpyAsync         0.02%     391.000us         0.02%     391.000us       7.820us       0.000us         0.00%       0.000us       0.000us            50  
                                         cudaEventQuery         0.02%     571.000us         0.02%     571.000us       1.005us       0.000us         0.00%       0.000us       0.000us           568  
                                  cudaStreamIsCapturing         0.00%       8.000us         0.00%       8.000us       0.267us       0.000us         0.00%       0.000us       0.000us            30  
                                        cudaGraphLaunch         0.80%      19.331ms         0.80%      19.331ms       1.933ms       0.000us         0.00%       0.000us       0.000us            10  
                                   cudaDriverGetVersion         0.00%       0.000us         0.00%       0.000us       0.000us       0.000us         0.00%       0.000us       0.000us            10  
                                        cudaMemsetAsync         0.11%       2.610ms         0.11%       2.610ms       8.700us       0.000us         0.00%       0.000us       0.000us           300  
                                 cudaDeviceGetAttribute         0.00%      37.000us         0.00%      37.000us       0.247us       0.000us         0.00%       0.000us       0.000us           150  
cudaOccupancyMaxActiveBlocksPerMultiprocessorWithFla...         0.00%      84.000us         0.00%      84.000us       1.400us       0.000us         0.00%       0.000us       0.000us            60  
                                    cudaPeekAtLastError         0.00%       0.000us         0.00%       0.000us       0.000us       0.000us         0.00%       0.000us       0.000us           510  
                                    cudaStreamWaitEvent         0.00%      45.000us         0.00%      45.000us       2.250us       0.000us         0.00%       0.000us       0.000us            20  
                                  cudaDeviceSynchronize        20.31%     493.267ms        20.31%     493.267ms     493.267ms       0.000us         0.00%       0.000us       0.000us             1  
-------------------------------------------------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  
Self CPU time total: 2.429s
Self CUDA time total: 2.469s