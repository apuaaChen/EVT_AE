-------------------------------------------------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  
                                                   Name    Self CPU %      Self CPU   CPU total %     CPU total  CPU time avg     Self CUDA   Self CUDA %    CUDA total  CUDA time avg    # of Calls  
-------------------------------------------------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  
void cutlass::Kernel<cutlass_80_tensorop_f16_s16816g...         0.00%       0.000us         0.00%       0.000us       0.000us      39.168ms        11.41%      39.168ms     408.000us            96  
ampere_fp16_s16816gemm_fp16_128x128_ldg8_f2f_stages_...         0.00%       0.000us         0.00%       0.000us       0.000us      26.412ms         7.70%      26.412ms     218.281us           121  
void cutlass::Kernel<cutlass_80_tensorop_f16_s16816g...         0.00%       0.000us         0.00%       0.000us       0.000us      25.174ms         7.33%      25.174ms     208.050us           121  
void at::native::vectorized_elementwise_kernel<4, at...         0.00%       0.000us         0.00%       0.000us       0.000us      18.559ms         5.41%      18.559ms     386.646us            48  
void at::native::elementwise_kernel<128, 4, at::nati...         0.00%       0.000us         0.00%       0.000us       0.000us      17.377ms         5.06%      17.377ms     599.207us            29  
void at::native::(anonymous namespace)::fused_dropou...         0.00%       0.000us         0.00%       0.000us       0.000us      16.932ms         4.93%      16.932ms     231.945us            73  
ampere_fp16_s16816gemm_fp16_128x128_ldg8_relu_f2f_st...         0.00%       0.000us         0.00%       0.000us       0.000us      16.276ms         4.74%      16.276ms     173.149us            94  
void at::native::vectorized_elementwise_kernel<4, at...         0.00%       0.000us         0.00%       0.000us       0.000us      14.254ms         4.15%      14.254ms     593.917us            24  
void at::native::vectorized_elementwise_kernel<4, at...         0.00%       0.000us         0.00%       0.000us       0.000us      14.253ms         4.15%      14.253ms     195.247us            73  
void (anonymous namespace)::softmax_warp_backward<c1...         0.00%       0.000us         0.00%       0.000us       0.000us      14.223ms         4.14%      14.223ms     592.625us            24  
void at::native::vectorized_elementwise_kernel<4, at...         0.00%       0.000us         0.00%       0.000us       0.000us      13.097ms         3.82%      13.097ms      23.987us           546  
ampere_fp16_s16816gemm_fp16_64x128_sliced1x2_ldg8_f2...         0.00%       0.000us         0.00%       0.000us       0.000us      12.505ms         3.64%      12.505ms     260.521us            48  
void cutlass::Kernel<cutlass_80_tensorop_f16_s16816g...         0.00%       0.000us         0.00%       0.000us       0.000us      12.462ms         3.63%      12.462ms     519.250us            24  
void cutlass::Kernel<cutlass_80_tensorop_f16_s16816g...         0.00%       0.000us         0.00%       0.000us       0.000us      12.027ms         3.50%      12.027ms     501.125us            24  
void at::native::elementwise_kernel<128, 4, at::nati...         0.00%       0.000us         0.00%       0.000us       0.000us      10.713ms         3.12%      10.713ms     103.010us           104  
void (anonymous namespace)::softmax_warp_forward<c10...         0.00%       0.000us         0.00%       0.000us       0.000us      10.288ms         3.00%      10.288ms     428.667us            24  
void at::native::reduce_kernel<128, 4, at::native::R...         0.00%       0.000us         0.00%       0.000us       0.000us       9.549ms         2.78%       9.549ms      65.404us           146  
void at::native::vectorized_elementwise_kernel<4, at...         0.00%       0.000us         0.00%       0.000us       0.000us       6.974ms         2.03%       6.974ms     278.960us            25  
void cutlass::Kernel<cutlass_80_tensorop_f16_s16816g...         0.00%       0.000us         0.00%       0.000us       0.000us       6.161ms         1.80%       6.161ms     256.708us            24  
void at::native::(anonymous namespace)::GammaBetaBac...         0.00%       0.000us         0.00%       0.000us       0.000us       5.986ms         1.74%       5.986ms     119.720us            50  
void cutlass::Kernel<cutlass_80_tensorop_f16_s16816g...         0.00%       0.000us         0.00%       0.000us       0.000us       5.827ms         1.70%       5.827ms     242.792us            24  
void at::native::vectorized_elementwise_kernel<4, at...         0.00%       0.000us         0.00%       0.000us       0.000us       4.839ms         1.41%       4.839ms     193.560us            25  
void at::native::(anonymous namespace)::layer_norm_g...         0.00%       0.000us         0.00%       0.000us       0.000us       4.684ms         1.36%       4.684ms      93.680us            50  
ampere_fp16_s16816gemm_fp16_128x128_ldg8_f2f_stages_...         0.00%       0.000us         0.00%       0.000us       0.000us       4.307ms         1.25%       4.307ms       1.077ms             4  
void at::native::(anonymous namespace)::vectorized_l...         0.00%       0.000us         0.00%       0.000us       0.000us       3.651ms         1.06%       3.651ms      73.020us            50  
ampere_fp16_s16816gemm_fp16_256x128_ldg8_f2f_stages_...         0.00%       0.000us         0.00%       0.000us       0.000us       3.604ms         1.05%       3.604ms       3.604ms             1  
ampere_fp16_s16816gemm_fp16_256x128_ldg8_f2f_stages_...         0.00%       0.000us         0.00%       0.000us       0.000us       3.533ms         1.03%       3.533ms       3.533ms             1  
void at::native::(anonymous namespace)::cunn_SoftMax...         0.00%       0.000us         0.00%       0.000us       0.000us       2.904ms         0.85%       2.904ms       2.904ms             1  
void at::native::(anonymous namespace)::cunn_SoftMax...         0.00%       0.000us         0.00%       0.000us       0.000us       2.303ms         0.67%       2.303ms       2.303ms             1  
void at::native::vectorized_elementwise_kernel<4, at...         0.00%       0.000us         0.00%       0.000us       0.000us       1.757ms         0.51%       1.757ms       4.338us           405  
                                       Memset (Unknown)         0.00%       0.000us         0.00%       0.000us       0.000us     867.000us         0.25%     867.000us       1.998us           434  
void at::native::(anonymous namespace)::nll_loss_for...         0.00%       0.000us         0.00%       0.000us       0.000us     571.000us         0.17%     571.000us     285.500us             2  
void at::native::(anonymous namespace)::indexSelectL...         0.00%       0.000us         0.00%       0.000us       0.000us     533.000us         0.16%     533.000us     177.667us             3  
void at::native::(anonymous namespace)::sum_and_scat...         0.00%       0.000us         0.00%       0.000us       0.000us     309.000us         0.09%     309.000us     103.000us             3  
void at::native::(anonymous namespace)::nll_loss_bac...         0.00%       0.000us         0.00%       0.000us       0.000us     284.000us         0.08%     284.000us     142.000us             2  
void at::native::(anonymous namespace)::compute_grad...         0.00%       0.000us         0.00%       0.000us       0.000us     277.000us         0.08%     277.000us      92.333us             3  
void at_cuda_detail::cub::DeviceRadixSortOnesweepKer...         0.00%       0.000us         0.00%       0.000us       0.000us     226.000us         0.07%     226.000us       9.417us            24  
                         Memcpy DtoD (Device -> Device)         0.00%       0.000us         0.00%       0.000us       0.000us      60.000us         0.02%      60.000us      12.000us             5  
                                              memcpy128         0.00%       0.000us         0.00%       0.000us       0.000us      55.000us         0.02%      55.000us      55.000us             1  
void at::native::reduce_kernel<512, 1, at::native::R...         0.00%       0.000us         0.00%       0.000us       0.000us      52.000us         0.02%      52.000us      52.000us             1  
                                        Memset (Device)         0.00%       0.000us         0.00%       0.000us       0.000us      36.000us         0.01%      36.000us       1.200us            30  
void at_cuda_detail::cub::DeviceRadixSortHistogramKe...         0.00%       0.000us         0.00%       0.000us       0.000us      15.000us         0.00%      15.000us       5.000us             3  
void at_cuda_detail::cub::DeviceUniqueByKeySweepKern...         0.00%       0.000us         0.00%       0.000us       0.000us      15.000us         0.00%      15.000us       5.000us             3  
void at::native::(anonymous namespace)::krn_partial_...         0.00%       0.000us         0.00%       0.000us       0.000us      14.000us         0.00%      14.000us       4.667us             3  
ampere_fp16_s16816gemm_fp16_64x64_sliced1x2_ldg8_f2f...         0.00%       0.000us         0.00%       0.000us       0.000us      10.000us         0.00%      10.000us      10.000us             1  
ampere_fp16_s16816gemm_fp16_64x64_sliced1x2_ldg8_f2f...         0.00%       0.000us         0.00%       0.000us       0.000us       8.000us         0.00%       8.000us       8.000us             1  
void at_cuda_detail::cub::DeviceScanKernel<at_cuda_d...         0.00%       0.000us         0.00%       0.000us       0.000us       8.000us         0.00%       8.000us       2.667us             3  
void (anonymous namespace)::elementwise_kernel_with_...         0.00%       0.000us         0.00%       0.000us       0.000us       7.000us         0.00%       7.000us       1.750us             4  
void at::native::elementwise_kernel<128, 2, at::nati...         0.00%       0.000us         0.00%       0.000us       0.000us       7.000us         0.00%       7.000us       3.500us             2  
sm80_xmma_gemm_f16f16_f16f32_f32_nt_n_tilesize96x64x...         0.00%       0.000us         0.00%       0.000us       0.000us       6.000us         0.00%       6.000us       6.000us             1  
void at_cuda_detail::cub::DeviceRadixSortExclusiveSu...         0.00%       0.000us         0.00%       0.000us       0.000us       6.000us         0.00%       6.000us       2.000us             3  
void at::native::reduce_kernel<512, 1, at::native::R...         0.00%       0.000us         0.00%       0.000us       0.000us       5.000us         0.00%       5.000us       5.000us             1  
void at::native::(anonymous namespace)::krn_partials...         0.00%       0.000us         0.00%       0.000us       0.000us       5.000us         0.00%       5.000us       1.667us             3  
void at::native::unrolled_elementwise_kernel<at::nat...         0.00%       0.000us         0.00%       0.000us       0.000us       4.000us         0.00%       4.000us       2.000us             2  
void cutlass::Kernel<cutlass_75_wmma_tensorop_f16_s1...         0.00%       0.000us         0.00%       0.000us       0.000us       4.000us         0.00%       4.000us       4.000us             1  
void at_cuda_detail::cub::DeviceCompactInitKernel<at...         0.00%       0.000us         0.00%       0.000us       0.000us       4.000us         0.00%       4.000us       1.333us             3  
void cutlass::Kernel<cutlass_80_wmma_tensorop_f16_s1...         0.00%       0.000us         0.00%       0.000us       0.000us       3.000us         0.00%       3.000us       3.000us             1  
void at::native::unrolled_elementwise_kernel<at::nat...         0.00%       0.000us         0.00%       0.000us       0.000us       3.000us         0.00%       3.000us       3.000us             1  
void cutlass::Kernel<cutlass_80_wmma_tensorop_f16_s1...         0.00%       0.000us         0.00%       0.000us       0.000us       3.000us         0.00%       3.000us       3.000us             1  
void at::native::reduce_kernel<256, 2, at::native::R...         0.00%       0.000us         0.00%       0.000us       0.000us       3.000us         0.00%       3.000us       3.000us             1  
void at_cuda_detail::cub::DeviceScanInitKernel<at_cu...         0.00%       0.000us         0.00%       0.000us       0.000us       3.000us         0.00%       3.000us       1.000us             3  
void at::native::(anonymous namespace)::compute_num_...         0.00%       0.000us         0.00%       0.000us       0.000us       3.000us         0.00%       3.000us       1.000us             3  
void at::native::vectorized_elementwise_kernel<4, at...         0.00%       0.000us         0.00%       0.000us       0.000us       2.000us         0.00%       2.000us       1.000us             2  
void at::native::vectorized_elementwise_kernel<4, at...         0.00%       0.000us         0.00%       0.000us       0.000us       2.000us         0.00%       2.000us       2.000us             1  
void at::native::vectorized_elementwise_kernel<4, at...         0.00%       0.000us         0.00%       0.000us       0.000us       2.000us         0.00%       2.000us       2.000us             1  
void splitKreduce_kernel<32, 16, int, __half, __half...         0.00%       0.000us         0.00%       0.000us       0.000us       2.000us         0.00%       2.000us       2.000us             1  
void at::native::unrolled_elementwise_kernel<at::nat...         0.00%       0.000us         0.00%       0.000us       0.000us       2.000us         0.00%       2.000us       2.000us             1  
void at::native::vectorized_elementwise_kernel<4, at...         0.00%       0.000us         0.00%       0.000us       0.000us       2.000us         0.00%       2.000us       2.000us             1  
void at::native::vectorized_elementwise_kernel<4, at...         0.00%       0.000us         0.00%       0.000us       0.000us       1.000us         0.00%       1.000us       1.000us             1  
void (anonymous namespace)::softmax_warp_forward<c10...         0.00%       0.000us         0.00%       0.000us       0.000us       1.000us         0.00%       1.000us       1.000us             1  
void at::native::vectorized_elementwise_kernel<4, at...         0.00%       0.000us         0.00%       0.000us       0.000us       1.000us         0.00%       1.000us       1.000us             1  
void at::native::vectorized_elementwise_kernel<4, at...         0.00%       0.000us         0.00%       0.000us       0.000us       1.000us         0.00%       1.000us       1.000us             1  
void at::native::vectorized_elementwise_kernel<4, at...         0.00%       0.000us         0.00%       0.000us       0.000us       1.000us         0.00%       1.000us       1.000us             1  
void at::native::unrolled_elementwise_kernel<at::nat...         0.00%       0.000us         0.00%       0.000us       0.000us       1.000us         0.00%       1.000us       1.000us             1  
void (anonymous namespace)::softmax_warp_backward<c1...         0.00%       0.000us         0.00%       0.000us       0.000us       1.000us         0.00%       1.000us       1.000us             1  
                                       cudaLaunchKernel         0.69%       2.370ms         0.69%       2.370ms       4.927us       0.000us         0.00%       0.000us       0.000us           481  
                                        cudaMemcpyAsync         0.02%      59.000us         0.02%      59.000us      11.800us       0.000us         0.00%       0.000us       0.000us             5  
                                  cudaStreamIsCapturing         0.00%       2.000us         0.00%       2.000us       0.667us       0.000us         0.00%       0.000us       0.000us             3  
                                        cudaGraphLaunch         1.20%       4.121ms         1.20%       4.121ms       4.121ms       0.000us         0.00%       0.000us       0.000us             1  
                                   cudaDriverGetVersion         0.00%       0.000us         0.00%       0.000us       0.000us       0.000us         0.00%       0.000us       0.000us             1  
                                        cudaMemsetAsync         0.07%     250.000us         0.07%     250.000us       8.333us       0.000us         0.00%       0.000us       0.000us            30  
                                 cudaDeviceGetAttribute         0.00%       4.000us         0.00%       4.000us       0.267us       0.000us         0.00%       0.000us       0.000us            15  
cudaOccupancyMaxActiveBlocksPerMultiprocessorWithFla...         0.00%      10.000us         0.00%      10.000us       1.667us       0.000us         0.00%       0.000us       0.000us             6  
                                    cudaPeekAtLastError         0.00%       0.000us         0.00%       0.000us       0.000us       0.000us         0.00%       0.000us       0.000us            51  
                                  cudaDeviceSynchronize        98.01%     336.287ms        98.01%     336.287ms     336.287ms       0.000us         0.00%       0.000us       0.000us             1  
-------------------------------------------------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  
Self CPU time total: 343.103ms
Self CUDA time total: 343.224ms