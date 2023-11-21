-------------------------------------------------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  
                                                   Name    Self CPU %      Self CPU   CPU total %     CPU total  CPU time avg     Self CUDA   Self CUDA %    CUDA total  CUDA time avg    # of Calls  
-------------------------------------------------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  
void cutlass::Kernel<cutlass_80_tensorop_f16_s16816g...         0.00%       0.000us         0.00%       0.000us       0.000us     393.860ms        11.42%     393.860ms     410.271us           960  
ampere_fp16_s16816gemm_fp16_128x128_ldg8_f2f_stages_...         0.00%       0.000us         0.00%       0.000us       0.000us     266.340ms         7.73%     266.340ms     220.116us          1210  
void cutlass::Kernel<cutlass_80_tensorop_f16_s16816g...         0.00%       0.000us         0.00%       0.000us       0.000us     253.561ms         7.35%     253.561ms     209.555us          1210  
void at::native::vectorized_elementwise_kernel<4, at...         0.00%       0.000us         0.00%       0.000us       0.000us     185.657ms         5.39%     185.657ms     386.785us           480  
void at::native::elementwise_kernel<128, 4, at::nati...         0.00%       0.000us         0.00%       0.000us       0.000us     174.919ms         5.07%     174.919ms     603.169us           290  
void at::native::(anonymous namespace)::fused_dropou...         0.00%       0.000us         0.00%       0.000us       0.000us     169.936ms         4.93%     169.936ms     232.789us           730  
ampere_fp16_s16816gemm_fp16_128x128_ldg8_relu_f2f_st...         0.00%       0.000us         0.00%       0.000us       0.000us     163.921ms         4.75%     163.921ms     174.384us           940  
void at::native::vectorized_elementwise_kernel<4, at...         0.00%       0.000us         0.00%       0.000us       0.000us     142.829ms         4.14%     142.829ms     595.121us           240  
void at::native::vectorized_elementwise_kernel<4, at...         0.00%       0.000us         0.00%       0.000us       0.000us     142.713ms         4.14%     142.713ms     195.497us           730  
void (anonymous namespace)::softmax_warp_backward<c1...         0.00%       0.000us         0.00%       0.000us       0.000us     142.165ms         4.12%     142.165ms     592.354us           240  
void at::native::vectorized_elementwise_kernel<4, at...         0.00%       0.000us         0.00%       0.000us       0.000us     131.549ms         3.82%     131.549ms      24.093us          5460  
void cutlass::Kernel<cutlass_80_tensorop_f16_s16816g...         0.00%       0.000us         0.00%       0.000us       0.000us     125.539ms         3.64%     125.539ms     523.079us           240  
ampere_fp16_s16816gemm_fp16_64x128_sliced1x2_ldg8_f2...         0.00%       0.000us         0.00%       0.000us       0.000us     125.416ms         3.64%     125.416ms     261.283us           480  
void cutlass::Kernel<cutlass_80_tensorop_f16_s16816g...         0.00%       0.000us         0.00%       0.000us       0.000us     121.044ms         3.51%     121.044ms     504.350us           240  
void at::native::elementwise_kernel<128, 4, at::nati...         0.00%       0.000us         0.00%       0.000us       0.000us     107.850ms         3.13%     107.850ms     103.702us          1040  
void (anonymous namespace)::softmax_warp_forward<c10...         0.00%       0.000us         0.00%       0.000us       0.000us     103.077ms         2.99%     103.077ms     429.488us           240  
void at::native::reduce_kernel<128, 4, at::native::R...         0.00%       0.000us         0.00%       0.000us       0.000us      96.150ms         2.79%      96.150ms      65.856us          1460  
void at::native::vectorized_elementwise_kernel<4, at...         0.00%       0.000us         0.00%       0.000us       0.000us      70.021ms         2.03%      70.021ms     280.084us           250  
void cutlass::Kernel<cutlass_80_tensorop_f16_s16816g...         0.00%       0.000us         0.00%       0.000us       0.000us      61.676ms         1.79%      61.676ms     256.983us           240  
void at::native::(anonymous namespace)::GammaBetaBac...         0.00%       0.000us         0.00%       0.000us       0.000us      59.809ms         1.73%      59.809ms     119.618us           500  
void cutlass::Kernel<cutlass_80_tensorop_f16_s16816g...         0.00%       0.000us         0.00%       0.000us       0.000us      58.496ms         1.70%      58.496ms     243.733us           240  
void at::native::vectorized_elementwise_kernel<4, at...         0.00%       0.000us         0.00%       0.000us       0.000us      48.561ms         1.41%      48.561ms     194.244us           250  
void at::native::(anonymous namespace)::layer_norm_g...         0.00%       0.000us         0.00%       0.000us       0.000us      45.967ms         1.33%      45.967ms      91.934us           500  
ampere_fp16_s16816gemm_fp16_128x128_ldg8_f2f_stages_...         0.00%       0.000us         0.00%       0.000us       0.000us      43.431ms         1.26%      43.431ms       1.086ms            40  
ampere_fp16_s16816gemm_fp16_256x128_ldg8_f2f_stages_...         0.00%       0.000us         0.00%       0.000us       0.000us      36.849ms         1.07%      36.849ms       3.685ms            10  
void at::native::(anonymous namespace)::vectorized_l...         0.00%       0.000us         0.00%       0.000us       0.000us      36.786ms         1.07%      36.786ms      73.572us           500  
ampere_fp16_s16816gemm_fp16_256x128_ldg8_f2f_stages_...         0.00%       0.000us         0.00%       0.000us       0.000us      36.144ms         1.05%      36.144ms       3.614ms            10  
void at::native::(anonymous namespace)::cunn_SoftMax...         0.00%       0.000us         0.00%       0.000us       0.000us      29.019ms         0.84%      29.019ms       2.902ms            10  
void at::native::(anonymous namespace)::cunn_SoftMax...         0.00%       0.000us         0.00%       0.000us       0.000us      23.011ms         0.67%      23.011ms       2.301ms            10  
void at::native::vectorized_elementwise_kernel<4, at...         0.00%       0.000us         0.00%       0.000us       0.000us      17.695ms         0.51%      17.695ms       4.369us          4050  
                                       Memset (Unknown)         0.00%       0.000us         0.00%       0.000us       0.000us       7.829ms         0.23%       7.829ms       1.804us          4340  
void at::native::(anonymous namespace)::nll_loss_for...         0.00%       0.000us         0.00%       0.000us       0.000us       5.760ms         0.17%       5.760ms     288.000us            20  
void at::native::(anonymous namespace)::indexSelectL...         0.00%       0.000us         0.00%       0.000us       0.000us       5.366ms         0.16%       5.366ms     178.867us            30  
void at::native::(anonymous namespace)::sum_and_scat...         0.00%       0.000us         0.00%       0.000us       0.000us       3.102ms         0.09%       3.102ms     103.400us            30  
void at::native::(anonymous namespace)::nll_loss_bac...         0.00%       0.000us         0.00%       0.000us       0.000us       2.891ms         0.08%       2.891ms     144.550us            20  
void at::native::(anonymous namespace)::compute_grad...         0.00%       0.000us         0.00%       0.000us       0.000us       2.772ms         0.08%       2.772ms      92.400us            30  
void at_cuda_detail::cub::DeviceRadixSortOnesweepKer...         0.00%       0.000us         0.00%       0.000us       0.000us       2.198ms         0.06%       2.198ms       9.158us           240  
                         Memcpy DtoD (Device -> Device)         0.00%       0.000us         0.00%       0.000us       0.000us     603.000us         0.02%     603.000us      12.060us            50  
                                              memcpy128         0.00%       0.000us         0.00%       0.000us       0.000us     550.000us         0.02%     550.000us      55.000us            10  
void at::native::reduce_kernel<512, 1, at::native::R...         0.00%       0.000us         0.00%       0.000us       0.000us     521.000us         0.02%     521.000us      52.100us            10  
                                        Memset (Device)         0.00%       0.000us         0.00%       0.000us       0.000us     345.000us         0.01%     345.000us       1.150us           300  
void at_cuda_detail::cub::DeviceRadixSortHistogramKe...         0.00%       0.000us         0.00%       0.000us       0.000us     150.000us         0.00%     150.000us       5.000us            30  
void at_cuda_detail::cub::DeviceUniqueByKeySweepKern...         0.00%       0.000us         0.00%       0.000us       0.000us     149.000us         0.00%     149.000us       4.967us            30  
void at::native::(anonymous namespace)::krn_partial_...         0.00%       0.000us         0.00%       0.000us       0.000us     140.000us         0.00%     140.000us       4.667us            30  
ampere_fp16_s16816gemm_fp16_64x64_sliced1x2_ldg8_f2f...         0.00%       0.000us         0.00%       0.000us       0.000us     104.000us         0.00%     104.000us      10.400us            10  
ampere_fp16_s16816gemm_fp16_64x64_sliced1x2_ldg8_f2f...         0.00%       0.000us         0.00%       0.000us       0.000us      80.000us         0.00%      80.000us       8.000us            10  
void at_cuda_detail::cub::DeviceScanKernel<at_cuda_d...         0.00%       0.000us         0.00%       0.000us       0.000us      80.000us         0.00%      80.000us       2.667us            30  
void (anonymous namespace)::elementwise_kernel_with_...         0.00%       0.000us         0.00%       0.000us       0.000us      70.000us         0.00%      70.000us       1.750us            40  
void at::native::elementwise_kernel<128, 2, at::nati...         0.00%       0.000us         0.00%       0.000us       0.000us      70.000us         0.00%      70.000us       3.500us            20  
sm80_xmma_gemm_f16f16_f16f32_f32_nt_n_tilesize96x64x...         0.00%       0.000us         0.00%       0.000us       0.000us      70.000us         0.00%      70.000us       7.000us            10  
void at_cuda_detail::cub::DeviceRadixSortExclusiveSu...         0.00%       0.000us         0.00%       0.000us       0.000us      60.000us         0.00%      60.000us       2.000us            30  
void at::native::reduce_kernel<512, 1, at::native::R...         0.00%       0.000us         0.00%       0.000us       0.000us      50.000us         0.00%      50.000us       5.000us            10  
void at::native::(anonymous namespace)::krn_partials...         0.00%       0.000us         0.00%       0.000us       0.000us      50.000us         0.00%      50.000us       1.667us            30  
void cutlass::Kernel<cutlass_75_wmma_tensorop_f16_s1...         0.00%       0.000us         0.00%       0.000us       0.000us      48.000us         0.00%      48.000us       4.800us            10  
void at::native::unrolled_elementwise_kernel<at::nat...         0.00%       0.000us         0.00%       0.000us       0.000us      40.000us         0.00%      40.000us       2.000us            20  
void at_cuda_detail::cub::DeviceCompactInitKernel<at...         0.00%       0.000us         0.00%       0.000us       0.000us      35.000us         0.00%      35.000us       1.167us            30  
void at::native::reduce_kernel<256, 2, at::native::R...         0.00%       0.000us         0.00%       0.000us       0.000us      31.000us         0.00%      31.000us       3.100us            10  
void cutlass::Kernel<cutlass_80_wmma_tensorop_f16_s1...         0.00%       0.000us         0.00%       0.000us       0.000us      30.000us         0.00%      30.000us       3.000us            10  
void at::native::unrolled_elementwise_kernel<at::nat...         0.00%       0.000us         0.00%       0.000us       0.000us      30.000us         0.00%      30.000us       3.000us            10  
void cutlass::Kernel<cutlass_80_wmma_tensorop_f16_s1...         0.00%       0.000us         0.00%       0.000us       0.000us      30.000us         0.00%      30.000us       3.000us            10  
void at_cuda_detail::cub::DeviceScanInitKernel<at_cu...         0.00%       0.000us         0.00%       0.000us       0.000us      30.000us         0.00%      30.000us       1.000us            30  
void at::native::(anonymous namespace)::compute_num_...         0.00%       0.000us         0.00%       0.000us       0.000us      30.000us         0.00%      30.000us       1.000us            30  
void at::native::vectorized_elementwise_kernel<4, at...         0.00%       0.000us         0.00%       0.000us       0.000us      26.000us         0.00%      26.000us       1.300us            20  
void splitKreduce_kernel<32, 16, int, __half, __half...         0.00%       0.000us         0.00%       0.000us       0.000us      22.000us         0.00%      22.000us       2.200us            10  
void at::native::vectorized_elementwise_kernel<4, at...         0.00%       0.000us         0.00%       0.000us       0.000us      20.000us         0.00%      20.000us       2.000us            10  
void at::native::vectorized_elementwise_kernel<4, at...         0.00%       0.000us         0.00%       0.000us       0.000us      20.000us         0.00%      20.000us       2.000us            10  
void at::native::unrolled_elementwise_kernel<at::nat...         0.00%       0.000us         0.00%       0.000us       0.000us      20.000us         0.00%      20.000us       2.000us            10  
void at::native::unrolled_elementwise_kernel<at::nat...         0.00%       0.000us         0.00%       0.000us       0.000us      20.000us         0.00%      20.000us       2.000us            10  
void at::native::vectorized_elementwise_kernel<4, at...         0.00%       0.000us         0.00%       0.000us       0.000us      20.000us         0.00%      20.000us       2.000us            10  
void (anonymous namespace)::softmax_warp_forward<c10...         0.00%       0.000us         0.00%       0.000us       0.000us      18.000us         0.00%      18.000us       1.800us            10  
void at::native::vectorized_elementwise_kernel<4, at...         0.00%       0.000us         0.00%       0.000us       0.000us      10.000us         0.00%      10.000us       1.000us            10  
void at::native::vectorized_elementwise_kernel<4, at...         0.00%       0.000us         0.00%       0.000us       0.000us      10.000us         0.00%      10.000us       1.000us            10  
void at::native::vectorized_elementwise_kernel<4, at...         0.00%       0.000us         0.00%       0.000us       0.000us      10.000us         0.00%      10.000us       1.000us            10  
void at::native::vectorized_elementwise_kernel<4, at...         0.00%       0.000us         0.00%       0.000us       0.000us      10.000us         0.00%      10.000us       1.000us            10  
void (anonymous namespace)::softmax_warp_backward<c1...         0.00%       0.000us         0.00%       0.000us       0.000us      10.000us         0.00%      10.000us       1.000us            10  
                                       cudaLaunchKernel        29.58%        1.009s        29.58%        1.009s     209.737us       0.000us         0.00%       0.000us       0.000us          4810  
                                        cudaMemcpyAsync         0.01%     440.000us         0.01%     440.000us       8.800us       0.000us         0.00%       0.000us       0.000us            50  
                                  cudaStreamIsCapturing         0.00%      11.000us         0.00%      11.000us       0.367us       0.000us         0.00%       0.000us       0.000us            30  
                                        cudaGraphLaunch        56.38%        1.922s        56.38%        1.922s     192.242ms       0.000us         0.00%       0.000us       0.000us            10  
                                   cudaDriverGetVersion         0.00%       0.000us         0.00%       0.000us       0.000us       0.000us         0.00%       0.000us       0.000us            10  
                                        cudaMemsetAsync         0.07%       2.527ms         0.07%       2.527ms       8.423us       0.000us         0.00%       0.000us       0.000us           300  
                                 cudaDeviceGetAttribute         0.00%      32.000us         0.00%      32.000us       0.213us       0.000us         0.00%       0.000us       0.000us           150  
cudaOccupancyMaxActiveBlocksPerMultiprocessorWithFla...         0.00%     105.000us         0.00%     105.000us       1.750us       0.000us         0.00%       0.000us       0.000us            60  
                                    cudaPeekAtLastError         0.00%       0.000us         0.00%       0.000us       0.000us       0.000us         0.00%       0.000us       0.000us           510  
                                  cudaDeviceSynchronize        13.95%     475.685ms        13.95%     475.685ms     475.685ms       0.000us         0.00%       0.000us       0.000us             1  
-------------------------------------------------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  
Self CPU time total: 3.410s
Self CUDA time total: 3.448s