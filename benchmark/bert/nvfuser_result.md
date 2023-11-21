-------------------------------------------------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  
                                                   Name    Self CPU %      Self CPU   CPU total %     CPU total  CPU time avg     Self CUDA   Self CUDA %    CUDA total  CUDA time avg    # of Calls  
-------------------------------------------------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  
ampere_fp16_s16816gemm_fp16_128x128_ldg8_f2f_stages_...         0.00%       0.000us         0.00%       0.000us       0.000us     303.186ms         9.52%     303.186ms     248.513us          1220  
CudaCodeGen::kernel7(CudaCodeGen::Tensor<CudaCodeGen...         0.00%       0.000us         0.00%       0.000us       0.000us     286.707ms         9.00%     286.707ms       1.247ms           230  
ampere_fp16_s16816gemm_fp16_128x128_ldg8_f2f_stages_...         0.00%       0.000us         0.00%       0.000us       0.000us     267.141ms         8.39%     267.141ms     220.778us          1210  
void cutlass::Kernel<cutlass_80_tensorop_f16_s16816g...         0.00%       0.000us         0.00%       0.000us       0.000us     254.381ms         7.98%     254.381ms     353.307us           720  
void at::native::elementwise_kernel<128, 4, at::nati...         0.00%       0.000us         0.00%       0.000us       0.000us     183.856ms         5.77%     183.856ms     125.072us          1470  
ampere_fp16_s16816gemm_fp16_256x128_ldg8_f2f_stages_...         0.00%       0.000us         0.00%       0.000us       0.000us     169.816ms         5.33%     169.816ms     679.264us           250  
CudaCodeGen::kernel29(CudaCodeGen::Tensor<CudaCodeGe...         0.00%       0.000us         0.00%       0.000us       0.000us     168.935ms         5.30%     168.935ms     703.896us           240  
sm80_xmma_gemm_f16f16_f16f32_f32_nn_n_tilesize96x128...         0.00%       0.000us         0.00%       0.000us       0.000us     154.238ms         4.84%     154.238ms     164.083us           940  
void cutlass::Kernel<cutlass_80_tensorop_f16_s16816g...         0.00%       0.000us         0.00%       0.000us       0.000us     134.027ms         4.21%     134.027ms     558.446us           240  
ampere_fp16_s16816gemm_fp16_64x128_sliced1x2_ldg8_f2...         0.00%       0.000us         0.00%       0.000us       0.000us     123.853ms         3.89%     123.853ms     258.027us           480  
void cutlass::Kernel<cutlass_80_tensorop_f16_s16816g...         0.00%       0.000us         0.00%       0.000us       0.000us     123.793ms         3.89%     123.793ms     515.804us           240  
void at::native::elementwise_kernel<128, 4, at::nati...         0.00%       0.000us         0.00%       0.000us       0.000us     104.137ms         3.27%     104.137ms     106.262us           980  
CudaCodeGen::kernel24(CudaCodeGen::Tensor<CudaCodeGe...         0.00%       0.000us         0.00%       0.000us       0.000us      69.736ms         2.19%      69.736ms     290.567us           240  
CudaCodeGen::kernel6(CudaCodeGen::Tensor<CudaCodeGen...         0.00%       0.000us         0.00%       0.000us       0.000us      67.272ms         2.11%      67.272ms     146.243us           460  
CudaCodeGen::kernel16(CudaCodeGen::Tensor<CudaCodeGe...         0.00%       0.000us         0.00%       0.000us       0.000us      65.382ms         2.05%      65.382ms      55.408us          1180  
void cutlass::Kernel<cutlass_80_tensorop_f16_s16816g...         0.00%       0.000us         0.00%       0.000us       0.000us      63.420ms         1.99%      63.420ms     264.250us           240  
void cutlass::Kernel<cutlass_80_tensorop_f16_s16816g...         0.00%       0.000us         0.00%       0.000us       0.000us      58.203ms         1.83%      58.203ms     242.512us           240  
CudaCodeGen::kernel25(CudaCodeGen::Tensor<CudaCodeGe...         0.00%       0.000us         0.00%       0.000us       0.000us      52.858ms         1.66%      52.858ms     220.242us           240  
CudaCodeGen::kernel30(CudaCodeGen::Tensor<CudaCodeGe...         0.00%       0.000us         0.00%       0.000us       0.000us      51.957ms         1.63%      51.957ms     225.900us           230  
CudaCodeGen::kernel17(CudaCodeGen::Tensor<CudaCodeGe...         0.00%       0.000us         0.00%       0.000us       0.000us      51.554ms         1.62%      51.554ms      43.690us          1180  
void at::native::vectorized_elementwise_kernel<4, at...         0.00%       0.000us         0.00%       0.000us       0.000us      48.048ms         1.51%      48.048ms     200.200us           240  
CudaCodeGen::kernel1(CudaCodeGen::Tensor<CudaCodeGen...         0.00%       0.000us         0.00%       0.000us       0.000us      37.447ms         1.18%      37.447ms      78.015us           480  
ampere_fp16_s16816gemm_fp16_256x128_ldg8_f2f_stages_...         0.00%       0.000us         0.00%       0.000us       0.000us      35.764ms         1.12%      35.764ms       3.576ms            10  
CudaCodeGen::kernel27(CudaCodeGen::Tensor<float, 3>,...         0.00%       0.000us         0.00%       0.000us       0.000us      34.261ms         1.08%      34.261ms     142.754us           240  
CudaCodeGen::kernel26(CudaCodeGen::Tensor<CudaCodeGe...         0.00%       0.000us         0.00%       0.000us       0.000us      31.478ms         0.99%      31.478ms     131.158us           240  
void at::native::(anonymous namespace)::cunn_SoftMax...         0.00%       0.000us         0.00%       0.000us       0.000us      29.026ms         0.91%      29.026ms       2.903ms            10  
CudaCodeGen::kernel3(CudaCodeGen::Tensor<CudaCodeGen...         0.00%       0.000us         0.00%       0.000us       0.000us      28.941ms         0.91%      28.941ms       2.894ms            10  
CudaCodeGen::kernel28(CudaCodeGen::Tensor<bool, 3>, ...         0.00%       0.000us         0.00%       0.000us       0.000us      25.616ms         0.80%      25.616ms     106.733us           240  
void at::native::vectorized_elementwise_kernel<4, at...         0.00%       0.000us         0.00%       0.000us       0.000us      23.002ms         0.72%      23.002ms       5.722us          4020  
void at::native::(anonymous namespace)::cunn_SoftMax...         0.00%       0.000us         0.00%       0.000us       0.000us      22.809ms         0.72%      22.809ms       2.281ms            10  
CudaCodeGen::kernel2(CudaCodeGen::Tensor<CudaCodeGen...         0.00%       0.000us         0.00%       0.000us       0.000us      19.582ms         0.61%      19.582ms      81.592us           240  
CudaCodeGen::kernel31(CudaCodeGen::Tensor<CudaCodeGe...         0.00%       0.000us         0.00%       0.000us       0.000us      18.039ms         0.57%      18.039ms      78.430us           230  
void at::native::vectorized_elementwise_kernel<4, at...         0.00%       0.000us         0.00%       0.000us       0.000us      17.787ms         0.56%      17.787ms       4.392us          4050  
void at::native::reduce_kernel<128, 4, at::native::R...         0.00%       0.000us         0.00%       0.000us       0.000us       8.468ms         0.27%       8.468ms     282.267us            30  
void at::native::(anonymous namespace)::nll_loss_for...         0.00%       0.000us         0.00%       0.000us       0.000us       5.737ms         0.18%       5.737ms     286.850us            20  
void at::native::(anonymous namespace)::indexSelectL...         0.00%       0.000us         0.00%       0.000us       0.000us       5.436ms         0.17%       5.436ms     181.200us            30  
void cutlass::Kernel<cutlass_80_tensorop_f16_s16816g...         0.00%       0.000us         0.00%       0.000us       0.000us       4.127ms         0.13%       4.127ms     137.567us            30  
void at::native::(anonymous namespace)::sum_and_scat...         0.00%       0.000us         0.00%       0.000us       0.000us       3.136ms         0.10%       3.136ms     104.533us            30  
void at::native::(anonymous namespace)::nll_loss_bac...         0.00%       0.000us         0.00%       0.000us       0.000us       2.857ms         0.09%       2.857ms     142.850us            20  
void at::native::(anonymous namespace)::compute_grad...         0.00%       0.000us         0.00%       0.000us       0.000us       2.795ms         0.09%       2.795ms      93.167us            30  
void at::native::vectorized_elementwise_kernel<4, at...         0.00%       0.000us         0.00%       0.000us       0.000us       2.426ms         0.08%       2.426ms       1.002us          2420  
void at_cuda_detail::cub::DeviceRadixSortOnesweepKer...         0.00%       0.000us         0.00%       0.000us       0.000us       2.293ms         0.07%       2.293ms       9.554us           240  
                                       Memset (Unknown)         0.00%       0.000us         0.00%       0.000us       0.000us       2.069ms         0.06%       2.069ms       1.989us          1040  
CudaCodeGen::kernel5(CudaCodeGen::Tensor<CudaCodeGen...         0.00%       0.000us         0.00%       0.000us       0.000us       1.309ms         0.04%       1.309ms     130.900us            10  
CudaCodeGen::kernel20(CudaCodeGen::Tensor<CudaCodeGe...         0.00%       0.000us         0.00%       0.000us       0.000us       1.225ms         0.04%       1.225ms     122.500us            10  
CudaCodeGen::kernel23(CudaCodeGen::Tensor<bool, 3>, ...         0.00%       0.000us         0.00%       0.000us       0.000us       1.210ms         0.04%       1.210ms     121.000us            10  
CudaCodeGen::kernel8(CudaCodeGen::Tensor<CudaCodeGen...         0.00%       0.000us         0.00%       0.000us       0.000us       1.204ms         0.04%       1.204ms     120.400us            10  
CudaCodeGen::kernel32(CudaCodeGen::Tensor<CudaCodeGe...         0.00%       0.000us         0.00%       0.000us       0.000us       1.197ms         0.04%       1.197ms     119.700us            10  
CudaCodeGen::kernel11(CudaCodeGen::Tensor<CudaCodeGe...         0.00%       0.000us         0.00%       0.000us       0.000us       1.182ms         0.04%       1.182ms     118.200us            10  
void at::native::(anonymous namespace)::GammaBetaBac...         0.00%       0.000us         0.00%       0.000us       0.000us       1.181ms         0.04%       1.181ms     118.100us            10  
CudaCodeGen::kernel15(CudaCodeGen::Tensor<CudaCodeGe...         0.00%       0.000us         0.00%       0.000us       0.000us       1.135ms         0.04%       1.135ms     113.500us            10  
                                              memcpy128         0.00%       0.000us         0.00%       0.000us       0.000us       1.047ms         0.03%       1.047ms      52.350us            20  
CudaCodeGen::kernel21(CudaCodeGen::Tensor<float, 3>,...         0.00%       0.000us         0.00%       0.000us       0.000us       1.019ms         0.03%       1.019ms     101.900us            10  
CudaCodeGen::kernel14(CudaCodeGen::Tensor<float, 3>,...         0.00%       0.000us         0.00%       0.000us       0.000us     885.000us         0.03%     885.000us      88.500us            10  
void at::native::(anonymous namespace)::layer_norm_g...         0.00%       0.000us         0.00%       0.000us       0.000us     859.000us         0.03%     859.000us      85.900us            10  
void at::native::(anonymous namespace)::vectorized_l...         0.00%       0.000us         0.00%       0.000us       0.000us     744.000us         0.02%     744.000us      74.400us            10  
void at::native::(anonymous namespace)::fused_dropou...         0.00%       0.000us         0.00%       0.000us       0.000us     710.000us         0.02%     710.000us      71.000us            10  
CudaCodeGen::kernel22(CudaCodeGen::Tensor<CudaCodeGe...         0.00%       0.000us         0.00%       0.000us       0.000us     672.000us         0.02%     672.000us      67.200us            10  
void at::native::vectorized_elementwise_kernel<4, at...         0.00%       0.000us         0.00%       0.000us       0.000us     661.000us         0.02%     661.000us      66.100us            10  
CudaCodeGen::kernel9(CudaCodeGen::Tensor<CudaCodeGen...         0.00%       0.000us         0.00%       0.000us       0.000us     609.000us         0.02%     609.000us      60.900us            10  
                         Memcpy DtoD (Device -> Device)         0.00%       0.000us         0.00%       0.000us       0.000us     605.000us         0.02%     605.000us      12.100us            50  
void at::native::reduce_kernel<512, 1, at::native::R...         0.00%       0.000us         0.00%       0.000us       0.000us     490.000us         0.02%     490.000us      49.000us            10  
CudaCodeGen::kernel4(CudaCodeGen::Tensor<CudaCodeGen...         0.00%       0.000us         0.00%       0.000us       0.000us     460.000us         0.01%     460.000us      46.000us            10  
                                        Memset (Device)         0.00%       0.000us         0.00%       0.000us       0.000us     358.000us         0.01%     358.000us       1.193us           300  
void at_cuda_detail::cub::DeviceRadixSortHistogramKe...         0.00%       0.000us         0.00%       0.000us       0.000us     150.000us         0.00%     150.000us       5.000us            30  
void at_cuda_detail::cub::DeviceUniqueByKeySweepKern...         0.00%       0.000us         0.00%       0.000us       0.000us     146.000us         0.00%     146.000us       4.867us            30  
void at::native::(anonymous namespace)::krn_partial_...         0.00%       0.000us         0.00%       0.000us       0.000us     140.000us         0.00%     140.000us       4.667us            30  
void cutlass::Kernel<cutlass_80_wmma_tensorop_f16_s1...         0.00%       0.000us         0.00%       0.000us       0.000us      91.000us         0.00%      91.000us       9.100us            10  
ampere_fp16_s16816gemm_fp16_64x64_sliced1x2_ldg8_f2f...         0.00%       0.000us         0.00%       0.000us       0.000us      80.000us         0.00%      80.000us       8.000us            10  
void at_cuda_detail::cub::DeviceScanKernel<at_cuda_d...         0.00%       0.000us         0.00%       0.000us       0.000us      80.000us         0.00%      80.000us       2.667us            30  
void (anonymous namespace)::elementwise_kernel_with_...         0.00%       0.000us         0.00%       0.000us       0.000us      70.000us         0.00%      70.000us       1.750us            40  
void at::native::elementwise_kernel<128, 2, at::nati...         0.00%       0.000us         0.00%       0.000us       0.000us      70.000us         0.00%      70.000us       3.500us            20  
ampere_fp16_s16816gemm_fp16_64x64_sliced1x2_ldg8_f2f...         0.00%       0.000us         0.00%       0.000us       0.000us      70.000us         0.00%      70.000us       7.000us            10  
void at_cuda_detail::cub::DeviceRadixSortExclusiveSu...         0.00%       0.000us         0.00%       0.000us       0.000us      60.000us         0.00%      60.000us       2.000us            30  
void at::native::(anonymous namespace)::krn_partials...         0.00%       0.000us         0.00%       0.000us       0.000us      52.000us         0.00%      52.000us       1.733us            30  
void at::native::reduce_kernel<512, 1, at::native::R...         0.00%       0.000us         0.00%       0.000us       0.000us      50.000us         0.00%      50.000us       5.000us            10  
sm80_xmma_gemm_f16f16_f16f32_f32_nn_n_tilesize96x64x...         0.00%       0.000us         0.00%       0.000us       0.000us      50.000us         0.00%      50.000us       5.000us            10  
void at_cuda_detail::cub::DeviceCompactInitKernel<at...         0.00%       0.000us         0.00%       0.000us       0.000us      37.000us         0.00%      37.000us       1.233us            30  
void at::native::unrolled_elementwise_kernel<at::nat...         0.00%       0.000us         0.00%       0.000us       0.000us      33.000us         0.00%      33.000us       1.650us            20  
void at::native::unrolled_elementwise_kernel<at::nat...         0.00%       0.000us         0.00%       0.000us       0.000us      30.000us         0.00%      30.000us       3.000us            10  
void cutlass::Kernel<cutlass_80_wmma_tensorop_f16_s1...         0.00%       0.000us         0.00%       0.000us       0.000us      30.000us         0.00%      30.000us       3.000us            10  
void cutlass::Kernel<cutlass_80_wmma_tensorop_f16_s1...         0.00%       0.000us         0.00%       0.000us       0.000us      30.000us         0.00%      30.000us       3.000us            10  
CudaCodeGen::kernel18(CudaCodeGen::Tensor<CudaCodeGe...         0.00%       0.000us         0.00%       0.000us       0.000us      30.000us         0.00%      30.000us       3.000us            10  
void at_cuda_detail::cub::DeviceScanInitKernel<at_cu...         0.00%       0.000us         0.00%       0.000us       0.000us      30.000us         0.00%      30.000us       1.000us            30  
void at::native::(anonymous namespace)::compute_num_...         0.00%       0.000us         0.00%       0.000us       0.000us      30.000us         0.00%      30.000us       1.000us            30  
CudaCodeGen::kernel10(CudaCodeGen::Tensor<CudaCodeGe...         0.00%       0.000us         0.00%       0.000us       0.000us      20.000us         0.00%      20.000us       2.000us            10  
void at::native::unrolled_elementwise_kernel<at::nat...         0.00%       0.000us         0.00%       0.000us       0.000us      20.000us         0.00%      20.000us       2.000us            10  
CudaCodeGen::kernel19(CudaCodeGen::Tensor<CudaCodeGe...         0.00%       0.000us         0.00%       0.000us       0.000us      20.000us         0.00%      20.000us       2.000us            10  
void (anonymous namespace)::softmax_warp_forward<c10...         0.00%       0.000us         0.00%       0.000us       0.000us      16.000us         0.00%      16.000us       1.600us            10  
CudaCodeGen::kernel13(CudaCodeGen::Tensor<CudaCodeGe...         0.00%       0.000us         0.00%       0.000us       0.000us      16.000us         0.00%      16.000us       1.600us            10  
void at::native::vectorized_elementwise_kernel<4, at...         0.00%       0.000us         0.00%       0.000us       0.000us      10.000us         0.00%      10.000us       1.000us            10  
void at::native::vectorized_elementwise_kernel<4, at...         0.00%       0.000us         0.00%       0.000us       0.000us      10.000us         0.00%      10.000us       1.000us            10  
void at::native::vectorized_elementwise_kernel<4, at...         0.00%       0.000us         0.00%       0.000us       0.000us      10.000us         0.00%      10.000us       1.000us            10  
void at::native::unrolled_elementwise_kernel<at::nat...         0.00%       0.000us         0.00%       0.000us       0.000us      10.000us         0.00%      10.000us       1.000us            10  
void (anonymous namespace)::softmax_warp_backward<c1...         0.00%       0.000us         0.00%       0.000us       0.000us      10.000us         0.00%      10.000us       1.000us            10  
CudaCodeGen::kernel12(CudaCodeGen::Tensor<CudaCodeGe...         0.00%       0.000us         0.00%       0.000us       0.000us      10.000us         0.00%      10.000us       1.000us            10  
                                       cudaLaunchKernel        33.34%        1.053s        33.34%        1.053s     218.837us       0.000us         0.00%       0.000us       0.000us          4810  
                                        cudaMemcpyAsync         0.01%     382.000us         0.01%     382.000us       7.640us       0.000us         0.00%       0.000us       0.000us            50  
                                  cudaStreamIsCapturing         0.00%       9.000us         0.00%       9.000us       0.300us       0.000us         0.00%       0.000us       0.000us            30  
                                        cudaGraphLaunch        46.83%        1.479s        46.83%        1.479s     147.865ms       0.000us         0.00%       0.000us       0.000us            10  
-------------------------------------------------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  
Self CPU time total: 3.157s
Self CUDA time total: 3.186s