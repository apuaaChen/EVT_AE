-------------------------------------------------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  
                                                   Name    Self CPU %      Self CPU   CPU total %     CPU total  CPU time avg     Self CUDA   Self CUDA %    CUDA total  CUDA time avg    # of Calls  
-------------------------------------------------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  
fmha_cutlassB_f16_aligned_64x64_k64_dropout_sm80(Att...         0.00%       0.000us         0.00%       0.000us       0.000us     265.493ms        10.69%     265.493ms       1.106ms           240  
void cutlass::Kernel<cutlass_80_tensorop_f16_s16816g...         0.00%       0.000us         0.00%       0.000us       0.000us     261.955ms        10.55%     261.955ms     545.740us           480  
void cutlass::Kernel<cutlass_80_tensorop_f16_s16816g...         0.00%       0.000us         0.00%       0.000us       0.000us     243.460ms         9.81%     243.460ms     333.507us           730  
void cutlass::Kernel<cutlass_80_tensorop_f16_s16816g...         0.00%       0.000us         0.00%       0.000us       0.000us     222.022ms         8.94%     222.022ms     462.546us           480  
ampere_fp16_s16816gemm_fp16_128x128_ldg8_f2f_stages_...         0.00%       0.000us         0.00%       0.000us       0.000us     174.090ms         7.01%     174.090ms     355.286us           490  
ampere_fp16_s16816gemm_fp16_128x128_ldg8_f2f_stages_...         0.00%       0.000us         0.00%       0.000us       0.000us     161.149ms         6.49%     161.149ms     328.876us           490  
void at::native::elementwise_kernel<128, 4, at::nati...         0.00%       0.000us         0.00%       0.000us       0.000us     146.900ms         5.92%     146.900ms     204.028us           720  
void at::native::elementwise_kernel<128, 4, at::nati...         0.00%       0.000us         0.00%       0.000us       0.000us     132.368ms         5.33%     132.368ms      77.864us          1700  
void cutlass::Kernel<cutlass_80_tensorop_f16_s16816g...         0.00%       0.000us         0.00%       0.000us       0.000us     119.507ms         4.81%     119.507ms     497.946us           240  
fmha_cutlassF_f16_aligned_64x64_rf_sm80(AttentionKer...         0.00%       0.000us         0.00%       0.000us       0.000us     115.170ms         4.64%     115.170ms     479.875us           240  
void at::native::vectorized_elementwise_kernel<4, at...         0.00%       0.000us         0.00%       0.000us       0.000us      71.135ms         2.87%      71.135ms     284.540us           250  
void at::native::(anonymous namespace)::GammaBetaBac...         0.00%       0.000us         0.00%       0.000us       0.000us      58.663ms         2.36%      58.663ms     117.326us           500  
void at::native::(anonymous namespace)::CatArrayBatc...         0.00%       0.000us         0.00%       0.000us       0.000us      58.385ms         2.35%      58.385ms     243.271us           240  
void at::native::vectorized_elementwise_kernel<4, at...         0.00%       0.000us         0.00%       0.000us       0.000us      55.929ms         2.25%      55.929ms      18.159us          3080  
void at::native::vectorized_elementwise_kernel<4, at...         0.00%       0.000us         0.00%       0.000us       0.000us      49.652ms         2.00%      49.652ms     198.608us           250  
void at::native::reduce_kernel<128, 4, at::native::R...         0.00%       0.000us         0.00%       0.000us       0.000us      48.755ms         1.96%      48.755ms      95.598us           510  
void at::native::(anonymous namespace)::layer_norm_g...         0.00%       0.000us         0.00%       0.000us       0.000us      42.202ms         1.70%      42.202ms      84.404us           500  
ampere_fp16_s16816gemm_fp16_256x128_ldg8_f2f_stages_...         0.00%       0.000us         0.00%       0.000us       0.000us      37.344ms         1.50%      37.344ms       3.734ms            10  
void at::native::(anonymous namespace)::vectorized_l...         0.00%       0.000us         0.00%       0.000us       0.000us      36.850ms         1.48%      36.850ms      73.700us           500  
ampere_fp16_s16816gemm_fp16_256x128_ldg8_f2f_stages_...         0.00%       0.000us         0.00%       0.000us       0.000us      36.737ms         1.48%      36.737ms       3.674ms            10  
void at::native::(anonymous namespace)::cunn_SoftMax...         0.00%       0.000us         0.00%       0.000us       0.000us      29.004ms         1.17%      29.004ms       2.900ms            10  
void at::native::vectorized_elementwise_kernel<4, at...         0.00%       0.000us         0.00%       0.000us       0.000us      23.760ms         0.96%      23.760ms      99.000us           240  
void at::native::(anonymous namespace)::cunn_SoftMax...         0.00%       0.000us         0.00%       0.000us       0.000us      22.641ms         0.91%      22.641ms       2.264ms            10  
void at::native::(anonymous namespace)::fused_dropou...         0.00%       0.000us         0.00%       0.000us       0.000us      16.909ms         0.68%      16.909ms      67.636us           250  
void at::native::vectorized_elementwise_kernel<4, at...         0.00%       0.000us         0.00%       0.000us       0.000us      13.033ms         0.52%      13.033ms      52.132us           250  
void at::native::vectorized_elementwise_kernel<4, at...         0.00%       0.000us         0.00%       0.000us       0.000us       8.572ms         0.35%       8.572ms     122.457us            70  
void at::native::(anonymous namespace)::nll_loss_for...         0.00%       0.000us         0.00%       0.000us       0.000us       5.834ms         0.24%       5.834ms     291.700us            20  
void at::native::(anonymous namespace)::indexSelectL...         0.00%       0.000us         0.00%       0.000us       0.000us       5.344ms         0.22%       5.344ms     178.133us            30  
                                        Memset (Device)         0.00%       0.000us         0.00%       0.000us       0.000us       4.740ms         0.19%       4.740ms       1.881us          2520  
void at::native::(anonymous namespace)::sum_and_scat...         0.00%       0.000us         0.00%       0.000us       0.000us       3.088ms         0.12%       3.088ms     102.933us            30  
void at::native::(anonymous namespace)::nll_loss_bac...         0.00%       0.000us         0.00%       0.000us       0.000us       2.947ms         0.12%       2.947ms     147.350us            20  
void at::native::(anonymous namespace)::compute_grad...         0.00%       0.000us         0.00%       0.000us       0.000us       2.752ms         0.11%       2.752ms      91.733us            30  
void at_cuda_detail::cub::DeviceRadixSortOnesweepKer...         0.00%       0.000us         0.00%       0.000us       0.000us       2.116ms         0.09%       2.116ms       8.817us           240  
ampere_fp16_s16816gemm_fp16_128x128_ldg8_relu_f2f_st...         0.00%       0.000us         0.00%       0.000us       0.000us       1.785ms         0.07%       1.785ms     178.500us            10  
                         Memcpy DtoD (Device -> Device)         0.00%       0.000us         0.00%       0.000us       0.000us     556.000us         0.02%     556.000us      55.600us            10  
void at_cuda_detail::cub::DeviceRadixSortHistogramKe...         0.00%       0.000us         0.00%       0.000us       0.000us     150.000us         0.01%     150.000us       5.000us            30  
void at_cuda_detail::cub::DeviceUniqueByKeySweepKern...         0.00%       0.000us         0.00%       0.000us       0.000us     143.000us         0.01%     143.000us       4.767us            30  
void at::native::(anonymous namespace)::krn_partial_...         0.00%       0.000us         0.00%       0.000us       0.000us     140.000us         0.01%     140.000us       4.667us            30  
ampere_fp16_s16816gemm_fp16_64x64_sliced1x2_ldg8_f2f...         0.00%       0.000us         0.00%       0.000us       0.000us      90.000us         0.00%      90.000us       9.000us            10  
ampere_fp16_s16816gemm_fp16_64x64_sliced1x2_ldg8_f2f...         0.00%       0.000us         0.00%       0.000us       0.000us      80.000us         0.00%      80.000us       8.000us            10  
void at_cuda_detail::cub::DeviceScanKernel<at_cuda_d...         0.00%       0.000us         0.00%       0.000us       0.000us      80.000us         0.00%      80.000us       2.667us            30  
void (anonymous namespace)::elementwise_kernel_with_...         0.00%       0.000us         0.00%       0.000us       0.000us      79.000us         0.00%      79.000us       1.975us            40  
void at::native::elementwise_kernel<128, 2, at::nati...         0.00%       0.000us         0.00%       0.000us       0.000us      70.000us         0.00%      70.000us       3.500us            20  
void at::native::reduce_kernel<512, 1, at::native::R...         0.00%       0.000us         0.00%       0.000us       0.000us      60.000us         0.00%      60.000us       6.000us            10  
sm80_xmma_gemm_f16f16_f16f32_f32_nt_n_tilesize96x64x...         0.00%       0.000us         0.00%       0.000us       0.000us      60.000us         0.00%      60.000us       6.000us            10  
void at_cuda_detail::cub::DeviceRadixSortExclusiveSu...         0.00%       0.000us         0.00%       0.000us       0.000us      60.000us         0.00%      60.000us       2.000us            30  
void at::native::(anonymous namespace)::krn_partials...         0.00%       0.000us         0.00%       0.000us       0.000us      52.000us         0.00%      52.000us       1.733us            30  
void cutlass::Kernel<cutlass_75_wmma_tensorop_f16_s1...         0.00%       0.000us         0.00%       0.000us       0.000us      50.000us         0.00%      50.000us       5.000us            10  
void at::native::reduce_kernel<256, 2, at::native::R...         0.00%       0.000us         0.00%       0.000us       0.000us      50.000us         0.00%      50.000us       5.000us            10  
void cutlass::Kernel<cutlass_80_wmma_tensorop_f16_s1...         0.00%       0.000us         0.00%       0.000us       0.000us      40.000us         0.00%      40.000us       4.000us            10  
void at::native::unrolled_elementwise_kernel<at::nat...         0.00%       0.000us         0.00%       0.000us       0.000us      40.000us         0.00%      40.000us       2.000us            20  
void cutlass::Kernel<cutlass_80_wmma_tensorop_f16_s1...         0.00%       0.000us         0.00%       0.000us       0.000us      40.000us         0.00%      40.000us       4.000us            10  
void at_cuda_detail::cub::DeviceCompactInitKernel<at...         0.00%       0.000us         0.00%       0.000us       0.000us      34.000us         0.00%      34.000us       1.133us            30  
void at::native::unrolled_elementwise_kernel<at::nat...         0.00%       0.000us         0.00%       0.000us       0.000us      31.000us         0.00%      31.000us       3.100us            10  
void splitKreduce_kernel<32, 16, int, __half, __half...         0.00%       0.000us         0.00%       0.000us       0.000us      30.000us         0.00%      30.000us       3.000us            10  
void at_cuda_detail::cub::DeviceScanInitKernel<at_cu...         0.00%       0.000us         0.00%       0.000us       0.000us      30.000us         0.00%      30.000us       1.000us            30  
void at::native::(anonymous namespace)::compute_num_...         0.00%       0.000us         0.00%       0.000us       0.000us      30.000us         0.00%      30.000us       1.000us            30  
void at::native::vectorized_elementwise_kernel<4, at...         0.00%       0.000us         0.00%       0.000us       0.000us      20.000us         0.00%      20.000us       2.000us            10  
void at::native::vectorized_elementwise_kernel<4, at...         0.00%       0.000us         0.00%       0.000us       0.000us      20.000us         0.00%      20.000us       2.000us            10  
void at::native::vectorized_elementwise_kernel<4, at...         0.00%       0.000us         0.00%       0.000us       0.000us      20.000us         0.00%      20.000us       2.000us            10  
void (anonymous namespace)::softmax_warp_forward<c10...         0.00%       0.000us         0.00%       0.000us       0.000us      20.000us         0.00%      20.000us       2.000us            10  
void at::native::vectorized_elementwise_kernel<4, at...         0.00%       0.000us         0.00%       0.000us       0.000us      20.000us         0.00%      20.000us       2.000us            10  
void at::native::vectorized_elementwise_kernel<4, at...         0.00%       0.000us         0.00%       0.000us       0.000us      20.000us         0.00%      20.000us       1.000us            20  
void at::native::unrolled_elementwise_kernel<at::nat...         0.00%       0.000us         0.00%       0.000us       0.000us      20.000us         0.00%      20.000us       2.000us            10  
void at::native::unrolled_elementwise_kernel<at::nat...         0.00%       0.000us         0.00%       0.000us       0.000us      20.000us         0.00%      20.000us       2.000us            10  
void at::native::vectorized_elementwise_kernel<4, at...         0.00%       0.000us         0.00%       0.000us       0.000us      20.000us         0.00%      20.000us       2.000us            10  
void at::native::vectorized_elementwise_kernel<4, at...         0.00%       0.000us         0.00%       0.000us       0.000us      10.000us         0.00%      10.000us       1.000us            10  
void at::native::vectorized_elementwise_kernel<4, at...         0.00%       0.000us         0.00%       0.000us       0.000us      10.000us         0.00%      10.000us       1.000us            10  
void (anonymous namespace)::softmax_warp_backward<c1...         0.00%       0.000us         0.00%       0.000us       0.000us      10.000us         0.00%      10.000us       1.000us            10  
                                       cudaLaunchKernel        80.52%        1.635s        80.52%        1.635s     121.536us       0.000us         0.00%       0.000us       0.000us         13450  
                                  cudaStreamIsCapturing         0.02%     499.000us         0.02%     499.000us       1.018us       0.000us         0.00%       0.000us       0.000us           490  
          cudaOccupancyMaxActiveBlocksPerMultiprocessor         0.03%     518.000us         0.03%     518.000us       0.996us       0.000us         0.00%       0.000us       0.000us           520  
cudaOccupancyMaxActiveBlocksPerMultiprocessorWithFla...         0.04%     902.000us         0.04%     902.000us       0.447us       0.000us         0.00%       0.000us       0.000us          2020  
                                   cudaFuncSetAttribute         0.02%     414.000us         0.02%     414.000us       0.191us       0.000us         0.00%       0.000us       0.000us          2170  
                                        cudaMemsetAsync        12.70%     257.833ms        12.70%     257.833ms     102.315us       0.000us         0.00%       0.000us       0.000us          2520  
                                        cudaMemcpyAsync         0.03%     607.000us         0.03%     607.000us      60.700us       0.000us         0.00%       0.000us       0.000us            10  
                                  cudaFuncGetAttributes         0.05%     960.000us         0.05%     960.000us       4.000us       0.000us         0.00%       0.000us       0.000us           240  
                                 cudaDeviceGetAttribute         0.00%       8.000us         0.00%       8.000us       0.053us       0.000us         0.00%       0.000us       0.000us           150  
                                    cudaPeekAtLastError         0.00%       9.000us         0.00%       9.000us       0.018us       0.000us         0.00%       0.000us       0.000us           510  
                                  cudaDeviceSynchronize         6.58%     133.601ms         6.58%     133.601ms     133.601ms       0.000us         0.00%       0.000us       0.000us             1  
-------------------------------------------------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  
Self CPU time total: 2.030s
Self CUDA time total: 2.482s