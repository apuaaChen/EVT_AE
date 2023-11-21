-------------------------------------------------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  
                                                   Name    Self CPU %      Self CPU   CPU total %     CPU total  CPU time avg     Self CUDA   Self CUDA %    CUDA total  CUDA time avg    # of Calls  
-------------------------------------------------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  
void cudnn::batchnorm_bwtr_nhwc_semiPersist<__half, ...         0.00%       0.000us         0.00%       0.000us       0.000us      73.223ms        13.12%      73.223ms     292.892us           250  
void at::native::vectorized_elementwise_kernel<4, at...         0.00%       0.000us         0.00%       0.000us       0.000us      64.963ms        11.64%      64.963ms      74.670us           870  
void at::native::vectorized_elementwise_kernel<4, at...         0.00%       0.000us         0.00%       0.000us       0.000us      52.757ms         9.45%      52.757ms     109.910us           480  
void cudnn::batchnorm_fwtr_nhwc_semiPersist<__half, ...         0.00%       0.000us         0.00%       0.000us       0.000us      36.143ms         6.47%      36.143ms      88.154us           410  
void at::native::vectorized_elementwise_kernel<4, at...         0.00%       0.000us         0.00%       0.000us       0.000us      35.236ms         6.31%      35.236ms      71.910us           490  
void cudnn::batchnorm_fwtr_nhwc_semiPersist<__half, ...         0.00%       0.000us         0.00%       0.000us       0.000us      31.514ms         5.65%      31.514ms     262.617us           120  
void cutlass_cudnn_infer::Kernel<cutlass_tensorop_f1...         0.00%       0.000us         0.00%       0.000us       0.000us      21.841ms         3.91%      21.841ms     128.476us           170  
void cudnn::batchnorm_bwtr_nhwc_semiPersist<__half, ...         0.00%       0.000us         0.00%       0.000us       0.000us      19.950ms         3.57%      19.950ms      71.250us           280  
sm80_xmma_fprop_implicit_gemm_f16f16_f16f32_f32_nhwc...         0.00%       0.000us         0.00%       0.000us       0.000us      17.602ms         3.15%      17.602ms     117.347us           150  
sm80_xmma_wgrad_implicit_gemm_indexed_f16f16_f16f32_...         0.00%       0.000us         0.00%       0.000us       0.000us      16.199ms         2.90%      16.199ms     147.264us           110  
void cutlass_cudnn_infer::Kernel<cutlass_tensorop_f1...         0.00%       0.000us         0.00%       0.000us       0.000us      14.689ms         2.63%      14.689ms     183.613us            80  
void at::native::(anonymous namespace)::max_pool_bac...         0.00%       0.000us         0.00%       0.000us       0.000us      14.549ms         2.61%      14.549ms       1.455ms            10  
ampere_fp16_s16816gemm_fp16_128x128_ldg8_f2f_stages_...         0.00%       0.000us         0.00%       0.000us       0.000us      13.761ms         2.47%      13.761ms      98.293us           140  
ampere_fp16_s16816gemm_fp16_128x128_ldg8_f2f_stages_...         0.00%       0.000us         0.00%       0.000us       0.000us      11.897ms         2.13%      11.897ms      91.515us           130  
ampere_fp16_s16816gemm_fp16_128x128_ldg8_f2f_stages_...         0.00%       0.000us         0.00%       0.000us       0.000us       9.943ms         1.78%       9.943ms     165.717us            60  
void cutlass_cudnn_infer::Kernel<cutlass_tensorop_f1...         0.00%       0.000us         0.00%       0.000us       0.000us       9.756ms         1.75%       9.756ms     195.120us            50  
ampere_fp16_s16816gemm_fp16_128x128_ldg8_f2f_stages_...         0.00%       0.000us         0.00%       0.000us       0.000us       8.955ms         1.60%       8.955ms      74.625us           120  
void cutlass_cudnn_infer::Kernel<cutlass_tensorop_f1...         0.00%       0.000us         0.00%       0.000us       0.000us       8.193ms         1.47%       8.193ms     117.043us            70  
void cutlass::Kernel<cutlass_80_tensorop_f16_s16816g...         0.00%       0.000us         0.00%       0.000us       0.000us       8.082ms         1.45%       8.082ms     202.050us            40  
sm80_xmma_dgrad_implicit_gemm_f16f16_f16f32_f32_nhwc...         0.00%       0.000us         0.00%       0.000us       0.000us       7.794ms         1.40%       7.794ms     155.880us            50  
void cutlass::Kernel<cutlass_80_tensorop_f16_s16816g...         0.00%       0.000us         0.00%       0.000us       0.000us       7.784ms         1.39%       7.784ms     194.600us            40  
void at::native::(anonymous namespace)::max_pool_for...         0.00%       0.000us         0.00%       0.000us       0.000us       7.696ms         1.38%       7.696ms     769.600us            10  
sm80_xmma_wgrad_implicit_gemm_indexed_f16f16_f16f32_...         0.00%       0.000us         0.00%       0.000us       0.000us       6.303ms         1.13%       6.303ms     210.100us            30  
void cutlass_cudnn_train::Kernel<cutlass_tensorop_f1...         0.00%       0.000us         0.00%       0.000us       0.000us       5.630ms         1.01%       5.630ms     140.750us            40  
sm80_xmma_fprop_implicit_gemm_f16f16_f16f32_f32_nhwc...         0.00%       0.000us         0.00%       0.000us       0.000us       5.233ms         0.94%       5.233ms     174.433us            30  
void cutlass::Kernel<cutlass_80_tensorop_f16_s16816g...         0.00%       0.000us         0.00%       0.000us       0.000us       5.193ms         0.93%       5.193ms     103.860us            50  
ampere_fp16_s16816gemm_fp16_128x256_ldg8_f2f_stages_...         0.00%       0.000us         0.00%       0.000us       0.000us       4.259ms         0.76%       4.259ms      70.983us            60  
sm80_xmma_gemm_f16f16_f16f32_f32_tn_n_tilesize96x64x...         0.00%       0.000us         0.00%       0.000us       0.000us       3.954ms         0.71%       3.954ms     197.700us            20  
sm80_xmma_wgrad_image_first_layer_f16f16_f32_f32_nhw...         0.00%       0.000us         0.00%       0.000us       0.000us       3.483ms         0.62%       3.483ms     348.300us            10  
void at::native::vectorized_elementwise_kernel<4, at...         0.00%       0.000us         0.00%       0.000us       0.000us       3.471ms         0.62%       3.471ms       3.275us          1060  
sm80_xmma_fprop_image_first_layer_f16f16_f32_f16_nhw...         0.00%       0.000us         0.00%       0.000us       0.000us       3.269ms         0.59%       3.269ms     326.900us            10  
void cutlass_cudnn_infer::Kernel<cutlass_tensorop_f1...         0.00%       0.000us         0.00%       0.000us       0.000us       2.778ms         0.50%       2.778ms     277.800us            10  
void at::native::vectorized_elementwise_kernel<4, at...         0.00%       0.000us         0.00%       0.000us       0.000us       2.622ms         0.47%       2.622ms       4.444us           590  
void at::native::elementwise_kernel<128, 4, at::nati...         0.00%       0.000us         0.00%       0.000us       0.000us       2.477ms         0.44%       2.477ms     123.850us            20  
                                       Memset (Unknown)         0.00%       0.000us         0.00%       0.000us       0.000us       2.004ms         0.36%       2.004ms       1.616us          1240  
void cutlass::Kernel<cutlass_80_tensorop_f16_s16816g...         0.00%       0.000us         0.00%       0.000us       0.000us       1.770ms         0.32%       1.770ms      59.000us            30  
sm80_xmma_gemm_f16f16_f16f32_f32_nt_n_tilesize96x128...         0.00%       0.000us         0.00%       0.000us       0.000us       1.554ms         0.28%       1.554ms      77.700us            20  
void splitKreduce_kernel<32, 16, int, __half, __half...         0.00%       0.000us         0.00%       0.000us       0.000us       1.507ms         0.27%       1.507ms       5.023us           300  
_ZN19cutlass_cudnn_train6KernelINS_4conv6kernel23Imp...         0.00%       0.000us         0.00%       0.000us       0.000us       1.436ms         0.26%       1.436ms     143.600us            10  
ampere_fp16_s16816gemm_fp16_128x256_ldg8_f2f_stages_...         0.00%       0.000us         0.00%       0.000us       0.000us       1.120ms         0.20%       1.120ms      56.000us            20  
void at::native::elementwise_kernel<128, 4, at::nati...         0.00%       0.000us         0.00%       0.000us       0.000us       1.076ms         0.19%       1.076ms     107.600us            10  
void at::native::vectorized_elementwise_kernel<4, at...         0.00%       0.000us         0.00%       0.000us       0.000us       1.060ms         0.19%       1.060ms       1.000us          1060  
ampere_s16816gemm_fp16_64x64_sliced1x2_ldg8_stages_6...         0.00%       0.000us         0.00%       0.000us       0.000us       1.015ms         0.18%       1.015ms     101.500us            10  
                         Memcpy DtoD (Device -> Device)         0.00%       0.000us         0.00%       0.000us       0.000us     834.000us         0.15%     834.000us      41.700us            20  
sm80_xmma_gemm_f16f16_f16f32_f32_tn_n_tilesize96x64x...         0.00%       0.000us         0.00%       0.000us       0.000us     822.000us         0.15%     822.000us      82.200us            10  
sm80_xmma_gemm_f16f16_f16f32_f32_nn_n_tilesize96x64x...         0.00%       0.000us         0.00%       0.000us       0.000us     788.000us         0.14%     788.000us      78.800us            10  
                                                memset8         0.00%       0.000us         0.00%       0.000us       0.000us     701.000us         0.13%     701.000us       5.007us           140  
void at::native::elementwise_kernel<128, 4, at::nati...         0.00%       0.000us         0.00%       0.000us       0.000us     360.000us         0.06%     360.000us      36.000us            10  
void at::native::reduce_kernel<128, 4, at::native::R...         0.00%       0.000us         0.00%       0.000us       0.000us     230.000us         0.04%     230.000us      23.000us            10  
ampere_fp16_s16816gemm_fp16_64x64_ldg8_f2f_stages_64...         0.00%       0.000us         0.00%       0.000us       0.000us      90.000us         0.02%      90.000us       9.000us            10  
void cutlass::Kernel<cutlass_80_tensorop_f16_s16816g...         0.00%       0.000us         0.00%       0.000us       0.000us      90.000us         0.02%      90.000us       9.000us            10  
ampere_fp16_s16816gemm_fp16_256x128_ldg8_f2f_stages_...         0.00%       0.000us         0.00%       0.000us       0.000us      90.000us         0.02%      90.000us       9.000us            10  
                                               memset32         0.00%       0.000us         0.00%       0.000us       0.000us      82.000us         0.01%      82.000us       4.100us            20  
void splitKreduce_kernel<32, 16, int, float, __half,...         0.00%       0.000us         0.00%       0.000us       0.000us      62.000us         0.01%      62.000us       6.200us            10  
void (anonymous namespace)::softmax_warp_backward<c1...         0.00%       0.000us         0.00%       0.000us       0.000us      50.000us         0.01%      50.000us       5.000us            10  
void at::native::reduce_kernel<128, 4, at::native::R...         0.00%       0.000us         0.00%       0.000us       0.000us      50.000us         0.01%      50.000us       5.000us            10  
void cutlass_cudnn_train::Kernel<cutlass_cudnn_train...         0.00%       0.000us         0.00%       0.000us       0.000us      41.000us         0.01%      41.000us       4.100us            10  
void (anonymous namespace)::softmax_warp_forward<c10...         0.00%       0.000us         0.00%       0.000us       0.000us      40.000us         0.01%      40.000us       4.000us            10  
void at::native::(anonymous namespace)::nll_loss_for...         0.00%       0.000us         0.00%       0.000us       0.000us      40.000us         0.01%      40.000us       4.000us            10  
void splitKreduce_kernel<32, 16, int, __half, __half...         0.00%       0.000us         0.00%       0.000us       0.000us      30.000us         0.01%      30.000us       3.000us            10  
void at::native::vectorized_elementwise_kernel<4, at...         0.00%       0.000us         0.00%       0.000us       0.000us      20.000us         0.00%      20.000us       1.000us            20  
void at::native::(anonymous namespace)::nll_loss_bac...         0.00%       0.000us         0.00%       0.000us       0.000us      20.000us         0.00%      20.000us       2.000us            10  
                                          memcpy32_post         0.00%       0.000us         0.00%       0.000us       0.000us      20.000us         0.00%      20.000us       2.000us            10  
                                       cudaLaunchKernel        14.36%      78.614ms        14.36%      78.614ms      48.229us       0.000us         0.00%       0.000us       0.000us          1630  
                                        cudaMemcpyAsync         0.36%       1.972ms         0.36%       1.972ms      98.600us       0.000us         0.00%       0.000us       0.000us            20  
                                  cudaStreamIsCapturing         0.00%       2.000us         0.00%       2.000us       0.100us       0.000us         0.00%       0.000us       0.000us            20  
                                        cudaGraphLaunch        44.01%     241.034ms        44.01%     241.034ms      24.103ms       0.000us         0.00%       0.000us       0.000us            10  
                                   cudaDriverGetVersion         0.00%       0.000us         0.00%       0.000us       0.000us       0.000us         0.00%       0.000us       0.000us            10  
                                  cudaDeviceSynchronize        41.27%     226.016ms        41.27%     226.016ms     226.016ms       0.000us         0.00%       0.000us       0.000us             1  
-------------------------------------------------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  
Self CPU time total: 547.638ms
Self CUDA time total: 558.201ms