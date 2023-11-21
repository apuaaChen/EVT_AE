-------------------------------------------------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  
                                                   Name    Self CPU %      Self CPU   CPU total %     CPU total  CPU time avg     Self CUDA   Self CUDA %    CUDA total  CUDA time avg    # of Calls  
-------------------------------------------------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  
cutlass_tensorop_f16_s16816gemm_f16_128x128_32x5_tn_...         0.00%       0.000us         0.00%       0.000us       0.000us      41.226ms        16.70%      41.226ms       4.123ms            10  
void cudnn::ops::nchwToNhwcKernel<__half, __half, fl...         0.00%       0.000us         0.00%       0.000us       0.000us      29.873ms        12.10%      29.873ms     248.942us           120  
cutlass_tensorop_f16_s16816gemm_f16_128x128_32x4_nt_...         0.00%       0.000us         0.00%       0.000us       0.000us      28.258ms        11.45%      28.258ms       1.413ms            20  
cutlass_tensorop_f16_s16816gemm_f16_128x256_32x3_tt_...         0.00%       0.000us         0.00%       0.000us       0.000us      26.814ms        10.86%      26.814ms       2.681ms            10  
                         Memcpy DtoD (Device -> Device)         0.00%       0.000us         0.00%       0.000us       0.000us      25.539ms        10.34%      25.539ms       1.277ms            20  
void at::native::elementwise_kernel<128, 4, at::nati...         0.00%       0.000us         0.00%       0.000us       0.000us      19.468ms         7.89%      19.468ms       1.947ms            10  
void at::native::vectorized_elementwise_kernel<4, at...         0.00%       0.000us         0.00%       0.000us       0.000us      15.710ms         6.36%      15.710ms     157.100us           100  
sm80_xmma_wgrad_implicit_gemm_indexed_f16f16_f16f32_...         0.00%       0.000us         0.00%       0.000us       0.000us       8.911ms         3.61%       8.911ms     891.100us            10  
sm80_xmma_fprop_implicit_gemm_indexed_f16f16_f16f32_...         0.00%       0.000us         0.00%       0.000us       0.000us       7.158ms         2.90%       7.158ms     357.900us            20  
void at::native::(anonymous namespace)::fused_dropou...         0.00%       0.000us         0.00%       0.000us       0.000us       6.631ms         2.69%       6.631ms     663.100us            10  
void at::native::vectorized_elementwise_kernel<4, at...         0.00%       0.000us         0.00%       0.000us       0.000us       5.799ms         2.35%       5.799ms      36.244us           160  
void at::native::(anonymous namespace)::max_pool_bac...         0.00%       0.000us         0.00%       0.000us       0.000us       5.545ms         2.25%       5.545ms     184.833us            30  
sm80_xmma_wgrad_implicit_gemm_indexed_f16f16_f16f32_...         0.00%       0.000us         0.00%       0.000us       0.000us       5.362ms         2.17%       5.362ms     536.200us            10  
void at::native::(anonymous namespace)::max_pool_for...         0.00%       0.000us         0.00%       0.000us       0.000us       4.373ms         1.77%       4.373ms     145.767us            30  
_ZN19cutlass_cudnn_train6KernelINS_4conv6kernel23Imp...         0.00%       0.000us         0.00%       0.000us       0.000us       3.471ms         1.41%       3.471ms     347.100us            10  
sm80_xmma_fprop_implicit_gemm_f16f16_f16f32_f32_nhwc...         0.00%       0.000us         0.00%       0.000us       0.000us       3.111ms         1.26%       3.111ms     311.100us            10  
void at::native::elementwise_kernel<128, 4, at::nati...         0.00%       0.000us         0.00%       0.000us       0.000us       1.743ms         0.71%       1.743ms      58.100us            30  
cutlass_tensorop_f16_s16816gemm_f16_128x128_64x4_tn_...         0.00%       0.000us         0.00%       0.000us       0.000us       1.286ms         0.52%       1.286ms     128.600us            10  
cutlass_tensorop_f16_s16816gemm_f16_128x128_32x3_tt_...         0.00%       0.000us         0.00%       0.000us       0.000us       1.230ms         0.50%       1.230ms     123.000us            10  
void at::native::(anonymous namespace)::CatArrayBatc...         0.00%       0.000us         0.00%       0.000us       0.000us       1.211ms         0.49%       1.211ms     121.100us            10  
void at::native::elementwise_kernel<128, 4, at::nati...         0.00%       0.000us         0.00%       0.000us       0.000us       1.098ms         0.44%       1.098ms      36.600us            30  
CudaCodeGen::kernel1(CudaCodeGen::Tensor<CudaCodeGen...         0.00%       0.000us         0.00%       0.000us       0.000us     976.000us         0.40%     976.000us      32.533us            30  
void at::native::reduce_kernel<512, 1, at::native::R...         0.00%       0.000us         0.00%       0.000us       0.000us     931.000us         0.38%     931.000us      31.033us            30  
void cudnn::ops::nhwcToNchwKernel<__half, __half, fl...         0.00%       0.000us         0.00%       0.000us       0.000us     812.000us         0.33%     812.000us      13.533us            60  
                                                memset8         0.00%       0.000us         0.00%       0.000us       0.000us     110.000us         0.04%     110.000us       5.500us            20  
                                       Memset (Unknown)         0.00%       0.000us         0.00%       0.000us       0.000us      95.000us         0.04%      95.000us       1.900us            50  
void cutlass_cudnn_train::Kernel<cutlass_cudnn_train...         0.00%       0.000us         0.00%       0.000us       0.000us      80.000us         0.03%      80.000us       8.000us            10  
void at::native::vectorized_elementwise_kernel<4, at...         0.00%       0.000us         0.00%       0.000us       0.000us      20.000us         0.01%      20.000us       1.000us            20  
void at::native::vectorized_elementwise_kernel<4, at...         0.00%       0.000us         0.00%       0.000us       0.000us      20.000us         0.01%      20.000us       1.000us            20  
                                               memset32         0.00%       0.000us         0.00%       0.000us       0.000us      20.000us         0.01%      20.000us       2.000us            10  
                                       cudaLaunchKernel         0.31%     753.000us         0.31%     753.000us       6.275us       0.000us         0.00%       0.000us       0.000us           120  
                                        cudaMemcpyAsync         0.08%     188.000us         0.08%     188.000us       9.400us       0.000us         0.00%       0.000us       0.000us            20  
                                  cudaStreamIsCapturing         0.00%       2.000us         0.00%       2.000us       0.100us       0.000us         0.00%       0.000us       0.000us            20  
                                        cudaGraphLaunch         0.40%     991.000us         0.40%     991.000us      99.100us       0.000us         0.00%       0.000us       0.000us            10  
                                   cudaDriverGetVersion         0.00%       0.000us         0.00%       0.000us       0.000us       0.000us         0.00%       0.000us       0.000us            10  
                                  cudaDeviceSynchronize        99.21%     244.145ms        99.21%     244.145ms     244.145ms       0.000us         0.00%       0.000us       0.000us             1  
-------------------------------------------------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  
Self CPU time total: 246.079ms
Self CUDA time total: 246.881ms