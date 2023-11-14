-------------------------------------------------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  
                                                   Name    Self CPU %      Self CPU   CPU total %     CPU total  CPU time avg     Self CUDA   Self CUDA %    CUDA total  CUDA time avg    # of Calls  
-------------------------------------------------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  
cutlass_tensorop_f16_s16816gemm_f16_128x128_32x5_tn_...         0.00%       0.000us         0.00%       0.000us       0.000us      41.394ms        16.67%      41.394ms       4.139ms            10  
void cudnn::ops::nchwToNhwcKernel<__half, __half, fl...         0.00%       0.000us         0.00%       0.000us       0.000us      29.879ms        12.03%      29.879ms     248.992us           120  
cutlass_tensorop_f16_s16816gemm_f16_128x128_32x4_nt_...         0.00%       0.000us         0.00%       0.000us       0.000us      28.430ms        11.45%      28.430ms       1.421ms            20  
cutlass_tensorop_f16_s16816gemm_f16_128x256_32x3_tt_...         0.00%       0.000us         0.00%       0.000us       0.000us      27.009ms        10.88%      27.009ms       2.701ms            10  
                         Memcpy DtoD (Device -> Device)         0.00%       0.000us         0.00%       0.000us       0.000us      25.546ms        10.29%      25.546ms       1.277ms            20  
void at::native::elementwise_kernel<128, 4, at::nati...         0.00%       0.000us         0.00%       0.000us       0.000us      19.480ms         7.84%      19.480ms       1.948ms            10  
void at::native::vectorized_elementwise_kernel<4, at...         0.00%       0.000us         0.00%       0.000us       0.000us      15.694ms         6.32%      15.694ms     156.940us           100  
sm80_xmma_wgrad_implicit_gemm_indexed_f16f16_f16f32_...         0.00%       0.000us         0.00%       0.000us       0.000us       8.965ms         3.61%       8.965ms     896.500us            10  
sm80_xmma_fprop_implicit_gemm_indexed_f16f16_f16f32_...         0.00%       0.000us         0.00%       0.000us       0.000us       7.175ms         2.89%       7.175ms     358.750us            20  
void at::native::(anonymous namespace)::fused_dropou...         0.00%       0.000us         0.00%       0.000us       0.000us       6.642ms         2.67%       6.642ms     664.200us            10  
void at::native::(anonymous namespace)::max_pool_bac...         0.00%       0.000us         0.00%       0.000us       0.000us       5.901ms         2.38%       5.901ms     196.700us            30  
void at::native::vectorized_elementwise_kernel<4, at...         0.00%       0.000us         0.00%       0.000us       0.000us       5.811ms         2.34%       5.811ms      36.319us           160  
sm80_xmma_wgrad_implicit_gemm_indexed_f16f16_f16f32_...         0.00%       0.000us         0.00%       0.000us       0.000us       5.399ms         2.17%       5.399ms     539.900us            10  
void at::native::(anonymous namespace)::max_pool_for...         0.00%       0.000us         0.00%       0.000us       0.000us       4.394ms         1.77%       4.394ms     146.467us            30  
_ZN19cutlass_cudnn_train6KernelINS_4conv6kernel23Imp...         0.00%       0.000us         0.00%       0.000us       0.000us       3.471ms         1.40%       3.471ms     347.100us            10  
sm80_xmma_fprop_implicit_gemm_f16f16_f16f32_f32_nhwc...         0.00%       0.000us         0.00%       0.000us       0.000us       3.103ms         1.25%       3.103ms     310.300us            10  
void at::native::elementwise_kernel<128, 4, at::nati...         0.00%       0.000us         0.00%       0.000us       0.000us       1.735ms         0.70%       1.735ms      57.833us            30  
cutlass_tensorop_f16_s16816gemm_f16_128x128_64x4_tn_...         0.00%       0.000us         0.00%       0.000us       0.000us       1.303ms         0.52%       1.303ms     130.300us            10  
cutlass_tensorop_f16_s16816gemm_f16_128x128_32x3_tt_...         0.00%       0.000us         0.00%       0.000us       0.000us       1.240ms         0.50%       1.240ms     124.000us            10  
void at::native::(anonymous namespace)::CatArrayBatc...         0.00%       0.000us         0.00%       0.000us       0.000us       1.201ms         0.48%       1.201ms     120.100us            10  
void at::native::elementwise_kernel<128, 4, at::nati...         0.00%       0.000us         0.00%       0.000us       0.000us       1.099ms         0.44%       1.099ms      36.633us            30  
void at::native::reduce_kernel<512, 1, at::native::R...         0.00%       0.000us         0.00%       0.000us       0.000us     930.000us         0.37%     930.000us      31.000us            30  
void cudnn::ops::nhwcToNchwKernel<__half, __half, fl...         0.00%       0.000us         0.00%       0.000us       0.000us     819.000us         0.33%     819.000us      13.650us            60  
void at::native::vectorized_elementwise_kernel<4, at...         0.00%       0.000us         0.00%       0.000us       0.000us     754.000us         0.30%     754.000us      25.133us            30  
void at::native::vectorized_elementwise_kernel<4, at...         0.00%       0.000us         0.00%       0.000us       0.000us     620.000us         0.25%     620.000us      20.667us            30  
                                        Memset (Device)         0.00%       0.000us         0.00%       0.000us       0.000us     227.000us         0.09%     227.000us       2.837us            80  
void cutlass_cudnn_train::Kernel<cutlass_cudnn_train...         0.00%       0.000us         0.00%       0.000us       0.000us      70.000us         0.03%      70.000us       7.000us            10  
void at::native::vectorized_elementwise_kernel<4, at...         0.00%       0.000us         0.00%       0.000us       0.000us      20.000us         0.01%      20.000us       1.000us            20  
void at::native::vectorized_elementwise_kernel<4, at...         0.00%       0.000us         0.00%       0.000us       0.000us      20.000us         0.01%      20.000us       1.000us            20  
                                       cudaLaunchKernel         0.24%       1.244ms         0.24%       1.244ms      10.367us       0.000us         0.00%       0.000us       0.000us           120  
                                        cudaMemcpyAsync         0.04%     184.000us         0.04%     184.000us       9.200us       0.000us         0.00%       0.000us       0.000us            20  
                                  cudaStreamIsCapturing         0.00%       3.000us         0.00%       3.000us       0.150us       0.000us         0.00%       0.000us       0.000us            20  
                                        cudaGraphLaunch         0.22%       1.151ms         0.22%       1.151ms     115.100us       0.000us         0.00%       0.000us       0.000us            10  
                                   cudaDriverGetVersion         0.00%       0.000us         0.00%       0.000us       0.000us       0.000us         0.00%       0.000us       0.000us            10  
                                  cudaDeviceSynchronize        99.50%     509.061ms        99.50%     509.061ms     509.061ms       0.000us         0.00%       0.000us       0.000us             1  
-------------------------------------------------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  
Self CPU time total: 511.643ms
Self CUDA time total: 248.331ms