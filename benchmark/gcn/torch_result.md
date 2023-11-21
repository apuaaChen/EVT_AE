-------------------------------------------------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  
                                                   Name    Self CPU %      Self CPU   CPU total %     CPU total  CPU time avg     Self CUDA   Self CUDA %    CUDA total  CUDA time avg    # of Calls  
-------------------------------------------------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  
void at::native::(anonymous namespace)::nll_loss_for...         0.00%       0.000us         0.00%       0.000us       0.000us     247.484ms        41.53%     247.484ms      24.748ms            10  
void at::native::(anonymous namespace)::nll_loss_bac...         0.00%       0.000us         0.00%       0.000us       0.000us     127.835ms        21.45%     127.835ms      12.784ms            10  
void cusparse::csrmm_alg2_kernel<cusparse::CsrMMPoli...         0.00%       0.000us         0.00%       0.000us       0.000us      71.434ms        11.99%      71.434ms       1.191ms            60  
void (anonymous namespace)::softmax_warp_backward<fl...         0.00%       0.000us         0.00%       0.000us       0.000us      22.759ms         3.82%      22.759ms       2.276ms            10  
void (anonymous namespace)::softmax_warp_forward<flo...         0.00%       0.000us         0.00%       0.000us       0.000us      14.981ms         2.51%      14.981ms       1.498ms            10  
void at::native::unrolled_elementwise_kernel<at::nat...         0.00%       0.000us         0.00%       0.000us       0.000us      13.415ms         2.25%      13.415ms       1.341ms            10  
void at::native::unrolled_elementwise_kernel<at::nat...         0.00%       0.000us         0.00%       0.000us       0.000us      13.256ms         2.22%      13.256ms       1.326ms            10  
void at::native::vectorized_elementwise_kernel<4, at...         0.00%       0.000us         0.00%       0.000us       0.000us      12.407ms         2.08%      12.407ms     103.392us           120  
                                          memcpy32_post         0.00%       0.000us         0.00%       0.000us       0.000us       8.605ms         1.44%       8.605ms     143.417us            60  
void at::native::reduce_kernel<128, 4, at::native::R...         0.00%       0.000us         0.00%       0.000us       0.000us       7.807ms         1.31%       7.807ms     260.233us            30  
void at::native::vectorized_elementwise_kernel<4, at...         0.00%       0.000us         0.00%       0.000us       0.000us       6.643ms         1.11%       6.643ms     332.150us            20  
ampere_fp16_s1688gemm_fp16_128x128_ldg8_relu_f2f_sta...         0.00%       0.000us         0.00%       0.000us       0.000us       6.568ms         1.10%       6.568ms     656.800us            10  
void cutlass::Kernel<cutlass_80_tensorop_f16_s16816g...         0.00%       0.000us         0.00%       0.000us       0.000us       6.207ms         1.04%       6.207ms     310.350us            20  
ampere_fp16_s16816gemm_fp16_64x128_ldg8_f2f_stages_6...         0.00%       0.000us         0.00%       0.000us       0.000us       4.679ms         0.79%       4.679ms     467.900us            10  
void cusparse::matrix_scalar_multiply_kernel<cuspars...         0.00%       0.000us         0.00%       0.000us       0.000us       4.620ms         0.78%       4.620ms      77.000us            60  
void at::native::vectorized_elementwise_kernel<4, at...         0.00%       0.000us         0.00%       0.000us       0.000us       4.220ms         0.71%       4.220ms      35.167us           120  
void at::native::vectorized_elementwise_kernel<4, at...         0.00%       0.000us         0.00%       0.000us       0.000us       4.197ms         0.70%       4.197ms     209.850us            20  
void at::native::(anonymous namespace)::fused_dropou...         0.00%       0.000us         0.00%       0.000us       0.000us       4.048ms         0.68%       4.048ms     202.400us            20  
void cutlass::Kernel<cutlass_80_tensorop_f16_s16816g...         0.00%       0.000us         0.00%       0.000us       0.000us       3.608ms         0.61%       3.608ms     180.400us            20  
void at::native::vectorized_elementwise_kernel<4, at...         0.00%       0.000us         0.00%       0.000us       0.000us       3.353ms         0.56%       3.353ms     167.650us            20  
void at::native::vectorized_elementwise_kernel<4, at...         0.00%       0.000us         0.00%       0.000us       0.000us       2.694ms         0.45%       2.694ms     134.700us            20  
ampere_fp16_s16816gemm_fp16_128x64_ldg8_f2f_stages_6...         0.00%       0.000us         0.00%       0.000us       0.000us       2.132ms         0.36%       2.132ms     213.200us            10  
ampere_fp16_s16816gemm_fp16_64x64_sliced1x2_ldg8_f2f...         0.00%       0.000us         0.00%       0.000us       0.000us       1.518ms         0.25%       1.518ms     151.800us            10  
void cusparse::csrmm_alg2_partition_kernel<128, long...         0.00%       0.000us         0.00%       0.000us       0.000us       1.198ms         0.20%       1.198ms      19.967us            60  
void splitKreduce_kernel<32, 16, int, __half, __half...         0.00%       0.000us         0.00%       0.000us       0.000us     160.000us         0.03%     160.000us       5.333us            30  
                                       Memset (Unknown)         0.00%       0.000us         0.00%       0.000us       0.000us      41.000us         0.01%      41.000us       1.367us            30  
void at::native::vectorized_elementwise_kernel<4, at...         0.00%       0.000us         0.00%       0.000us       0.000us      24.000us         0.00%      24.000us       1.200us            20  
void at::native::vectorized_elementwise_kernel<4, at...         0.00%       0.000us         0.00%       0.000us       0.000us      20.000us         0.00%      20.000us       1.000us            20  
                                       cudaLaunchKernel         0.09%     535.000us         0.09%     535.000us       6.688us       0.000us         0.00%       0.000us       0.000us            80  
                                  cudaStreamIsCapturing         0.00%       9.000us         0.00%       9.000us       0.450us       0.000us         0.00%       0.000us       0.000us            20  
                                        cudaGraphLaunch         0.21%       1.246ms         0.21%       1.246ms     124.600us       0.000us         0.00%       0.000us       0.000us            10  
                                   cudaDriverGetVersion         0.00%       0.000us         0.00%       0.000us       0.000us       0.000us         0.00%       0.000us       0.000us            10  
                                  cudaDeviceSynchronize        99.70%     592.753ms        99.70%     592.753ms     592.753ms       0.000us         0.00%       0.000us       0.000us             1  
-------------------------------------------------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  
Self CPU time total: 594.543ms
Self CUDA time total: 595.913ms