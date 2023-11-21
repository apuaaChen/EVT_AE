-------------------------------------------------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  
                                                   Name    Self CPU %      Self CPU   CPU total %     CPU total  CPU time avg     Self CUDA   Self CUDA %    CUDA total  CUDA time avg    # of Calls  
-------------------------------------------------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  
ampere_fp16_s16816gemm_fp16_128x128_ldg8_relu_f2f_st...         0.00%       0.000us         0.00%       0.000us       0.000us     239.338ms        12.80%     239.338ms     332.414us           720  
ampere_fp16_s16816gemm_fp16_128x128_ldg8_f2f_stages_...         0.00%       0.000us         0.00%       0.000us       0.000us     160.465ms         8.58%     160.465ms     267.442us           600  
ampere_fp16_s16816gemm_fp16_256x128_ldg8_f2f_stages_...         0.00%       0.000us         0.00%       0.000us       0.000us     135.044ms         7.22%     135.044ms     562.683us           240  
void at::native::(anonymous namespace)::fused_dropou...         0.00%       0.000us         0.00%       0.000us       0.000us     133.542ms         7.14%     133.542ms     272.535us           490  
void cutlass::Kernel<cutlass_80_tensorop_f16_s16816g...         0.00%       0.000us         0.00%       0.000us       0.000us      94.903ms         5.07%      94.903ms     395.429us           240  
void at::native::vectorized_elementwise_kernel<4, at...         0.00%       0.000us         0.00%       0.000us       0.000us      91.155ms         4.87%      91.155ms     364.620us           250  
void at::native::elementwise_kernel<128, 4, at::nati...         0.00%       0.000us         0.00%       0.000us       0.000us      83.172ms         4.45%      83.172ms     159.946us           520  
void at::native::vectorized_elementwise_kernel<4, at...         0.00%       0.000us         0.00%       0.000us       0.000us      73.856ms         3.95%      73.856ms     284.062us           260  
void cutlass::Kernel<cutlass_80_tensorop_f16_s16816g...         0.00%       0.000us         0.00%       0.000us       0.000us      71.341ms         3.81%      71.341ms     594.508us           120  
void cutlass::Kernel<cutlass_80_tensorop_f16_s16816g...         0.00%       0.000us         0.00%       0.000us       0.000us      69.041ms         3.69%      69.041ms     143.835us           480  
void at::native::reduce_kernel<128, 4, at::native::R...         0.00%       0.000us         0.00%       0.000us       0.000us      65.469ms         3.50%      65.469ms      87.292us           750  
void at::native::vectorized_elementwise_kernel<4, at...         0.00%       0.000us         0.00%       0.000us       0.000us      63.251ms         3.38%      63.251ms      25.608us          2470  
void cutlass::Kernel<cutlass_80_tensorop_f16_s16816g...         0.00%       0.000us         0.00%       0.000us       0.000us      61.763ms         3.30%      61.763ms     257.346us           240  
void (anonymous namespace)::softmax_warp_backward<c1...         0.00%       0.000us         0.00%       0.000us       0.000us      58.184ms         3.11%      58.184ms     484.867us           120  
void at::native::vectorized_elementwise_kernel<4, at...         0.00%       0.000us         0.00%       0.000us       0.000us      56.454ms         3.02%      56.454ms     470.450us           120  
void at::native::vectorized_elementwise_kernel<4, at...         0.00%       0.000us         0.00%       0.000us       0.000us      56.241ms         3.01%      56.241ms     468.675us           120  
void at::native::(anonymous namespace)::GammaBetaBac...         0.00%       0.000us         0.00%       0.000us       0.000us      54.509ms         2.91%      54.509ms     218.036us           250  
void (anonymous namespace)::softmax_warp_forward<c10...         0.00%       0.000us         0.00%       0.000us       0.000us      53.961ms         2.89%      53.961ms     449.675us           120  
void at::native::vectorized_elementwise_kernel<4, at...         0.00%       0.000us         0.00%       0.000us       0.000us      38.090ms         2.04%      38.090ms     317.417us           120  
void at::native::elementwise_kernel<128, 4, at::nati...         0.00%       0.000us         0.00%       0.000us       0.000us      37.117ms         1.98%      37.117ms     142.758us           260  
void at::native::elementwise_kernel<128, 4, at::nati...         0.00%       0.000us         0.00%       0.000us       0.000us      32.636ms         1.74%      32.636ms     135.983us           240  
void at::native::(anonymous namespace)::vectorized_l...         0.00%       0.000us         0.00%       0.000us       0.000us      31.725ms         1.70%      31.725ms     126.900us           250  
ampere_fp16_s16816gemm_fp16_128x64_ldg8_f2f_stages_3...         0.00%       0.000us         0.00%       0.000us       0.000us      31.281ms         1.67%      31.281ms     240.623us           130  
void at::native::(anonymous namespace)::layer_norm_g...         0.00%       0.000us         0.00%       0.000us       0.000us      30.548ms         1.63%      30.548ms     122.192us           250  
void cutlass::Kernel<cutlass_80_tensorop_f16_s16816g...         0.00%       0.000us         0.00%       0.000us       0.000us      29.151ms         1.56%      29.151ms     242.925us           120  
void at::native::vectorized_elementwise_kernel<4, at...         0.00%       0.000us         0.00%       0.000us       0.000us       3.510ms         0.19%       3.510ms       1.738us          2020  
void cutlass::Kernel<cutlass_80_tensorop_f16_s16816g...         0.00%       0.000us         0.00%       0.000us       0.000us       3.247ms         0.17%       3.247ms     324.700us            10  
void cutlass::Kernel<cutlass_80_tensorop_f16_s16816g...         0.00%       0.000us         0.00%       0.000us       0.000us       2.838ms         0.15%       2.838ms     283.800us            10  
void splitKreduce_kernel<32, 16, int, __half, __half...         0.00%       0.000us         0.00%       0.000us       0.000us       2.817ms         0.15%       2.817ms       5.749us           490  
                                       Memset (Unknown)         0.00%       0.000us         0.00%       0.000us       0.000us       1.675ms         0.09%       1.675ms       1.727us           970  
void at::native::(anonymous namespace)::CatArrayBatc...         0.00%       0.000us         0.00%       0.000us       0.000us       1.589ms         0.08%       1.589ms     158.900us            10  
void at::native::elementwise_kernel<128, 4, at::nati...         0.00%       0.000us         0.00%       0.000us       0.000us     869.000us         0.05%     869.000us      86.900us            10  
                         Memcpy DtoD (Device -> Device)         0.00%       0.000us         0.00%       0.000us       0.000us     643.000us         0.03%     643.000us      32.150us            20  
void at::native::reduce_kernel<128, 4, at::native::R...         0.00%       0.000us         0.00%       0.000us       0.000us     500.000us         0.03%     500.000us      50.000us            10  
                                               memset32         0.00%       0.000us         0.00%       0.000us       0.000us     120.000us         0.01%     120.000us       1.000us           120  
ampere_fp16_s16816gemm_fp16_64x64_sliced1x2_ldg8_f2f...         0.00%       0.000us         0.00%       0.000us       0.000us      80.000us         0.00%      80.000us       8.000us            10  
void cutlass::Kernel<cutlass_80_tensorop_f16_s16816g...         0.00%       0.000us         0.00%       0.000us       0.000us      70.000us         0.00%      70.000us       7.000us            10  
void at::native::(anonymous namespace)::nll_loss_for...         0.00%       0.000us         0.00%       0.000us       0.000us      50.000us         0.00%      50.000us       5.000us            10  
void (anonymous namespace)::softmax_warp_backward<c1...         0.00%       0.000us         0.00%       0.000us       0.000us      50.000us         0.00%      50.000us       5.000us            10  
void (anonymous namespace)::softmax_warp_forward<c10...         0.00%       0.000us         0.00%       0.000us       0.000us      40.000us         0.00%      40.000us       4.000us            10  
void at::native::(anonymous namespace)::nll_loss_bac...         0.00%       0.000us         0.00%       0.000us       0.000us      30.000us         0.00%      30.000us       3.000us            10  
void at::native::vectorized_elementwise_kernel<4, at...         0.00%       0.000us         0.00%       0.000us       0.000us      20.000us         0.00%      20.000us       1.000us            20  
                                       cudaLaunchKernel        48.83%     904.870ms        48.83%     904.870ms     447.955us       0.000us         0.00%       0.000us       0.000us          2020  
                                        cudaMemcpyAsync         0.01%     170.000us         0.01%     170.000us       8.500us       0.000us         0.00%       0.000us       0.000us            20  
                                  cudaStreamIsCapturing         0.00%       2.000us         0.00%       2.000us       0.100us       0.000us         0.00%       0.000us       0.000us            20  
                                        cudaGraphLaunch         0.73%      13.471ms         0.73%      13.471ms       1.347ms       0.000us         0.00%       0.000us       0.000us            10  
                                   cudaDriverGetVersion         0.00%       0.000us         0.00%       0.000us       0.000us       0.000us         0.00%       0.000us       0.000us            10  
                                  cudaDeviceSynchronize        50.43%     934.608ms        50.43%     934.608ms     934.608ms       0.000us         0.00%       0.000us       0.000us             1  
-------------------------------------------------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  
Self CPU time total: 1.853s
Self CUDA time total: 1.870s