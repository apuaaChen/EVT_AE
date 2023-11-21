-------------------------------------------------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  
                                                   Name    Self CPU %      Self CPU   CPU total %     CPU total  CPU time avg     Self CUDA   Self CUDA %    CUDA total  CUDA time avg    # of Calls  
-------------------------------------------------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  
cutlass_tensorop_f16_s16816gemm_f16_128x128_32x5_tn_...         0.00%       0.000us         0.00%       0.000us       0.000us     167.339ms        10.93%     167.339ms     697.246us           240  
cutlass_tensorop_f16_s16816gemm_f16_128x128_32x3_tt_...         0.00%       0.000us         0.00%       0.000us       0.000us     158.916ms        10.38%     158.916ms     331.075us           480  
cutlass_tensorop_f16_s16816gemm_f16_128x128_32x5_nt_...         0.00%       0.000us         0.00%       0.000us       0.000us     137.591ms         8.99%     137.591ms     229.318us           600  
cutlass_tensorop_f16_s16816gemm_f16_128x128_32x4_tt_...         0.00%       0.000us         0.00%       0.000us       0.000us     130.232ms         8.51%     130.232ms     542.633us           240  
cutlass_tensorop_f16_s16816gemm_f16_128x128_32x5_tt_...         0.00%       0.000us         0.00%       0.000us       0.000us     127.574ms         8.34%     127.574ms     531.558us           240  
void at::native::elementwise_kernel<128, 4, at::nati...         0.00%       0.000us         0.00%       0.000us       0.000us     125.397ms         8.19%     125.397ms     464.433us           270  
cutlass_tensorop_f16_s16816gemm_f16_128x64_32x6_nt_a...         0.00%       0.000us         0.00%       0.000us       0.000us     121.889ms         7.96%     121.889ms     507.871us           240  
cutlass_tensorop_f16_s16816gemm_f16_256x128_32x3_tn_...         0.00%       0.000us         0.00%       0.000us       0.000us      81.789ms         5.34%      81.789ms     681.575us           120  
                                         softmax_kernel         0.00%       0.000us         0.00%       0.000us       0.000us      75.439ms         4.93%      75.439ms     580.300us           130  
cutlass_tensorop_f16_s16816gemm_f16_256x128_32x3_nt_...         0.00%       0.000us         0.00%       0.000us       0.000us      66.330ms         4.33%      66.330ms     552.750us           120  
cutlass_tensorop_f16_s16816gemm_f16_128x128_32x3_tn_...         0.00%       0.000us         0.00%       0.000us       0.000us      65.097ms         4.25%      65.097ms     180.825us           360  
cutlass_tensorop_f16_s16816gemm_f16_128x64_64x3_tt_a...         0.00%       0.000us         0.00%       0.000us       0.000us      59.864ms         3.91%      59.864ms     498.867us           120  
                                softmax_backward_kernel         0.00%       0.000us         0.00%       0.000us       0.000us      55.928ms         3.65%      55.928ms     466.067us           120  
                              layernorm_backward_kernel         0.00%       0.000us         0.00%       0.000us       0.000us      47.912ms         3.13%      47.912ms     191.648us           250  
cutlass_tensorop_f16_s16816gemm_f16_128x64_32x6_tt_a...         0.00%       0.000us         0.00%       0.000us       0.000us      30.923ms         2.02%      30.923ms     257.692us           120  
void at::native::reduce_kernel<128, 4, at::native::R...         0.00%       0.000us         0.00%       0.000us       0.000us      20.235ms         1.32%      20.235ms      74.944us           270  
                                       layernorm_kernel         0.00%       0.000us         0.00%       0.000us       0.000us      17.873ms         1.17%      17.873ms      71.492us           250  
CudaCodeGen::kernel2(CudaCodeGen::Tensor<float, 3>, ...         0.00%       0.000us         0.00%       0.000us       0.000us      14.170ms         0.93%      14.170ms     118.083us           120  
void at::native::vectorized_elementwise_kernel<4, at...         0.00%       0.000us         0.00%       0.000us       0.000us       6.159ms         0.40%       6.159ms       3.095us          1990  
void at::native::vectorized_elementwise_kernel<4, at...         0.00%       0.000us         0.00%       0.000us       0.000us       4.230ms         0.28%       4.230ms       1.544us          2740  
                                              memcpy128         0.00%       0.000us         0.00%       0.000us       0.000us       2.187ms         0.14%       2.187ms      72.900us            30  
void at::native::vectorized_elementwise_kernel<4, at...         0.00%       0.000us         0.00%       0.000us       0.000us       2.184ms         0.14%       2.184ms       1.002us          2180  
cutlass_tensorop_f16_s16816gemm_f16_256x128_32x3_tn_...         0.00%       0.000us         0.00%       0.000us       0.000us       1.975ms         0.13%       1.975ms     197.500us            10  
CudaCodeGen::kernel1(CudaCodeGen::Tensor<CudaCodeGen...         0.00%       0.000us         0.00%       0.000us       0.000us       1.760ms         0.12%       1.760ms     176.000us            10  
void at::native::(anonymous namespace)::CatArrayBatc...         0.00%       0.000us         0.00%       0.000us       0.000us       1.586ms         0.10%       1.586ms     158.600us            10  
cutlass_tensorop_f16_s16816gemm_f16_128x128_32x3_nt_...         0.00%       0.000us         0.00%       0.000us       0.000us       1.445ms         0.09%       1.445ms     144.500us            10  
                                       Memset (Unknown)         0.00%       0.000us         0.00%       0.000us       0.000us     913.000us         0.06%     913.000us       3.652us           250  
void at::native::unrolled_elementwise_kernel<at::nat...         0.00%       0.000us         0.00%       0.000us       0.000us     834.000us         0.05%     834.000us       3.336us           250  
CudaCodeGen::kernel4(CudaCodeGen::Tensor<CudaCodeGen...         0.00%       0.000us         0.00%       0.000us       0.000us     829.000us         0.05%     829.000us      82.900us            10  
                         Memcpy DtoD (Device -> Device)         0.00%       0.000us         0.00%       0.000us       0.000us     643.000us         0.04%     643.000us      32.150us            20  
void at::native::reduce_kernel<128, 4, at::native::R...         0.00%       0.000us         0.00%       0.000us       0.000us     494.000us         0.03%     494.000us      49.400us            10  
void at::native::vectorized_elementwise_kernel<4, at...         0.00%       0.000us         0.00%       0.000us       0.000us     250.000us         0.02%     250.000us       1.000us           250  
cutlass_tensorop_f16_s16816gemm_f16_64x64_64x5_tt_al...         0.00%       0.000us         0.00%       0.000us       0.000us     100.000us         0.01%     100.000us      10.000us            10  
cutlass_tensorop_f16_s16816gemm_f16_64x64_64x5_tn_al...         0.00%       0.000us         0.00%       0.000us       0.000us      70.000us         0.00%      70.000us       7.000us            10  
void at::native::elementwise_kernel<128, 4, at::nati...         0.00%       0.000us         0.00%       0.000us       0.000us      70.000us         0.00%      70.000us       7.000us            10  
cutlass_tensorop_f16_s16816gemm_f16_64x128_32x6_nt_a...         0.00%       0.000us         0.00%       0.000us       0.000us      50.000us         0.00%      50.000us       5.000us            10  
void at::native::vectorized_elementwise_kernel<4, at...         0.00%       0.000us         0.00%       0.000us       0.000us      22.000us         0.00%      22.000us       1.100us            20  
void at::native::vectorized_elementwise_kernel<4, at...         0.00%       0.000us         0.00%       0.000us       0.000us      20.000us         0.00%      20.000us       1.000us            20  
CudaCodeGen::kernel3(CudaCodeGen::Tensor<CudaCodeGen...         0.00%       0.000us         0.00%       0.000us       0.000us      20.000us         0.00%      20.000us       2.000us            10  
                                       cudaLaunchKernel        58.89%     896.618ms        58.89%     896.618ms     443.870us       0.000us         0.00%       0.000us       0.000us          2020  
                                        cudaMemcpyAsync         0.01%     159.000us         0.01%     159.000us       7.950us       0.000us         0.00%       0.000us       0.000us            20  
                                  cudaStreamIsCapturing         0.00%       2.000us         0.00%       2.000us       0.100us       0.000us         0.00%       0.000us       0.000us            20  
                                        cudaGraphLaunch         0.67%      10.189ms         0.67%      10.189ms       1.019ms       0.000us         0.00%       0.000us       0.000us            10  
                                   cudaDriverGetVersion         0.00%       0.000us         0.00%       0.000us       0.000us       0.000us         0.00%       0.000us       0.000us            10  
                                  cudaDeviceSynchronize        40.43%     615.616ms        40.43%     615.616ms     615.616ms       0.000us         0.00%       0.000us       0.000us             1  
-------------------------------------------------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  
Self CPU time total: 1.523s
Self CUDA time total: 1.530s