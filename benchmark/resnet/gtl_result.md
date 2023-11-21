-------------------------------------------------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  
                                                   Name    Self CPU %      Self CPU   CPU total %     CPU total  CPU time avg     Self CUDA   Self CUDA %    CUDA total  CUDA time avg    # of Calls  
-------------------------------------------------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  
cutlass_sm80_tensorop_f16_s16x8x16dgrad_optimized_f1...         0.00%       0.000us         0.00%       0.000us       0.000us      38.981ms         8.57%      38.981ms     299.854us           130  
cutlass_sm80_tensorop_f16_s16x8x16fprop_optimized_f1...         0.00%       0.000us         0.00%       0.000us       0.000us      32.297ms         7.10%      32.297ms     146.805us           220  
cutlass_sm80_tensorop_f16_s16x8x16dgrad_optimized_f1...         0.00%       0.000us         0.00%       0.000us       0.000us      20.860ms         4.59%      20.860ms     130.375us           160  
CudaCodeGen::kernel34(CudaCodeGen::Tensor<float, 1>,...         0.00%       0.000us         0.00%       0.000us       0.000us      18.365ms         4.04%      18.365ms     459.125us            40  
cutlass_sm80_tensorop_f16_s16x8x16dgrad_optimized_f1...         0.00%       0.000us         0.00%       0.000us       0.000us      16.820ms         3.70%      16.820ms     210.250us            80  
sm80_xmma_wgrad_implicit_gemm_indexed_f16f16_f16f32_...         0.00%       0.000us         0.00%       0.000us       0.000us      15.970ms         3.51%      15.970ms     145.182us           110  
cutlass_sm80_tensorop_f16_s16x8x16fprop_optimized_f1...         0.00%       0.000us         0.00%       0.000us       0.000us      15.026ms         3.30%      15.026ms     136.600us           110  
void at::native::(anonymous namespace)::max_pool_bac...         0.00%       0.000us         0.00%       0.000us       0.000us      14.533ms         3.20%      14.533ms       1.453ms            10  
void at::native::elementwise_kernel<128, 4, at::nati...         0.00%       0.000us         0.00%       0.000us       0.000us      13.492ms         2.97%      13.492ms     337.300us            40  
cutlass_sm80_tensorop_f16_s16x8x16dgrad_optimized_f1...         0.00%       0.000us         0.00%       0.000us       0.000us      12.821ms         2.82%      12.821ms     256.420us            50  
CudaCodeGen::kernel30(CudaCodeGen::Tensor<float, 1>,...         0.00%       0.000us         0.00%       0.000us       0.000us      12.261ms         2.70%      12.261ms     245.220us            50  
CudaCodeGen::kernel5(CudaCodeGen::Tensor<CudaCodeGen...         0.00%       0.000us         0.00%       0.000us       0.000us      10.768ms         2.37%      10.768ms     538.400us            20  
cutlass_sm80_tensorop_f16_s16x8x16fprop_optimized_f1...         0.00%       0.000us         0.00%       0.000us       0.000us      10.394ms         2.29%      10.394ms     173.233us            60  
CudaCodeGen::kernel9(CudaCodeGen::Tensor<CudaCodeGen...         0.00%       0.000us         0.00%       0.000us       0.000us       9.790ms         2.15%       9.790ms     244.750us            40  
CudaCodeGen::kernel39(CudaCodeGen::Tensor<float, 1>,...         0.00%       0.000us         0.00%       0.000us       0.000us       9.189ms         2.02%       9.189ms     918.900us            10  
ampere_fp16_s16816gemm_fp16_128x128_ldg8_f2f_stages_...         0.00%       0.000us         0.00%       0.000us       0.000us       9.162ms         2.01%       9.162ms     152.700us            60  
CudaCodeGen::kernel36(CudaCodeGen::Tensor<CudaCodeGe...         0.00%       0.000us         0.00%       0.000us       0.000us       9.117ms         2.00%       9.117ms     455.850us            20  
ampere_fp16_s16816gemm_fp16_128x128_ldg8_f2f_stages_...         0.00%       0.000us         0.00%       0.000us       0.000us       8.562ms         1.88%       8.562ms      71.350us           120  
CudaCodeGen::kernel27(CudaCodeGen::Tensor<float, 1>,...         0.00%       0.000us         0.00%       0.000us       0.000us       8.131ms         1.79%       8.131ms     116.157us            70  
void cutlass::Kernel<cutlass_80_tensorop_f16_s16816g...         0.00%       0.000us         0.00%       0.000us       0.000us       7.763ms         1.71%       7.763ms     194.075us            40  
void at::native::(anonymous namespace)::max_pool_for...         0.00%       0.000us         0.00%       0.000us       0.000us       7.692ms         1.69%       7.692ms     769.200us            10  
CudaCodeGen::kernel2(CudaCodeGen::Tensor<CudaCodeGen...         0.00%       0.000us         0.00%       0.000us       0.000us       7.511ms         1.65%       7.511ms     125.183us            60  
cutlass_sm80_tensorop_f16_s16x8x16fprop_fixed_channe...         0.00%       0.000us         0.00%       0.000us       0.000us       7.304ms         1.61%       7.304ms     730.400us            10  
CudaCodeGen::kernel35(CudaCodeGen::Tensor<float, 1>,...         0.00%       0.000us         0.00%       0.000us       0.000us       7.204ms         1.58%       7.204ms     120.067us            60  
cutlass_sm80_tensorop_f16_s16x8x16fprop_optimized_f1...         0.00%       0.000us         0.00%       0.000us       0.000us       6.294ms         1.38%       6.294ms      89.914us            70  
sm80_xmma_wgrad_implicit_gemm_indexed_f16f16_f16f32_...         0.00%       0.000us         0.00%       0.000us       0.000us       6.264ms         1.38%       6.264ms     208.800us            30  
void cutlass_cudnn_train::Kernel<cutlass_tensorop_f1...         0.00%       0.000us         0.00%       0.000us       0.000us       5.492ms         1.21%       5.492ms     137.300us            40  
CudaCodeGen::kernel4(CudaCodeGen::Tensor<CudaCodeGen...         0.00%       0.000us         0.00%       0.000us       0.000us       5.450ms         1.20%       5.450ms     545.000us            10  
void cutlass::Kernel<cutlass_80_tensorop_f16_s16816g...         0.00%       0.000us         0.00%       0.000us       0.000us       5.318ms         1.17%       5.318ms     106.360us            50  
CudaCodeGen::kernel1(CudaCodeGen::Tensor<CudaCodeGen...         0.00%       0.000us         0.00%       0.000us       0.000us       4.944ms         1.09%       4.944ms     494.400us            10  
cutlass_sm80_tensorop_f16_s16x8x16dgrad_optimized_f1...         0.00%       0.000us         0.00%       0.000us       0.000us       4.881ms         1.07%       4.881ms      81.350us            60  
CudaCodeGen::kernel14(CudaCodeGen::Tensor<CudaCodeGe...         0.00%       0.000us         0.00%       0.000us       0.000us       4.755ms         1.05%       4.755ms     118.875us            40  
CudaCodeGen::kernel32(CudaCodeGen::Tensor<CudaCodeGe...         0.00%       0.000us         0.00%       0.000us       0.000us       4.518ms         0.99%       4.518ms     225.900us            20  
CudaCodeGen::kernel38(CudaCodeGen::Tensor<CudaCodeGe...         0.00%       0.000us         0.00%       0.000us       0.000us       4.267ms         0.94%       4.267ms     426.700us            10  
CudaCodeGen::kernel37(CudaCodeGen::Tensor<bool, 4>, ...         0.00%       0.000us         0.00%       0.000us       0.000us       4.025ms         0.89%       4.025ms     402.500us            10  
CudaCodeGen::kernel31(CudaCodeGen::Tensor<float, 1>,...         0.00%       0.000us         0.00%       0.000us       0.000us       3.974ms         0.87%       3.974ms      56.771us            70  
void at::native::vectorized_elementwise_kernel<4, at...         0.00%       0.000us         0.00%       0.000us       0.000us       3.645ms         0.80%       3.645ms       2.223us          1640  
sm80_xmma_wgrad_image_first_layer_f16f16_f32_f32_nhw...         0.00%       0.000us         0.00%       0.000us       0.000us       3.502ms         0.77%       3.502ms     350.200us            10  
CudaCodeGen::kernel8(CudaCodeGen::Tensor<CudaCodeGen...         0.00%       0.000us         0.00%       0.000us       0.000us       3.438ms         0.76%       3.438ms      49.114us            70  
cutlass_sm80_tensorop_f16_s16x8x16dgrad_optimized_f1...         0.00%       0.000us         0.00%       0.000us       0.000us       3.382ms         0.74%       3.382ms     338.200us            10  
cutlass_sm80_tensorop_f16_s16x8x16fprop_optimized_f1...         0.00%       0.000us         0.00%       0.000us       0.000us       3.286ms         0.72%       3.286ms     109.533us            30  
CudaCodeGen::kernel3(CudaCodeGen::Tensor<CudaCodeGen...         0.00%       0.000us         0.00%       0.000us       0.000us       3.061ms         0.67%       3.061ms     306.100us            10  
CudaCodeGen::kernel28(CudaCodeGen::Tensor<float, 1>,...         0.00%       0.000us         0.00%       0.000us       0.000us       2.928ms         0.64%       2.928ms      26.618us           110  
void at::native::vectorized_elementwise_kernel<4, at...         0.00%       0.000us         0.00%       0.000us       0.000us       2.897ms         0.64%       2.897ms       1.799us          1610  
cutlass_sm80_tensorop_f16_s16x8x16fprop_optimized_f1...         0.00%       0.000us         0.00%       0.000us       0.000us       2.822ms         0.62%       2.822ms      94.067us            30  
CudaCodeGen::kernel11(CudaCodeGen::Tensor<CudaCodeGe...         0.00%       0.000us         0.00%       0.000us       0.000us       2.527ms         0.56%       2.527ms      22.973us           110  
CudaCodeGen::kernel21(CudaCodeGen::Tensor<CudaCodeGe...         0.00%       0.000us         0.00%       0.000us       0.000us       2.514ms         0.55%       2.514ms     251.400us            10  
CudaCodeGen::kernel33(CudaCodeGen::Tensor<float, 1>,...         0.00%       0.000us         0.00%       0.000us       0.000us       2.307ms         0.51%       2.307ms     230.700us            10  
CudaCodeGen::kernel24(CudaCodeGen::Tensor<float, 1>,...         0.00%       0.000us         0.00%       0.000us       0.000us       2.298ms         0.51%       2.298ms      57.450us            40  
void at::native::unrolled_elementwise_kernel<at::nat...         0.00%       0.000us         0.00%       0.000us       0.000us       2.152ms         0.47%       2.152ms       2.030us          1060  
CudaCodeGen::kernel6(CudaCodeGen::Tensor<CudaCodeGen...         0.00%       0.000us         0.00%       0.000us       0.000us       2.012ms         0.44%       2.012ms     201.200us            10  
void at::native::vectorized_elementwise_kernel<4, at...         0.00%       0.000us         0.00%       0.000us       0.000us       1.970ms         0.43%       1.970ms       1.000us          1970  
cutlass_sm80_tensorop_f16_s16x8x16dgrad_optimized_f1...         0.00%       0.000us         0.00%       0.000us       0.000us       1.920ms         0.42%       1.920ms      64.000us            30  
void cutlass::Kernel<cutlass_80_tensorop_f16_s16816g...         0.00%       0.000us         0.00%       0.000us       0.000us       1.711ms         0.38%       1.711ms      57.033us            30  
CudaCodeGen::kernel7(CudaCodeGen::Tensor<CudaCodeGen...         0.00%       0.000us         0.00%       0.000us       0.000us       1.620ms         0.36%       1.620ms     162.000us            10  
void splitKreduce_kernel<32, 16, int, __half, __half...         0.00%       0.000us         0.00%       0.000us       0.000us       1.490ms         0.33%       1.490ms       4.967us           300  
sm80_xmma_gemm_f16f16_f16f32_f32_nt_n_tilesize96x128...         0.00%       0.000us         0.00%       0.000us       0.000us       1.467ms         0.32%       1.467ms      73.350us            20  
_ZN19cutlass_cudnn_train6KernelINS_4conv6kernel23Imp...         0.00%       0.000us         0.00%       0.000us       0.000us       1.392ms         0.31%       1.392ms     139.200us            10  
CudaCodeGen::kernel13(CudaCodeGen::Tensor<CudaCodeGe...         0.00%       0.000us         0.00%       0.000us       0.000us       1.379ms         0.30%       1.379ms     137.900us            10  
CudaCodeGen::kernel15(CudaCodeGen::Tensor<CudaCodeGe...         0.00%       0.000us         0.00%       0.000us       0.000us       1.350ms         0.30%       1.350ms     135.000us            10  
CudaCodeGen::kernel23(CudaCodeGen::Tensor<CudaCodeGe...         0.00%       0.000us         0.00%       0.000us       0.000us       1.220ms         0.27%       1.220ms      61.000us            20  
CudaCodeGen::kernel29(CudaCodeGen::Tensor<float, 1>,...         0.00%       0.000us         0.00%       0.000us       0.000us       1.200ms         0.26%       1.200ms     120.000us            10  
CudaCodeGen::kernel10(CudaCodeGen::Tensor<CudaCodeGe...         0.00%       0.000us         0.00%       0.000us       0.000us       1.060ms         0.23%       1.060ms     106.000us            10  
void at::native::reduce_kernel<512, 1, at::native::R...         0.00%       0.000us         0.00%       0.000us       0.000us     930.000us         0.20%     930.000us      93.000us            10  
ampere_s16816gemm_fp16_64x64_sliced1x2_ldg8_stages_6...         0.00%       0.000us         0.00%       0.000us       0.000us     846.000us         0.19%     846.000us      84.600us            10  
                         Memcpy DtoD (Device -> Device)         0.00%       0.000us         0.00%       0.000us       0.000us     836.000us         0.18%     836.000us      41.800us            20  
void at::native::unrolled_elementwise_kernel<at::nat...         0.00%       0.000us         0.00%       0.000us       0.000us     799.000us         0.18%     799.000us      79.900us            10  
CudaCodeGen::kernel19(CudaCodeGen::Tensor<CudaCodeGe...         0.00%       0.000us         0.00%       0.000us       0.000us     733.000us         0.16%     733.000us      73.300us            10  
CudaCodeGen::kernel20(CudaCodeGen::Tensor<CudaCodeGe...         0.00%       0.000us         0.00%       0.000us       0.000us     710.000us         0.16%     710.000us      71.000us            10  
CudaCodeGen::kernel12(CudaCodeGen::Tensor<CudaCodeGe...         0.00%       0.000us         0.00%       0.000us       0.000us     671.000us         0.15%     671.000us      67.100us            10  
CudaCodeGen::kernel25(CudaCodeGen::Tensor<float, 1>,...         0.00%       0.000us         0.00%       0.000us       0.000us     620.000us         0.14%     620.000us      12.400us            50  
CudaCodeGen::kernel17(CudaCodeGen::Tensor<CudaCodeGe...         0.00%       0.000us         0.00%       0.000us       0.000us     598.000us         0.13%     598.000us      11.960us            50  
CudaCodeGen::kernel26(CudaCodeGen::Tensor<float, 1>,...         0.00%       0.000us         0.00%       0.000us       0.000us     568.000us         0.12%     568.000us      56.800us            10  
CudaCodeGen::kernel16(CudaCodeGen::Tensor<CudaCodeGe...         0.00%       0.000us         0.00%       0.000us       0.000us     480.000us         0.11%     480.000us      48.000us            10  
                                                memset8         0.00%       0.000us         0.00%       0.000us       0.000us     460.000us         0.10%     460.000us       3.286us           140  
                                              memcpy128         0.00%       0.000us         0.00%       0.000us       0.000us     391.000us         0.09%     391.000us      39.100us            10  
                                       Memset (Unknown)         0.00%       0.000us         0.00%       0.000us       0.000us     360.000us         0.08%     360.000us       2.000us           180  
CudaCodeGen::kernel18(CudaCodeGen::Tensor<CudaCodeGe...         0.00%       0.000us         0.00%       0.000us       0.000us     291.000us         0.06%     291.000us      29.100us            10  
void at::native::vectorized_elementwise_kernel<4, at...         0.00%       0.000us         0.00%       0.000us       0.000us     160.000us         0.04%     160.000us       1.000us           160  
cutlass_tensorop_f16_s16816gemm_f16_64x64_64x5_tn_al...         0.00%       0.000us         0.00%       0.000us       0.000us     100.000us         0.02%     100.000us      10.000us            10  
cutlass_tensorop_f16_s16816gemm_f16_128x256_32x3_nt_...         0.00%       0.000us         0.00%       0.000us       0.000us      90.000us         0.02%      90.000us       9.000us            10  
cutlass_tensorop_f16_s16816gemm_f16_64x64_64x5_tt_al...         0.00%       0.000us         0.00%       0.000us       0.000us      80.000us         0.02%      80.000us       8.000us            10  
void at::native::reduce_kernel<128, 4, at::native::R...         0.00%       0.000us         0.00%       0.000us       0.000us      60.000us         0.01%      60.000us       6.000us            10  
void splitKreduce_kernel<32, 16, int, float, __half,...         0.00%       0.000us         0.00%       0.000us       0.000us      60.000us         0.01%      60.000us       6.000us            10  
                                          memcpy32_post         0.00%       0.000us         0.00%       0.000us       0.000us      55.000us         0.01%      55.000us       1.375us            40  
                                               memset32         0.00%       0.000us         0.00%       0.000us       0.000us      50.000us         0.01%      50.000us       2.500us            20  
void cutlass_cudnn_train::Kernel<cutlass_cudnn_train...         0.00%       0.000us         0.00%       0.000us       0.000us      41.000us         0.01%      41.000us       4.100us            10  
                                         softmax_kernel         0.00%       0.000us         0.00%       0.000us       0.000us      30.000us         0.01%      30.000us       3.000us            10  
CudaCodeGen::kernel22(CudaCodeGen::Tensor<CudaCodeGe...         0.00%       0.000us         0.00%       0.000us       0.000us      20.000us         0.00%      20.000us       2.000us            10  
void at::native::vectorized_elementwise_kernel<4, at...         0.00%       0.000us         0.00%       0.000us       0.000us      10.000us         0.00%      10.000us       1.000us            10  
                                       cudaLaunchKernel        26.40%     118.056ms        26.40%     118.056ms      72.427us       0.000us         0.00%       0.000us       0.000us          1630  
                                        cudaMemcpyAsync         0.04%     176.000us         0.04%     176.000us       8.800us       0.000us         0.00%       0.000us       0.000us            20  
                                  cudaStreamIsCapturing         0.00%       3.000us         0.00%       3.000us       0.150us       0.000us         0.00%       0.000us       0.000us            20  
                                        cudaGraphLaunch        26.94%     120.461ms        26.94%     120.461ms      12.046ms       0.000us         0.00%       0.000us       0.000us            10  
                                   cudaDriverGetVersion         0.00%       0.000us         0.00%       0.000us       0.000us       0.000us         0.00%       0.000us       0.000us            10  
                                  cudaDeviceSynchronize        46.61%     208.421ms        46.61%     208.421ms     208.421ms       0.000us         0.00%       0.000us       0.000us             1  
-------------------------------------------------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  
Self CPU time total: 447.117ms
Self CUDA time total: 454.794ms