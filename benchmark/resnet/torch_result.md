-------------------------------------------------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  
                                                   Name    Self CPU %      Self CPU   CPU total %     CPU total  CPU time avg     Self CUDA   Self CUDA %    CUDA total  CUDA time avg    # of Calls  
-------------------------------------------------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  
void cudnn::bn_bw_1C11_kernel_new<__half, float, flo...         0.00%       0.000us         0.00%       0.000us       0.000us      99.443ms        11.44%      99.443ms     904.027us           110  
void cudnn::ops::nchwToNhwcKernel<__half, __half, fl...         0.00%       0.000us         0.00%       0.000us       0.000us      77.376ms         8.90%      77.376ms      48.972us          1580  
void cudnn::bn_fw_tr_1C11_kernel_NCHW<__half, float,...         0.00%       0.000us         0.00%       0.000us       0.000us      68.682ms         7.90%      68.682ms     624.382us           110  
void at::native::vectorized_elementwise_kernel<4, at...         0.00%       0.000us         0.00%       0.000us       0.000us      64.489ms         7.42%      64.489ms      74.125us           870  
void cudnn::bn_bw_1C11_kernel_new<__half, float, flo...         0.00%       0.000us         0.00%       0.000us       0.000us      54.276ms         6.25%      54.276ms     246.709us           220  
void at::native::vectorized_elementwise_kernel<4, at...         0.00%       0.000us         0.00%       0.000us       0.000us      53.544ms         6.16%      53.544ms     109.273us           490  
void at::native::vectorized_elementwise_kernel<4, at...         0.00%       0.000us         0.00%       0.000us       0.000us      34.765ms         4.00%      34.765ms      70.949us           490  
void cudnn::bn_fw_tr_1C11_kernel_NCHW<__half, float,...         0.00%       0.000us         0.00%       0.000us       0.000us      31.512ms         3.63%      31.512ms     242.400us           130  
void cudnn::bn_bw_1C11_kernel_new<__half, float, flo...         0.00%       0.000us         0.00%       0.000us       0.000us      29.380ms         3.38%      29.380ms       2.938ms            10  
void cudnn::ops::nhwcToNchwKernel<__half, __half, fl...         0.00%       0.000us         0.00%       0.000us       0.000us      29.194ms         3.36%      29.194ms      42.932us           680  
void at::native::(anonymous namespace)::max_pool_bac...         0.00%       0.000us         0.00%       0.000us       0.000us      22.798ms         2.62%      22.798ms       2.280ms            10  
void cudnn::bn_fw_tr_1C11_kernel_NCHW<__half, float,...         0.00%       0.000us         0.00%       0.000us       0.000us      21.237ms         2.44%      21.237ms       2.124ms            10  
sm80_xmma_wgrad_implicit_gemm_indexed_f16f16_f16f32_...         0.00%       0.000us         0.00%       0.000us       0.000us      18.393ms         2.12%      18.393ms     131.379us           140  
void cutlass::Kernel<cutlass_80_tensorop_f16_s16816g...         0.00%       0.000us         0.00%       0.000us       0.000us      16.676ms         1.92%      16.676ms     151.600us           110  
void cudnn::bn_bw_1C11_singleread_spec<__half2, 512,...         0.00%       0.000us         0.00%       0.000us       0.000us      15.916ms         1.83%      15.916ms      83.768us           190  
sm80_xmma_fprop_implicit_gemm_f16f16_f16f32_f32_nhwc...         0.00%       0.000us         0.00%       0.000us       0.000us      13.611ms         1.57%      13.611ms     136.110us           100  
void cutlass_cudnn_infer::Kernel<cutlass_tensorop_f1...         0.00%       0.000us         0.00%       0.000us       0.000us      13.266ms         1.53%      13.266ms     102.046us           130  
     ampere_s16816gemm_fp16_128x128_ldg8_stages_32x5_tn         0.00%       0.000us         0.00%       0.000us       0.000us      11.878ms         1.37%      11.878ms     148.475us            80  
void cutlass_cudnn_infer::Kernel<cutlass_tensorop_f1...         0.00%       0.000us         0.00%       0.000us       0.000us      11.386ms         1.31%      11.386ms      94.883us           120  
void cutlass::Kernel<cutlass_80_tensorop_f16_s16816g...         0.00%       0.000us         0.00%       0.000us       0.000us      10.288ms         1.18%      10.288ms     171.467us            60  
void cudnn::bn_fw_tr_1C11_singleread_spec<__half2, 5...         0.00%       0.000us         0.00%       0.000us       0.000us      10.098ms         1.16%      10.098ms      53.147us           190  
void cutlass_cudnn_infer::Kernel<cutlass_tensorop_f1...         0.00%       0.000us         0.00%       0.000us       0.000us       9.973ms         1.15%       9.973ms     199.460us            50  
          ampere_fp16_s1688gemm_fp16_256x64_ldg8_f2f_nn         0.00%       0.000us         0.00%       0.000us       0.000us       9.771ms         1.12%       9.771ms     244.275us            40  
      ampere_s16816gemm_fp16_64x128_ldg8_stages_64x4_tn         0.00%       0.000us         0.00%       0.000us       0.000us       8.337ms         0.96%       8.337ms     208.425us            40  
void cutlass::Kernel<cutlass_80_tensorop_f16_s16816g...         0.00%       0.000us         0.00%       0.000us       0.000us       7.998ms         0.92%       7.998ms     159.960us            50  
ampere_fp16_s16816gemm_fp16_128x64_ldg8_f2f_stages_3...         0.00%       0.000us         0.00%       0.000us       0.000us       7.700ms         0.89%       7.700ms     192.500us            40  
sm80_xmma_dgrad_implicit_gemm_f16f16_f16f32_f32_nhwc...         0.00%       0.000us         0.00%       0.000us       0.000us       7.536ms         0.87%       7.536ms     150.720us            50  
ampere_fp16_s16816gemm_fp16_128x128_ldg8_f2f_stages_...         0.00%       0.000us         0.00%       0.000us       0.000us       7.355ms         0.85%       7.355ms     147.100us            50  
void cutlass::Kernel<cutlass_80_tensorop_f16_s16816g...         0.00%       0.000us         0.00%       0.000us       0.000us       7.255ms         0.83%       7.255ms     145.100us            50  
void cutlass_cudnn_train::Kernel<cutlass_tensorop_f1...         0.00%       0.000us         0.00%       0.000us       0.000us       7.214ms         0.83%       7.214ms     120.233us            60  
void at::native::(anonymous namespace)::max_pool_for...         0.00%       0.000us         0.00%       0.000us       0.000us       6.520ms         0.75%       6.520ms     652.000us            10  
sm80_xmma_wgrad_implicit_gemm_indexed_f16f16_f16f32_...         0.00%       0.000us         0.00%       0.000us       0.000us       6.304ms         0.73%       6.304ms     210.133us            30  
sm80_xmma_wgrad_implicit_gemm_indexed_f16f16_f16f32_...         0.00%       0.000us         0.00%       0.000us       0.000us       5.883ms         0.68%       5.883ms      84.043us            70  
_ZN19cutlass_cudnn_train6KernelINS_4conv6kernel23Imp...         0.00%       0.000us         0.00%       0.000us       0.000us       5.430ms         0.62%       5.430ms      90.500us            60  
void cutlass::Kernel<cutlass_80_tensorop_f16_s16816g...         0.00%       0.000us         0.00%       0.000us       0.000us       5.383ms         0.62%       5.383ms     134.575us            40  
sm80_xmma_fprop_implicit_gemm_f16f16_f16f32_f32_nhwc...         0.00%       0.000us         0.00%       0.000us       0.000us       5.086ms         0.59%       5.086ms     169.533us            30  
          ampere_fp16_s1688gemm_fp16_256x64_ldg8_f2f_nt         0.00%       0.000us         0.00%       0.000us       0.000us       4.802ms         0.55%       4.802ms     240.100us            20  
      ampere_s16816gemm_fp16_128x64_ldg8_stages_64x4_tn         0.00%       0.000us         0.00%       0.000us       0.000us       4.299ms         0.49%       4.299ms     214.950us            20  
ampere_fp16_s16816gemm_fp16_128x128_ldg8_f2f_stages_...         0.00%       0.000us         0.00%       0.000us       0.000us       4.156ms         0.48%       4.156ms     103.900us            40  
ampere_fp16_s16816gemm_fp16_128x64_ldg8_f2f_stages_3...         0.00%       0.000us         0.00%       0.000us       0.000us       4.135ms         0.48%       4.135ms     206.750us            20  
void cudnn::bn_fw_tr_1C11_singleread<__half, 512, tr...         0.00%       0.000us         0.00%       0.000us       0.000us       3.583ms         0.41%       3.583ms      39.811us            90  
void cudnn::cnn::reduce_wgrad_nchw_helper<float, __h...         0.00%       0.000us         0.00%       0.000us       0.000us       3.569ms         0.41%       3.569ms      22.306us           160  
sm80_xmma_wgrad_image_first_layer_f16f16_f32_f32_nhw...         0.00%       0.000us         0.00%       0.000us       0.000us       3.509ms         0.40%       3.509ms     350.900us            10  
sm80_xmma_fprop_image_first_layer_f16f16_f32_f16_nhw...         0.00%       0.000us         0.00%       0.000us       0.000us       3.270ms         0.38%       3.270ms     327.000us            10  
void at::native::vectorized_elementwise_kernel<4, at...         0.00%       0.000us         0.00%       0.000us       0.000us       3.020ms         0.35%       3.020ms       2.849us          1060  
     ampere_s16816gemm_fp16_256x128_ldg8_stages_64x3_tn         0.00%       0.000us         0.00%       0.000us       0.000us       2.836ms         0.33%       2.836ms     283.600us            10  
void cutlass_cudnn_infer::Kernel<cutlass_tensorop_f1...         0.00%       0.000us         0.00%       0.000us       0.000us       2.674ms         0.31%       2.674ms     267.400us            10  
void at::native::vectorized_elementwise_kernel<4, at...         0.00%       0.000us         0.00%       0.000us       0.000us       2.599ms         0.30%       2.599ms       4.481us           580  
sm80_xmma_dgrad_implicit_gemm_f16f16_f16f32_f32_nhwc...         0.00%       0.000us         0.00%       0.000us       0.000us       1.391ms         0.16%       1.391ms     139.100us            10  
void cutlass::Kernel<cutlass_80_wmma_tensorop_f16_s1...         0.00%       0.000us         0.00%       0.000us       0.000us       1.101ms         0.13%       1.101ms     110.100us            10  
void cutlass::Kernel<cutlass_80_wmma_tensorop_f16_s1...         0.00%       0.000us         0.00%       0.000us       0.000us       1.073ms         0.12%       1.073ms     107.300us            10  
void at::native::vectorized_elementwise_kernel<4, at...         0.00%       0.000us         0.00%       0.000us       0.000us       1.060ms         0.12%       1.060ms       1.000us          1060  
void at::native::reduce_kernel<512, 1, at::native::R...         0.00%       0.000us         0.00%       0.000us       0.000us     930.000us         0.11%     930.000us      93.000us            10  
                                                memset8         0.00%       0.000us         0.00%       0.000us       0.000us     929.000us         0.11%     929.000us       3.871us           240  
ampere_s16816gemm_fp16_64x64_sliced1x2_ldg8_stages_6...         0.00%       0.000us         0.00%       0.000us       0.000us     890.000us         0.10%     890.000us      89.000us            10  
                         Memcpy DtoD (Device -> Device)         0.00%       0.000us         0.00%       0.000us       0.000us     832.000us         0.10%     832.000us      41.600us            20  
                                       Memset (Unknown)         0.00%       0.000us         0.00%       0.000us       0.000us     600.000us         0.07%     600.000us       2.000us           300  
void at::native::elementwise_kernel<128, 4, at::nati...         0.00%       0.000us         0.00%       0.000us       0.000us     420.000us         0.05%     420.000us      42.000us            10  
sm80_xmma_wgrad_implicit_gemm_indexed_f16f16_f16f32_...         0.00%       0.000us         0.00%       0.000us       0.000us     275.000us         0.03%     275.000us       3.929us            70  
                                               memset32         0.00%       0.000us         0.00%       0.000us       0.000us     270.000us         0.03%     270.000us       3.857us            70  
void cutlass_cudnn_train::Kernel<cutlass_cudnn_train...         0.00%       0.000us         0.00%       0.000us       0.000us     244.000us         0.03%     244.000us       4.067us            60  
void cutlass::Kernel<cutlass_80_tensorop_f16_s16816g...         0.00%       0.000us         0.00%       0.000us       0.000us      90.000us         0.01%      90.000us       9.000us            10  
ampere_fp16_s16816gemm_fp16_256x128_ldg8_f2f_stages_...         0.00%       0.000us         0.00%       0.000us       0.000us      90.000us         0.01%      90.000us       9.000us            10  
ampere_fp16_s16816gemm_fp16_64x64_ldg8_f2f_stages_64...         0.00%       0.000us         0.00%       0.000us       0.000us      80.000us         0.01%      80.000us       8.000us            10  
void at::native::(anonymous namespace)::nll_loss_for...         0.00%       0.000us         0.00%       0.000us       0.000us      50.000us         0.01%      50.000us       5.000us            10  
void (anonymous namespace)::softmax_warp_backward<c1...         0.00%       0.000us         0.00%       0.000us       0.000us      50.000us         0.01%      50.000us       5.000us            10  
void at::native::reduce_kernel<128, 4, at::native::R...         0.00%       0.000us         0.00%       0.000us       0.000us      50.000us         0.01%      50.000us       5.000us            10  
void (anonymous namespace)::softmax_warp_forward<c10...         0.00%       0.000us         0.00%       0.000us       0.000us      40.000us         0.00%      40.000us       4.000us            10  
void splitKreduce_kernel<32, 16, int, __half, __half...         0.00%       0.000us         0.00%       0.000us       0.000us      30.000us         0.00%      30.000us       3.000us            10  
void at::native::vectorized_elementwise_kernel<4, at...         0.00%       0.000us         0.00%       0.000us       0.000us      20.000us         0.00%      20.000us       1.000us            20  
void at::native::(anonymous namespace)::nll_loss_bac...         0.00%       0.000us         0.00%       0.000us       0.000us      20.000us         0.00%      20.000us       2.000us            10  
                                       cudaLaunchKernel        37.92%     324.027ms        37.92%     324.027ms     198.790us       0.000us         0.00%       0.000us       0.000us          1630  
                                        cudaMemcpyAsync         0.02%     201.000us         0.02%     201.000us      10.050us       0.000us         0.00%       0.000us       0.000us            20  
                                  cudaStreamIsCapturing         0.00%       2.000us         0.00%       2.000us       0.100us       0.000us         0.00%       0.000us       0.000us            20  
                                        cudaGraphLaunch         1.13%       9.645ms         1.13%       9.645ms     964.500us       0.000us         0.00%       0.000us       0.000us            10  
                                   cudaDriverGetVersion         0.00%       0.000us         0.00%       0.000us       0.000us       0.000us         0.00%       0.000us       0.000us            10  
                                  cudaDeviceSynchronize        60.92%     520.544ms        60.92%     520.544ms     520.544ms       0.000us         0.00%       0.000us       0.000us             1  
-------------------------------------------------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  
Self CPU time total: 854.419ms
Self CUDA time total: 868.940ms