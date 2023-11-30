################################################################################
# Copyright [yyyy] [name of copyright owner]
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
################################################################################
# Benchmarking the fusion of conv2d with the reduction of BN
import torch.nn as nn
import torch
from torch.profiler import ProfilerActivity
from torch.profiler import profile as torch_profile
from typing import Any
import argparse

from gtl.compiler.passes import (
    GTLFrontend, pass_loss_elimination,
    pass_decomposition, pass_cse,
    pass_constant_propagation,
    pass_permute_propagation,
    pass_fusion,
    pass_clean_up,
    pass_print_graph)

from torch._dynamo.backends.common import aot_autograd
from gtl.helper import compiler_fn
from torch.utils._python_dispatch import _pop_mode_temporarily, _len_torch_dispatch_stack
from gtl.helper import autotvm_tuner, compile_tvm, ansor_tuner,  apex_autocast
from contextlib import nullcontext
from tvm import relay

from gtl.compiler.passes.pass_decomposition import convolution_forward_channel_last, batch_norm_stat, batch_norm_elemt

torch.fx.wrap('batch_norm_stat')
torch.fx.wrap('batch_norm_elemt')
################################################################################
# Define the module
################################################################################
class ConvBNEVT(nn.Module):
    def forward(self, input_cl, weight_cl, gamma, beta, K, reduction_factor):
        input = torch.ops.aten.permute(input_cl, [0, 3, 1, 2])
        weight = torch.ops.aten.permute(weight_cl, [0, 3, 1, 2])
        conv_output = torch.ops.aten.convolution(input, weight, None, [1, 1], [1, 1], [1, 1], False, [0, 0], 1)
        output, mean, rstd = torch.ops.aten._native_batch_norm_legit.no_stats(
            conv_output, gamma, beta, True, 0.1, 1e-6
        )
        output = torch.ops.aten.permute(output, [0, 2, 3, 1])
        output_relu = torch.ops.aten.relu(output)

        return output_relu, mean, rstd

class ConvBNTVM(nn.Module):
    def forward(self, input, weight, gamma, beta):
        conv_output = torch.ops.aten.convolution(input, weight, None, [1, 1], [1, 1], [1, 1], False, [0, 0], 1)
        mean = torch.ops.aten.mean(conv_output, [0, 2, 3], dtype=torch.float32, keepdim=True)
        square = torch.ops.aten.square(conv_output)
        meanx2 = torch.ops.aten.mean(square, [0, 2, 3], dtype=torch.float32, keepdim=True)
        # BN elem
        var = meanx2 - mean * mean
        rstd = torch.ops.aten.rsqrt(var + 1e-6)
        output = (conv_output - mean) * gamma * rstd + beta
        return output, mean, rstd

    
class ConvBN(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size) -> None:
        super().__init__()
        self.conv = nn.Conv2d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            bias=False
        )
        self.bn = nn.BatchNorm2d(out_channels, track_running_stats=False)

    def forward(self, input):
        conv_output = self.conv(input)
        output = torch.nn.functional.relu(self.bn(conv_output))
        return output
    

################################################################################
# Compilers
################################################################################

def evt_compile_fn(fx_module: torch.fx.GraphModule, _):
    frontend = GTLFrontend()
    fx_module, _ = frontend(fx_module)
    pass_decomposition(fx_module, fx_module.graph)
    pass_permute_propagation(fx_module, fx_module.graph)
    pass_cse(fx_module, fx_module.graph)
    pass_constant_propagation(fx_module, fx_module.graph)
    pass_print_graph(fx_module, "./self_atten.svg")
    pass_fusion(fx_module, fx_module.graph)
    # pass_clean_up(fx_module, fx_module.graph)
    fx_module.recompile()
    
    return fx_module


################################################################################
# Profiler
################################################################################
class ConvBNProfile:
    def __init__(self, method) -> None:
        self.method = method
    
    def profile(self, model, inputs):
        # warmup
        for _ in range(20):
            loss = model(*inputs)
        
        # profile
        with torch_profile(activities=[ProfilerActivity.CUDA], record_shapes=True) as prof:
            for _ in range(20):
                loss = model(*inputs)
        
        print(prof.key_averages().table(sort_by="cuda_time_total"))
    
    def __call__(self) -> Any:
        N = 32
        H = 28
        W = 28
        C = 256
        K = 256
        R = 3
        S = 3
    
        if self.method == "evt":
            model = ConvBNEVT()
            input_cl = torch.randn((N, H, W, C), dtype=torch.float16, device="cuda")
            weight_cl = torch.randn((K, R, S, C), dtype=torch.float16, device="cuda")
            gamma = torch.randn((K,), dtype=torch.float16, device="cuda")
            beta = torch.randn((K,), dtype=torch.float16, device="cuda")
            reduction_factor = 1. / (N * H * W)
            
            inputs = [input_cl, weight_cl, gamma, beta, K, reduction_factor]
        elif self.method == "tvm":
            model = ConvBNTVM()
            input = torch.randn((N, C, H, W), dtype=torch.float16, device="cuda")
            weight = torch.randn((K, C, R, S), dtype=torch.float16, device="cuda")
            gamma = torch.randn((1, K, 1, 1), dtype=torch.float16, device="cuda")
            beta = torch.randn((1, K, 1, 1), dtype=torch.float16, device="cuda")
            
            inputs = [input, weight, gamma, beta]
        else:
            model = ConvBN(C, K, R).to("cuda")
            param_optimizer = list()
            no_decay = ['bias', 'gamma', 'beta']

            optimizer_grouped_parameters = [
                {'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)], 'weight_decay': 0.0},
                {'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}]

            optimizer = torch.optim.SGD(optimizer_grouped_parameters, 6e-3)

            model, optimizer = apex_autocast(
                model, optimizer, True
            )
            model.to(memory_format=torch.channels_last)
            model.train()
            input = torch.randn((N, C, H, W), dtype=torch.float16, device="cuda").to(memory_format=torch.channels_last)

            inputs = [input]
        
        if self.method == "evt":
            evt_backend = aot_autograd(
                fw_compiler=evt_compile_fn, bw_compiler=compiler_fn)
            model = torch.compile(model, dynamic=False, backend=evt_backend)
        elif self.method == "triton":
            model = torch.compile(model, dynamic=False, backend="inductor", 
                                  options={"epilogue_fusion": True, "max_autotune": True})
        elif self.method == "tvm":
            scripted_model = torch.jit.trace(model, inputs).eval()
            shape_list = [(f"inp_{idx}", i.shape) for idx, i in enumerate(inputs)]
            mod, params = relay.frontend.from_pytorch(scripted_model, shape_list, default_dtype="float16")

            autotvm_log_file = "./tvm_autotvm.log"
            autotvm_tuner(mod, params, autotvm_log_file, 1000, 200)
            ansor_log_file = "./tvm_ansor.log"
            ansor_tuner(mod, params, ansor_log_file, 1000, skipped_kws = ["matmul", "dense", "conv"])

            # model = compile_tvm(mod, params, additional_outputs=[])
            # model = compile_tvm(mod, params, autotvm_log_file=autotvm_log_file, additional_outputs=[])
            model = compile_tvm(mod, params, ansor_log_file=ansor_log_file, additional_outputs=[])

        self.profile(model, inputs)


if __name__ == '__main__':
    ################################################################################
    # parse args
    parser = argparse.ArgumentParser(description="Operator Compiler Benchmarking")
    # method
    parser.add_argument(
        '--method', '-mt', type=str, default="torch", 
        choices=["torch", "evt", "tvm", "inductor", "triton", "bolt"])
    args = parser.parse_args()

    ################################################################################
    # logging.basicConfig(level=logging.DEBUG)
    profiler = ConvBNProfile(args.method)
    profiler()
