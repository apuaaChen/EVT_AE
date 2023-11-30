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
# Benchmarking on MLP and its backward pass with activation relu
import torch.nn as nn
import torch
from torch.profiler import ProfilerActivity
from torch.profiler import profile as torch_profile
from typing import Any
import argparse
import tvm

from gtl.compiler.passes import (
    GTLFrontend, pass_loss_elimination,
    pass_decomposition, pass_cse,
    pass_constant_propagation,
    pass_fusion,
    pass_clean_up,
    pass_print_graph)

from torch._dynamo.backends.common import aot_autograd
from gtl.helper import compiler_fn
from torch.utils._python_dispatch import _pop_mode_temporarily, _len_torch_dispatch_stack
from gtl.helper import autotvm_tuner, compile_tvm, ansor_tuner
from contextlib import nullcontext
from tvm import relay

################################################################################
# Define the module
################################################################################
class MLP(nn.Module):
    def forward(self, input, w1, w2):
        w1_t = torch.ops.aten.t(w1)
        w2_t = torch.ops.aten.t(w2)
        mm1 = torch.ops.aten.mm(input, w1_t)
        relu1 = torch.ops.aten.relu(mm1)
        mm2 = torch.ops.aten.mm(relu1, w2_t)
        relu2 = torch.ops.aten.relu(mm2)

        drelu2 = torch.ops.aten._softmax(relu2, 1, False)
        dmm2 = torch.ops.aten.threshold_backward(drelu2, relu2, 0)
        drelu1 = torch.ops.aten.mm(dmm2, w2)
        dmm2_t = torch.ops.aten.t(dmm2)
        dw2 = torch.ops.aten.mm(dmm2_t, relu1)
        dmm1 = torch.ops.aten.threshold_backward(drelu1, relu1, 0)
        dinput = torch.ops.aten.mm(dmm1, w1)
        dmm1_t = torch.ops.aten.t(dmm1)
        dw1 = torch.ops.aten.mm(dmm1_t, input)
        return dinput, dw1, dw2


batch_size = 4096
input_dim = 256
hidden_dim = 2048
output_dim = 256


def evt_compile_fn(fx_module: torch.fx.GraphModule, _):
    # pass_print_graph(fx_module, "./original_mlp.svg")
    frontend = GTLFrontend()
    fx_module, _ = frontend(fx_module)
    pass_decomposition(fx_module, fx_module.graph)
    pass_cse(fx_module, fx_module.graph)
    pass_constant_propagation(fx_module, fx_module.graph)

    pass_print_graph(fx_module, "./evt_mlp.svg")
    pass_fusion(fx_module, fx_module.graph)
    fx_module.recompile()
    
    return fx_module

def pass_create_example_inputs(gm: torch.fx.GraphModule):
    inputs = []
    for node in gm.graph.nodes:
        if node.op == "placeholder":
            print(node.name)
            meta = node.meta["tensor_meta"]
            inputs.append(
                torch.empty(meta.shape, dtype=meta.dtype, device="cuda")
            )
    return inputs

def tvm_compile_fn(model, _):
    frontend = GTLFrontend()
    model, _ = frontend(model)
    pass_decomposition(model, model.graph)
    model.graph.eliminate_dead_code()
    model.recompile()

    with (_pop_mode_temporarily() 
        if _len_torch_dispatch_stack() > 0 else nullcontext()):
        
        inputs = pass_create_example_inputs(model)
        model.eval()
        scripted_model = torch.jit.script(model)
        shape_list = [(f"inp_{idx}", i.shape) for idx, i in enumerate(inputs)]
        mod, params = relay.frontend.from_pytorch(scripted_model, shape_list, default_dtype="float16")
        
        # Aututuner
        autotvm_log_file = "./tvm_autotvm.log"
        ansor_log_file = "./tvm_ansor.log"
        autotvm_tuner(mod, params, autotvm_log_file, 1000, 200)
        ansor_tuner(mod, params, ansor_log_file, 1000)
        # exec_tvm = compile_tvm(mod, params, additional_outputs=[])
        # exec_tvm = compile_tvm(mod, params, autotvm_log_file=autotvm_log_file, additional_outputs=[])
        exec_tvm = compile_tvm(mod, params, ansor_log_file=ansor_log_file, additional_outputs=[])
    
    return exec_tvm
            


class MLPProfile:
    def __init__(self, method) -> None:
        self.method = method
    
    def profile(self, model, inputs):
        # warmup
        for _ in range(20):
            model(*inputs)
        
        # profile
        with torch_profile(activities=[ProfilerActivity.CUDA], record_shapes=True) as prof:
            for _ in range(20):
                model(*inputs)
        
        print(prof.key_averages().table(sort_by="cuda_time_total"))
    
    def __call__(self) -> Any:
        model = MLP().eval()
        input = torch.randn(
            (batch_size, input_dim), dtype=torch.float16, device="cuda"
        )

        w1 = torch.randn(
            (hidden_dim, input_dim), dtype=torch.float16, device="cuda"
        )

        w2 = torch.randn(
            (output_dim, hidden_dim), dtype=torch.float16, device="cuda"
        )

        inputs = [input, w1, w2]

        if self.method == "evt":
            evt_backend = aot_autograd(
                fw_compiler=evt_compile_fn, bw_compiler=compiler_fn)
            model = torch.compile(model, dynamic=False, backend=evt_backend)
        elif self.method == "triton":
            model = torch.compile(
                model, dynamic=False, backend="inductor",
                options={"epilogue_fusion": True, "max_autotune": True,
                "trace.enabled": True})
        elif self.method == "tvm":
            tvm_backend = aot_autograd(
                fw_compiler=tvm_compile_fn, bw_compiler=compiler_fn
            )
            model = torch.compile(model, dynamic=False, backend=tvm_backend)


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
    profiler = MLPProfile(args.method)
    profiler()




