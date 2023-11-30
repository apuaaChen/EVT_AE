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
# Benchmarking on GCN Uturn
from typing import Any
import torch.nn as nn
import torch
from torch.profiler import ProfilerActivity
from torch.profiler import profile as torch_profile
from apex.contrib.xentropy import SoftmaxCrossEntropyLoss

# EVT
from gtl.compiler.passes import (
    GTLFrontend, pass_loss_elimination,
    pass_decomposition, pass_cse,
    pass_constant_propagation,
    pass_fusion,
    pass_clean_up,
    pass_print_graph)
from functools import partial
from gtl.helper import compiler_fn, partition_func
from torch._dynamo.backends.common import aot_autograd

from torch.utils._python_dispatch import _pop_mode_temporarily, _len_torch_dispatch_stack
from contextlib import nullcontext
from torch._inductor.fx_passes.joint_graph import joint_graph_passes

import argparse
from torch._inductor.compile_fx import compile_fx
from torch import _TorchCompileInductorWrapper

# TVM
from tvm import relay
from gtl.helper import autotvm_tuner, compile_tvm, ansor_tuner


################################################################################
# Define the model
################################################################################
class GCNUturn(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.loss = nn.CrossEntropyLoss(reduction='sum')
    
    def forward(self, h, labels):
        return self.loss(h, labels)
    
class GCNUturnApex(nn.Module):
    def __init__(self) -> None:
        super().__init__()
    
    def forward(self, h, labels):
        return SoftmaxCrossEntropyLoss.apply(h, labels, 0.0, 0, True).sum()

# The models below are created to simulate when our graph-level optimizations
# are jointly applied with the other backends
class GCNUturnLoss(nn.Module):
    def forward(self, h, labels, tangents):
        softmax = torch.ops.aten._softmax(h, 1, False)
        ne = torch.ops.aten.ne(labels, -1)
        one_hot = torch.ops.aten.one_hot(labels, num_classes=352)
        neg = torch.ops.aten.neg(one_hot)
        mul = torch.ops.aten.mul(tangents, neg)
        unsqueeze = torch.ops.aten.unsqueeze(ne, 1)
        mul_1 = torch.ops.aten.mul(unsqueeze, mul)
        sum_1 = torch.ops.aten.sum(mul_1, 1)
        unsqueeze_1 = torch.ops.aten.unsqueeze(sum_1, 1)
        mul_2 = torch.ops.aten.mul(softmax, unsqueeze_1)
        sub = torch.ops.aten.sub(mul_2, mul_1)
        return sub

class GCNUturnLossCP(nn.Module):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)

        self.const_unsqueeze_1 = torch.randn((1, 1), dtype=torch.float16, device="cuda")

    def forward(self, h, labels, tangents):
        softmax = torch.ops.aten._softmax(h, 1, False)
        mul_2 = torch.ops.aten.mul(softmax, self.const_unsqueeze_1)
        one_hot = torch.ops.aten.one_hot(labels, num_classes=352)
        add = torch.ops.aten.add(mul_2, one_hot)
        ne = torch.ops.aten.ne(labels, -1)
        unsqueeze = torch.ops.aten.unsqueeze(ne, 1)
        mul_9 = torch.ops.aten.mul(add, unsqueeze)
        mul_8 = torch.ops.aten.mul(mul_9, tangents)
        mul_7 = torch.ops.aten.mul(mul_8, -1.0)
        return mul_7

################################################################################
# EVT Passes
################################################################################
def joint_optimization(joint_module):
    frontend = GTLFrontend()
    joint_module, _ = frontend(joint_module)

    pass_loss_elimination(joint_module, joint_module.graph)
    pass_decomposition(joint_module, joint_module.graph)
    pass_cse(joint_module, joint_module.graph)
    pass_constant_propagation(joint_module, joint_module.graph)
    pass_fusion(joint_module, joint_module.graph)
    joint_module.recompile()

    return joint_module

################################################################################
# Triton Passes
################################################################################

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

import torch._functorch.config as functorch_config

def triton_loss_joint_optimization(joint_module):
    frontend = GTLFrontend()
    joint_module, _ = frontend(joint_module)
    pass_loss_elimination(joint_module, joint_module.graph)
    pass_decomposition(joint_module, joint_module.graph)
    # pass_print_graph(joint_module, "./triton_loss.svg")
    # joint_module.graph.eliminate_dead_code()
    joint_module.recompile()
    
    return joint_module


################################################################################
# Profiler
################################################################################
batch_size = 736392
num_classes = 352

class UturnProfile:
    def __init__(self, method) -> None:
        self.method = method
    
    def profile(self, model, inputs, mode="train"):
        if mode == "train":
            # warmup
            for _ in range(20):
                loss = model(*inputs)
                loss.backward()
            
            # profile
            with torch_profile(activities=[ProfilerActivity.CUDA], record_shapes=True) as prof:
                for _ in range(20):
                    loss = model(*inputs)
                    loss.backward()
        else:
            # warmup
            for _ in range(20):
                model(*inputs)
            
            # profile
            with torch_profile(activities=[ProfilerActivity.CUDA], record_shapes=True) as prof:
                for _ in range(20):
                    model(*inputs)
        
        print(prof.key_averages().table(sort_by="cuda_time_total"))
    
    def __call__(self) -> Any:
        h = torch.randn((batch_size, num_classes), dtype=torch.float16, device="cuda").requires_grad_(True)
        y = torch.randint(
            low=0, high=351, size=(batch_size,), dtype=torch.int64, device="cuda"
        )

        if self.method in ["triton+loss"]:
            tangents = torch.randn((1,), dtype=torch.float16, device="cuda")
            model = GCNUturnLoss().to(torch.float16).eval()
            inputs = [h, y, tangents]
            mode = "eval"
        elif self.method in ["triton+loss+cp", "tvm", "ansor"]:
            tangents = torch.randn((1,), dtype=torch.float16, device="cuda")
            model = GCNUturnLossCP().to(torch.float16).eval()
            inputs = [h, y, tangents]
            mode = "eval"
        elif self.method in ["apex"]:
            model = GCNUturnApex().to(torch.float16).train()
            inputs = [h, y]
            mode = "train"
        else:
            model = GCNUturn().to(torch.float16).train()
            inputs = [h, y]
            mode = "train"

        if self.method == "evt":
            partition_fn = partial(
                partition_func,
                joint_compiler=joint_optimization
            )
            evt_backend = aot_autograd(
                fw_compiler=compiler_fn, bw_compiler=compiler_fn, partition_fn=partition_fn)
            model = torch.compile(model, fullgraph=True, dynamic=False, backend=evt_backend)
        elif self.method in ["triton", "triton+loss", "triton+loss+cp" ]:
            model = torch.compile(
                model, dynamic=False, backend="inductor",
                # options={
                #     "epilogue_fusion": True, "max_autotune": True})#, 
                #     #"trace.enabled": True, "trace.graph_diagram": True})
                options={
                    "epilogue_fusion": True, "max_autotune": True, 
                    "trace.enabled": True, "trace.graph_diagram": True})
        elif self.method == "tvm":
            scripted_model = torch.jit.trace(model, inputs).eval()
            shape_list = [(f"inp_{idx}", i.shape) for idx, i in enumerate(inputs)]
            mod, params = relay.frontend.from_pytorch(scripted_model, shape_list, default_dtype="float16")
            model = compile_tvm(mod, params, additional_outputs=[])
        elif self.method == "ansor":
            scripted_model = torch.jit.trace(model, inputs).eval()
            shape_list = [(f"inp_{idx}", i.shape) for idx, i in enumerate(inputs)]
            mod, params = relay.frontend.from_pytorch(scripted_model, shape_list, default_dtype="float16")
            ansor_log_file = "./tvm_ansor.log"
            ansor_tuner(mod, params, ansor_log_file, 200)
            model = compile_tvm(mod, params, ansor_log_file=ansor_log_file, additional_outputs=[])



        self.profile(model, inputs, mode)
    

if __name__ == '__main__':
    ################################################################################
    # parse args
    parser = argparse.ArgumentParser(description="Operator Compiler Benchmarking")
    # method
    parser.add_argument(
        '--method', '-mt', type=str, default="torch", 
        choices=["torch", "evt", "tvm", "ansor", "inductor", "triton", "bolt", "triton+loss", "triton+loss+cp", "apex"])
    args = parser.parse_args()

    ################################################################################
    # logging.basicConfig(level=logging.DEBUG)
    profiler = UturnProfile(args.method)
    profiler()
