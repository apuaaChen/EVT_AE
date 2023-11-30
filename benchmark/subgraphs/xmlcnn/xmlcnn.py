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
# Benchmarking on Bert SelfAttention
from typing import Any
import torch
from torch.profiler import ProfilerActivity
from torch.profiler import profile as torch_profile
from model_zoo.bert.bert_modeling import BertSelfAttention, BertConfig
import argparse
from gtl.helper import compiler_fn, partition_func
from gtl.compiler.passes import (
    GTLFrontend, pass_loss_elimination,
    pass_decomposition, pass_cse,
    pass_constant_propagation,
    pass_fusion,
    pass_clean_up,
    pass_print_graph)
from functools import partial
from gtl.helper import autotvm_tuner, compile_tvm, ansor_tuner

from torch._dynamo.backends.common import aot_autograd

from tvm import relay
from tvm.relay.op.contrib.cutlass import partition_for_cutlass
import tvm
from tvm.contrib.cutlass import (
    finalize_modules
)
import torch.nn as nn

class xmlcnnUturn(nn.Module):
    def __init__(self, hidden_dims, y_dim) -> None:
        super().__init__()
        self.out_layer = nn.Linear(hidden_dims, y_dim)
        self.loss_fn = torch.nn.functional.binary_cross_entropy_with_logits

    def forward(self, input, batch_y):
        o = self.out_layer(input)
        loss = self.loss_fn(o, batch_y, reduction="sum") / batch_y.size(0)
        return loss

################################################################################
# EVT Passes
################################################################################
# def pass_clean_backward_permute(fx_module: torch.fx.GraphModule, _):
#     # Step 1: get output node
#     output = [node for node in fx_module.graph.nodes if node.target == "output"][0]
#     output_nodes = output.args[0]
#     permute_nodes = [node for node in output_nodes if node.target == torch.ops.aten.permute]
#     for permute in permute_nodes:
#         output.replace_input_with(permute, permute.args[0])


def joint_optimization(joint_module):
    frontend = GTLFrontend()
    joint_module, _ = frontend(joint_module)

    pass_loss_elimination(joint_module, joint_module.graph)
    pass_decomposition(joint_module, joint_module.graph, disabled_ops = [
        "requires_convolution", "requires_convolution_backward",
        "requires_max_pool2d_with_indices", "requires_max_pool2d_with_indices_backward"
    ])
    pass_cse(joint_module, joint_module.graph)
    pass_constant_propagation(joint_module, joint_module.graph)
    pass_fusion(joint_module, joint_module.graph)
    pass_clean_up(joint_module, joint_module.graph)
    joint_module.recompile()
    
    return joint_module

def tvm_joint_optimization(joint_module):
    frontend = GTLFrontend()
    joint_module, _ = frontend(joint_module)

    pass_loss_elimination(joint_module, joint_module.graph)
    pass_print_graph(joint_module, "./tvm_graph.svg")
    joint_module.recompile()


batch_size = 2048
hidden_dims=512
y_dim= 4096 # Too large will lead to error in autotvm


def bolt_compile_fn(model, _):
    for node in model.graph.nodes:
        if node.op == "output":
            print(node.args)
            node.args = ([node.args[0][0], node.args[0][1]])

    model.graph.eliminate_dead_code()
    model.recompile()
    primals_1 = torch.randn((y_dim, hidden_dims), dtype=torch.float16, device="cuda")
    primals_2 = torch.randn((y_dim,), dtype=torch.float16, device="cuda")
    primals_3 = torch.randn((batch_size, hidden_dims), dtype=torch.float16, device="cuda")
    primals_4 = torch.randn((batch_size, y_dim), dtype=torch.float16, device="cuda")
    tangents_1 = torch.randn(size=(1,), dtype=torch.float16, device="cuda")
    
    inputs = [primals_1, primals_2, primals_3, primals_4, tangents_1]
    
    scripted_model = torch.jit.script(model)
    shape_list = [(f"inp_{idx}", i.shape) for idx, i in enumerate(inputs)]
    mod, params = relay.frontend.from_pytorch(scripted_model, shape_list, default_dtype="float16")
    mod = partition_for_cutlass(mod, params)
    host = tvm.target.Target("llvm")
    cuda = tvm.target.Target("cuda -arch=sm_80", host=host)
    cutlass = tvm.target.Target(
        {
            "kind": "cutlass",
            "sm": 80,
            "use_3xtf32": False,
            "split_k_slices": [1],
            "profile_all_alignments": False,
            "find_first_valid": False,
            "use_multiprocessing": True,
            "use_fast_math": True,
            "tmp_dir": "./tmp",
        },
        host=host,
    )
    # Not with ansor
    with tvm.transform.PassContext(opt_level=3):
        lib = relay.build(mod, target=[cuda, cutlass], params=params)
    
    lib = finalize_modules(lib, "compile.so", "./tmp")
    dev = tvm.device("cuda", 0)
    rt_mod = tvm.contrib.graph_executor.GraphModule(lib["default"](dev))
    def exec_tvm(*args):
        for idx, arg in enumerate(args, 0):
            rt_mod.set_input(
                f"inp_{idx}",
                tvm.nd.from_dlpack(torch.utils.dlpack.to_dlpack(arg.contiguous())),
            )
        rt_mod.run()
        outs = [
            torch.utils.dlpack.from_dlpack(rt_mod.get_output(i).to_dlpack())
            for i in range(rt_mod.get_num_outputs())
        ]

        return outs + [None, None, None]
    return exec_tvm


def tvm_compile_fn(model, _):
    for node in model.graph.nodes:
        if node.op == "output":
            print(node.args)
            node.args = ([node.args[0][0], node.args[0][1]])

    model.graph.eliminate_dead_code()
    model.recompile()
    primals_1 = torch.randn((y_dim, hidden_dims), dtype=torch.float16, device="cuda")
    primals_2 = torch.randn((y_dim,), dtype=torch.float16, device="cuda")
    primals_3 = torch.randn((batch_size, hidden_dims), dtype=torch.float16, device="cuda")
    primals_4 = torch.randn((batch_size, y_dim), dtype=torch.float16, device="cuda")
    tangents_1 = torch.randn(size=(1,), dtype=torch.float16, device="cuda")
    
    inputs = [primals_1, primals_2, primals_3, primals_4, tangents_1]
    
    scripted_model = torch.jit.script(model)
    shape_list = [(f"inp_{idx}", i.shape) for idx, i in enumerate(inputs)]
    mod, params = relay.frontend.from_pytorch(scripted_model, shape_list, default_dtype="float16")

    # Aututuner
    autotvm_log_file = "./tvm_autotvm.log"
    ansor_log_file = "./tvm_ansor.log"
    autotvm_tuner(mod, params, autotvm_log_file, 1000, 200)
    ansor_tuner(mod, params, ansor_log_file, 1000)

    exec_tvm = compile_tvm(mod, params, autotvm_log_file=autotvm_log_file, additional_outputs=[None, None, None])
    
    return exec_tvm


class XMLCNNProfile:
    def __init__(self, method) -> None:
        self.method = method

    def profile(self, model, inputs):
        # warmup
        for _ in range(20):
            loss = model(*inputs)
            loss.backward()
        
        # profile
        with torch_profile(activities=[ProfilerActivity.CUDA], record_shapes=True) as prof:
            for _ in range(20):
                loss = model(*inputs)
                loss.backward()
        
        print(prof.key_averages().table(sort_by="cuda_time_total"))
    
    def __call__(self) -> Any:
        model = xmlcnnUturn(hidden_dims, y_dim).to("cuda").to(torch.float16).train()
        input = torch.randn(size=(batch_size, hidden_dims), dtype=torch.float16, device="cuda")
        y = torch.empty(
            size=(batch_size, y_dim), 
            dtype=torch.float16, device="cuda").random_(2)
        
        inputs = [input, y]

        if self.method == "evt":
            partition_fn = partial(
                partition_func,
                joint_compiler=joint_optimization
            )
            evt_backend = aot_autograd(
                fw_compiler=compiler_fn, bw_compiler=compiler_fn, partition_fn=partition_fn)
            model = torch.compile(model, fullgraph=True, dynamic=False, backend=evt_backend)
        elif self.method == "triton":
            model = torch.compile(
                model, dynamic=False, backend="inductor", 
                options={"epilogue_fusion": True, "max_autotune": True,
                "trace.enabled": True})

        elif self.method == "bolt":
            partition_fn = partial(
                partition_func,
                joint_compiler=tvm_joint_optimization
            )
            bolt_backend = aot_autograd(
                fw_compiler=compiler_fn, bw_compiler=bolt_compile_fn, partition_fn=partition_fn)
            model = torch.compile(model, fullgraph=True, dynamic=False, backend=bolt_backend)
            
        elif self.method == "tvm":
            partition_fn = partial(
                partition_func,
                joint_compiler=tvm_joint_optimization
            )
            tvm_backend = aot_autograd(
                fw_compiler=compiler_fn, bw_compiler=tvm_compile_fn, partition_fn=partition_fn)
            model = torch.compile(model, fullgraph=True, dynamic=False, backend=tvm_backend)

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
    profiler = XMLCNNProfile(args.method)
    profiler()