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
from gtl.helper import compiler_fn
from gtl.compiler.passes import (
    GTLFrontend, pass_loss_elimination,
    pass_decomposition, pass_cse,
    pass_constant_propagation,
    pass_fusion,
    pass_clean_up,
    pass_print_graph)

from torch._dynamo.backends.common import aot_autograd

from tvm import relay, auto_scheduler
from tvm.contrib import graph_executor
from tvm.relay.op.contrib.cutlass import partition_for_cutlass
from gtl.helper import autotvm_tuner, compile_tvm, ansor_tuner
import tvm
from tvm.contrib.cutlass import (
    num_cutlass_partitions,
    finalize_modules
)
from torch.utils._python_dispatch import _pop_mode_temporarily, _len_torch_dispatch_stack
from contextlib import nullcontext
from tvm import autotvm
import os
import logging

batch_size = 4
seq_len = 512

config_file = "./large.json"

################################################################################
# EVT Passes
################################################################################
def pass_clean_backward_permute(fx_module: torch.fx.GraphModule, _):
    # Step 1: get output node
    output = [node for node in fx_module.graph.nodes if node.target == "output"][0]
    output_nodes = output.args[0]
    permute_nodes = [node for node in output_nodes if node.target == torch.ops.aten.permute]
    for permute in permute_nodes:
        output.replace_input_with(permute, permute.args[0])


def evt_compile_fn(fx_module: torch.fx.GraphModule, _):
    frontend = GTLFrontend()
    fx_module, _ = frontend(fx_module)
    pass_decomposition(fx_module, fx_module.graph)
    pass_cse(fx_module, fx_module.graph)
    pass_constant_propagation(fx_module, fx_module.graph)

    pass_clean_backward_permute(fx_module, fx_module.graph)

    # pass_print_graph(fx_module, "./self_atten.svg")
    pass_fusion(fx_module, fx_module.graph)
    # pass_clean_up(fx_module, fx_module.graph)
    fx_module.recompile()
    
    return fx_module


def autotvm_compile_fn(model, _):
    frontend = GTLFrontend()
    model, _ = frontend(model)
    pass_decomposition(model, model.graph, disabled_ops=["requires_native_dropout"])
    model.graph.eliminate_dead_code()
    model.recompile()
    # pass_print_graph(model, "./tvm_self_atten.svg")
    config = BertConfig.from_json_file(config_file)

    with (_pop_mode_temporarily() 
        if _len_torch_dispatch_stack() > 0 else nullcontext()):
            primals_1 = torch.randn((config.hidden_size, config.hidden_size), dtype=torch.float16, device="cuda")
            primals_2 = torch.randn((config.hidden_size,), dtype=torch.float16, device="cuda")
            primals_3 = torch.randn((config.hidden_size, config.hidden_size), dtype=torch.float16, device="cuda")
            primals_4 = torch.randn((config.hidden_size,), dtype=torch.float16, device="cuda")
            primals_5 = torch.randn((config.hidden_size, config.hidden_size), dtype=torch.float16, device="cuda")
            primals_6 = torch.randn((config.hidden_size,), dtype=torch.float16, device="cuda")
            primals_7 = torch.randn((seq_len, batch_size, config.hidden_size), dtype=torch.float16, device="cuda")
            primals_8 = torch.randn((batch_size, 1, 1, seq_len), dtype=torch.float16, device="cuda")
            
            inputs = [primals_1, primals_2, primals_3, primals_4, primals_5, primals_6, primals_7, primals_8]
            
            model.eval()
            scripted_model = torch.jit.script(model)
            shape_list = [(f"inp_{idx}", i.shape) for idx, i in enumerate(inputs)]
            mod, params = relay.frontend.from_pytorch(scripted_model, shape_list, default_dtype="float16")

            # Aututuner
            autotvm_log_file = "./tvm_autotvm.log"
            autotvm_tuner(mod, params, autotvm_log_file, 1000, 500)

            exec_tvm = compile_tvm(mod, params, autotvm_log_file=autotvm_log_file, additional_outputs=[])
    
    return exec_tvm


def ansor_compile_fn(model, _):
    frontend = GTLFrontend()
    model, _ = frontend(model)
    pass_decomposition(model, model.graph, disabled_ops=["requires_native_dropout"])
    model.graph.eliminate_dead_code()
    model.recompile()
    # pass_print_graph(model, "./tvm_self_atten.svg")
    config = BertConfig.from_json_file(config_file)

    with (_pop_mode_temporarily() 
        if _len_torch_dispatch_stack() > 0 else nullcontext()):
            primals_1 = torch.randn((config.hidden_size, config.hidden_size), dtype=torch.float16, device="cuda")
            primals_2 = torch.randn((config.hidden_size,), dtype=torch.float16, device="cuda")
            primals_3 = torch.randn((config.hidden_size, config.hidden_size), dtype=torch.float16, device="cuda")
            primals_4 = torch.randn((config.hidden_size,), dtype=torch.float16, device="cuda")
            primals_5 = torch.randn((config.hidden_size, config.hidden_size), dtype=torch.float16, device="cuda")
            primals_6 = torch.randn((config.hidden_size,), dtype=torch.float16, device="cuda")
            primals_7 = torch.randn((seq_len, batch_size, config.hidden_size), dtype=torch.float16, device="cuda")
            primals_8 = torch.randn((batch_size, 1, 1, seq_len), dtype=torch.float16, device="cuda")
            
            inputs = [primals_1, primals_2, primals_3, primals_4, primals_5, primals_6, primals_7, primals_8]
            
            model.eval()
            scripted_model = torch.jit.script(model)
            shape_list = [(f"inp_{idx}", i.shape) for idx, i in enumerate(inputs)]
            mod, params = relay.frontend.from_pytorch(scripted_model, shape_list, default_dtype="float16")

            # Aututuner
            ansor_log_file = "./tvm_ansor.log"
            ansor_tuner(mod, params, ansor_log_file, 1000)

            exec_tvm = compile_tvm(mod, params, ansor_log_file=ansor_log_file, additional_outputs=[])
    
    return exec_tvm


class SelfAttenProfile:
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
        config = BertConfig.from_json_file(config_file)
        if self.method in ["autotvm", "ansor"]:
            config.attention_probs_dropout_prob = 0
        model = BertSelfAttention(config).to("cuda").to(torch.float16).train()

        hidden_states = torch.randn(
            (seq_len, batch_size, config.hidden_size), 
            dtype=torch.float16, device="cuda")
        attention_mask = torch.ones(
            size=(batch_size, 1, 1, seq_len), 
            dtype=torch.float16, device="cuda")
        
        inputs = [hidden_states, attention_mask]

        if self.method == "evt":
            evt_backend = aot_autograd(
                fw_compiler=evt_compile_fn, bw_compiler=compiler_fn)
            model = torch.compile(model, dynamic=False, backend=evt_backend)
        elif self.method == "triton":
            model = torch.compile(
                model, dynamic=False, backend="inductor",
                options={"epilogue_fusion": True, "max_autotune": True,
                "trace.enabled": True})#, "trace.graph_diagram": True})
        elif self.method == "bolt":
            # TODO: bolt doesn't find any pattern
            scripted_model = torch.jit.trace(model, inputs).eval()
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
            model = exec_tvm
        elif self.method == "autotvm":
            tvm_backend = aot_autograd(
                fw_compiler=autotvm_compile_fn, bw_compiler=compiler_fn
            )
            model = torch.compile(model, dynamic=False, backend=tvm_backend)
        elif self.method == "ansor":
            tvm_backend = aot_autograd(
                fw_compiler=ansor_compile_fn, bw_compiler=compiler_fn
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
        choices=["torch", "evt", "autotvm",  "ansor", "inductor", "triton", "bolt"])
    args = parser.parse_args()

    ################################################################################
    # logging.basicConfig(level=logging.DEBUG)
    profiler = SelfAttenProfile(args.method)
    profiler()