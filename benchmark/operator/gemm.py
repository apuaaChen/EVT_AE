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
# Benchmarking on Operators
import argparse
import torch
from torch.profiler import ProfilerActivity
from torch.profiler import profile as torch_profile
from torch.fx import symbolic_trace
from torch.fx.passes.shape_prop import ShapeProp
from gtl.compiler.passes import pass_fusion
from tvm import relay, auto_scheduler
from tvm.contrib import graph_executor
from tvm.relay.op.contrib.cutlass import partition_for_cutlass
import tvm
from tvm.contrib.cutlass import (
    num_cutlass_partitions,
    finalize_modules
)
from tvm import autotvm
import os
import logging


class GemmProfile:
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
    
    def __call__(self):
        class MMPartition1(torch.nn.Module):
            def __init__(self) -> None:
                super().__init__()
            
            def forward(self, primals_1, view_11, primals_2):
                permute_24 = torch.ops.aten.permute(primals_1, [1, 0])
                mm_37 = torch.ops.aten.mm(view_11, permute_24)
                add_21 = torch.ops.aten.add(mm_37, primals_2)
                view_12 = torch.ops.aten.view(add_21, [512, 2, 4096])
                gelu = torch.ops.aten.gelu(view_12, approximate="tanh")
                view_13 = torch.ops.aten.view(gelu, [1024, 4096])
                return view_13, view_12
            
        # Inputs
        primals_1 = torch.randn((4096, 1024), dtype=torch.float16, device="cuda")
        view_11 = torch.randn((1024, 1024), dtype=torch.float16, device="cuda")
        primals_2 = torch.randn((4096,), dtype=torch.float16, device="cuda")

        model = MMPartition1()
        inputs = [primals_1, view_11, primals_2]

        if self.method == "evt":
            symbolic_traced: torch.fx.GraphModule = symbolic_trace(model)
            ShapeProp(symbolic_traced).propagate(*inputs)
            pass_fusion(symbolic_traced, symbolic_traced.graph)
            symbolic_traced.recompile()
            model = symbolic_traced
        elif self.method in ["tvm"]:
            # model = torch.compile(model, dynamic=False, backend=self.method)
            scripted_model = torch.jit.trace(model, inputs).eval()
            shape_list = [(f"inp_{idx}", i.shape) for idx, i in enumerate(inputs)]
            mod, params = relay.frontend.from_pytorch(scripted_model, shape_list, default_dtype="float16")
            host = tvm.target.Target("llvm")
            cuda = tvm.target.Target("cuda -arch=sm_80", host=host)
            # Not with ansor
            with tvm.transform.PassContext(opt_level=3):
                lib = relay.build(mod, target=[cuda,], params=params)
            
            # autotvm
            log_file = "./tvm_autotvm.log"

            tmp_log_file = log_file + ".tmp"
            tasks = autotvm.task.extract_from_program(
                mod["main"], target=cuda, params=params)
            
            for i, tsk in enumerate(reversed(tasks)):
                prefix = "[Task %2d/%2d] " % (i + 1, len(tasks))
                tuner = autotvm.tuner.XGBTuner(tsk)
                tsk_trial = min(400, len(tsk.config_space))
                tuner.tune(
                    n_trial=tsk_trial,
                    early_stopping = 10,
                    measure_option=autotvm.measure_option(
                        builder=autotvm.LocalBuilder(timeout=10),
                        runner=autotvm.LocalRunner(number=20, repeat=3, timeout=4, min_repeat_ms=150),
                    ),
                    callbacks=[
                        autotvm.callback.progress_bar(tsk_trial, prefix=prefix),
                        autotvm.callback.log_to_file(tmp_log_file),
                    ]

                )

            # pick best records to a cache file
            autotvm.record.pick_best(tmp_log_file, log_file)
            os.remove(tmp_log_file)

            # With ansor
            # with tvm.transform.PassContext(
            #     opt_level=3, config={"relay.backend.use_auto_scheduler": True}
            # ):
            #     tasks, task_weights = auto_scheduler.extract_tasks(
            #         mod, params, cuda, include_simple_tasks=True, opt_level=3, other_targets=[cutlass]
            #     )

            #     log_file = "./bolt_ansor.log"

            #     # auto-tuning is disabled by default
            #     if ansor_tuning:
            #         measure_ctx = auto_scheduler.LocalRPCMeasureContext(
            #             repeat=3, min_repeat_ms=200, timeout=10
            #         )
            #         tuner = auto_scheduler.TaskScheduler(tasks, task_weights)
            #         tuner.tune(
            #             auto_scheduler.TuningOptions(
            #                 num_measure_trials=100,
            #                 runner=measure_ctx.runner,
            #                 measure_callbacks=[
            #                     auto_scheduler.RecordToFile(log_file),
            #                 ],
            #             )
            #         )

            #     with auto_scheduler.ApplyHistoryBest(log_file):
            #         with tvm.transform.PassContext(
            #             opt_level=3,
            #             config={"relay.backend.use_auto_scheduler": True},
            #         ):
            #             lib = relay.build(
            #                 mod,
            #                 target=cuda,
            #                 target_host=host,
            #                 params=params,
            #             )

            lib = finalize_modules(lib, "compile_tvm.so", "./tmp")
            dev = tvm.device("cuda", 0)
            with autotvm.apply_history_best(log_file):
                with tvm.transform.PassContext(opt_level=3):
                    lib = relay.build_module.build(mod, target=cuda, params=params)
                
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

        elif self.method in ["triton"]:
            model = torch.compile(model, dynamic=False, backend="inductor", mode="max-autotune")
        elif self.method in ["inductor"]:
            model = torch.compile(model, dynamic=False, backend=self.method)
        elif self.method in ["bolt"]:
            scripted_model = torch.jit.trace(model, inputs).eval()
            shape_list = [(f"inp_{idx}", i.shape) for idx, i in enumerate(inputs)]
            mod, params = relay.frontend.from_pytorch(scripted_model, shape_list, default_dtype="float16")
            mod = partition_for_cutlass(mod)
            host = tvm.target.Target("llvm")
            cuda = tvm.target.Target("cuda", host=host)
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
            
            # # autotvm
            # log_file = "./bolt_autotvm.log"
            # tuning_option = {
            #     "log_filename": log_file,
            #     "tuner": "xgb",
            #     "n_trial": 20,
            #     "early_stopping": 10,
            #     "measure_option": autotvm.measure_option(
            #         builder=autotvm.LocalBuilder(timeout=10),
            #         runner=autotvm.LocalRunner(number=20, repeat=3, timeout=4, min_repeat_ms=150),
            #     ),
            # }
            ansor_tuning = True
            # With ansor
            # with tvm.transform.PassContext(
            #     opt_level=3, config={"relay.backend.use_auto_scheduler": True}
            # ):
            #     tasks, task_weights = auto_scheduler.extract_tasks(
            #         mod, params, cuda, include_simple_tasks=True, opt_level=3, other_targets=[cutlass]
            #     )

            #     log_file = "./bolt_ansor.log"

            #     # auto-tuning is disabled by default
            #     if ansor_tuning:
            #         measure_ctx = auto_scheduler.LocalRPCMeasureContext(
            #             repeat=3, min_repeat_ms=200, timeout=10
            #         )
            #         tuner = auto_scheduler.TaskScheduler(tasks, task_weights)
            #         tuner.tune(
            #             auto_scheduler.TuningOptions(
            #                 num_measure_trials=100,
            #                 runner=measure_ctx.runner,
            #                 measure_callbacks=[
            #                     auto_scheduler.RecordToFile(log_file),
            #                 ],
            #             )
            #         )

            #     with auto_scheduler.ApplyHistoryBest(log_file):
            #         with tvm.transform.PassContext(
            #             opt_level=3,
            #             config={"relay.backend.use_auto_scheduler": True},
            #         ):
            #             lib = relay.build(
            #                 mod,
            #                 target=cuda,
            #                 target_host=host,
            #                 params=params,
            #             )

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
    logging.basicConfig(level=logging.DEBUG)
    profiler = GemmProfile(args.method)
    profiler()
