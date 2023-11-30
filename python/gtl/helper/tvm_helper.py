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
# This file provides the helper functions for tvm
import tvm
from tvm import autotvm
from tvm import relay, auto_scheduler
import os
import torch


def autotvm_tuner(mod, params, log_file, n_trail, early_stopping):
    # Skip the tuning and directly use the previous result
    if os.path.exists(log_file):
        return
    # Create temperary logfile to reduce the record size
    tmp_log_file = log_file + ".tmp"

    # Clean the previous log file
    if os.path.exists(tmp_log_file):
        os.remove(tmp_log_file)

    host = tvm.target.Target("llvm")
    cuda = tvm.target.Target("cuda -arch=sm_80", host=host)

    with tvm.transform.PassContext(
        opt_level=3
    ):
        tasks = autotvm.task.extract_from_program(
            mod["main"], target=cuda, params=params)
    
    for i, tsk in enumerate(reversed(tasks)):
        prefix = "[Task %2d/%2d] " % (i + 1, len(tasks))
        tuner = autotvm.tuner.XGBTuner(tsk)
        tsk_trial = min(n_trail, len(tsk.config_space))
        tuner.tune(
            n_trial=tsk_trial,
            early_stopping = early_stopping,
            measure_option=autotvm.measure_option(
                builder=autotvm.LocalBuilder(timeout=10),
                runner=autotvm.LocalRunner(number=20, repeat=3, timeout=4, min_repeat_ms=150),
            ),
            callbacks=[
                autotvm.callback.progress_bar(tsk_trial, prefix=prefix),
                autotvm.callback.log_to_file(tmp_log_file),
            ]
        )
    
    autotvm.record.pick_best(tmp_log_file, log_file)
    os.remove(tmp_log_file)


def ansor_tuner(mod, params, log_file, num_measure_trails, skipped_kws=["matmul", "dense"]):
    # Skip the tuning and directly use the previous result
    if os.path.exists(log_file):
        return
    
    host = tvm.target.Target("llvm")
    cuda = tvm.target.Target("cuda -arch=sm_80", host=host)

    with tvm.transform.PassContext(
        opt_level=3, config={"relay.backend.use_auto_scheduler": True}
    ):
        tasks, task_weights = auto_scheduler.extract_tasks(
            mod, params, cuda, opt_level=3, include_simple_tasks=True
        )
        filtered_tasks = []
        filtered_task_weights = []
        for tsk, weight in zip(tasks, task_weights):
            to_skip = False
            for kw in skipped_kws:
                if kw in str(tsk.compute_dag):
                    to_skip = True
                    break
            if to_skip:
                continue

            filtered_tasks.append(tsk)
            filtered_task_weights.append(weight)

        tuner = auto_scheduler.TaskScheduler(filtered_tasks, filtered_task_weights)
        tuner.tune(
            auto_scheduler.TuningOptions(
                num_measure_trials=num_measure_trails,
                measure_callbacks=[
                    auto_scheduler.RecordToFile(log_file),
                ],
            )
        )

def compile_tvm(mod, params, autotvm_log_file=None, ansor_log_file=None, additional_outputs = []):
    if autotvm_log_file and ansor_log_file:
        raise ValueError("autotvm_log_file and ansor_log_file cannot be applied together")
    
    host = tvm.target.Target("llvm")
    cuda = tvm.target.Target("cuda -arch=sm_80", host=host)

    if autotvm_log_file:
        with autotvm.apply_history_best(autotvm_log_file):
            with tvm.transform.PassContext(
                opt_level=3, config={"relay.backend.use_auto_scheduler": True},
            ):
                lib = relay.build(
                    mod,
                    target=cuda,
                    target_host=host,
                    params=params,
                )
    elif ansor_log_file:
        with auto_scheduler.ApplyHistoryBest(ansor_log_file):
            with tvm.transform.PassContext(
                opt_level=3, config={"relay.backend.use_auto_scheduler": True},
            ):
                lib = relay.build(
                    mod,
                    target=cuda,
                    target_host=host,
                    params=params,
                )
    else:
        with tvm.transform.PassContext(opt_level=3):
            lib = relay.build(mod, target=cuda, target_host=host, params=params)
    
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

        return outs + additional_outputs
    return exec_tvm
