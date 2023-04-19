import torch
import torch.nn as nn
import sys
sys.path.append("/workspace/bert")
sys.path.append("/workspace/sparseTraining/Codegen/compiler")
from passes import *
from functorch.compile import aot_module
import logging
from functools import partial
import nvtx
import pycutlass
from pycutlass import *
pycutlass.get_memory_pool(manager="torch")
pycutlass.compiler.nvcc()

class xmlcnnUturn(nn.Module):
    def __init__(self, hidden_dims, y_dim) -> None:
        super().__init__()
        self.out_layer = nn.Linear(hidden_dims, y_dim)
        self.loss_fn = torch.nn.functional.binary_cross_entropy_with_logits

    def forward(self, input, batch_y):
        o = self.out_layer(input)
        loss = self.loss_fn(o, batch_y, reduction="sum") / batch_y.size(0)
        return loss

batch_size = 2048
hidden_dims=512
y_dim=670208//2
model = xmlcnnUturn(hidden_dims, y_dim).to("cuda").to(torch.float16)
input = torch.randn(size=(batch_size, hidden_dims), dtype=torch.float16, device="cuda")
y = torch.empty(
    size=(batch_size, y_dim), 
    dtype=torch.float16, device="cuda").random_(2)

for i in range(30):
    with nvtx.annotate("torch"):
        loss = model(input, y)
        loss.backward()


import torch.fx as fx
from functorch._src.partitioners import _is_primal, _extract_fwd_bwd_outputs, \
    _extract_graph_with_inputs_outputs, _extract_graph_with_inputs_outputs, \
    _extract_fwd_bwd_modules

def partition_func(joint_module: fx.GraphModule, _joint_inputs, pre_partition_optimization):
    pre_partition_optimization(joint_module)

    primal_inputs = list(filter(_is_primal, joint_module.graph.nodes))
    fwd_outputs, bwd_outputs = _extract_fwd_bwd_outputs(joint_module)

    forward_only_graph = _extract_graph_with_inputs_outputs(joint_module.graph, primal_inputs, fwd_outputs)
    forward_node_names = set([node.name for node in forward_only_graph.nodes if node.op != 'output'])

    def node_saved(node):
        return node.name in forward_node_names and 'tensor_meta' in node.meta
    saved_values = [node for node in joint_module.graph.nodes if node_saved(node)]
    return _extract_fwd_bwd_modules(joint_module, saved_values)

def compiler_fn(fx_module: torch.fx.GraphModule, _):
    logging.debug("============Optimized Source Code============")
    logging.debug(fx_module.code)
    return fx_module

def pre_partition_optimization_gtl(joint_module):
    # get graph
    graph = joint_module.graph
    pass_print_graph(joint_module, "./xmlcnn_uturn.svg")

    pass_suffix_elimination(joint_module, graph)

    # pass: loss elimination
    pass_loss_elimination(joint_module, graph)
    disabled_list = [torch.ops.aten.convolution_backward,]
    pass_composed_op_breakdown(joint_module, graph, disabled_list=disabled_list)

    # pass: eliminate expand
    pass_eliminate_transparent_node(
        joint_module, graph, 
        [torch.ops.aten.detach,]
    )

    # pass: constant reduction
    pass_constant_folding(joint_module, graph)

    # pass: strength reduction
    pass_stength_reduction(joint_module, graph)
    pass_gemm_fusion(joint_module, graph)

    joint_module.recompile()
    pass_print_graph(joint_module, "./xmlcnn_gtl.svg")

def pre_partition_optimization_uturn(joint_module):
    # get graph
    graph = joint_module.graph
    pass_print_graph(joint_module, "./xmlcnn_uturn.svg")

    pass_suffix_elimination(joint_module, graph)

    # pass: loss elimination
    pass_loss_elimination(joint_module, graph)
    # disabled_list = [torch.ops.aten.convolution_backward,]
    # pass_composed_op_breakdown(joint_module, graph, disabled_list=disabled_list)

    # # pass: eliminate expand
    # pass_eliminate_transparent_node(
    #     joint_module, graph, 
    #     [torch.ops.aten.detach,]
    # )

    # # pass: constant reduction
    # pass_constant_folding(joint_module, graph)

    # # pass: strength reduction
    # pass_stength_reduction(joint_module, graph)
    # pass_gemm_fusion(joint_module, graph)
    graph.eliminate_dead_code()
    joint_module.recompile()
    pass_print_graph(joint_module, "./xmlcnn_gtl.svg")

model_gtl = aot_module(
    model, compiler_fn, compiler_fn, 
    partial(
        partition_func, 
        pre_partition_optimization=pre_partition_optimization_gtl))

def pre_partition_optimization_tvm(joint_module):
    graph = joint_module.graph
    # pass_print_graph(joint_module, "./xmlcnn_uturn.svg")

    pass_suffix_elimination(joint_module, graph)

    # pass: loss elimination
    pass_loss_elimination(joint_module, graph)
    disabled_list = [torch.ops.aten.convolution_backward,]
    pass_composed_op_breakdown(joint_module, graph, disabled_list=disabled_list)

    # pass: eliminate expand
    pass_eliminate_transparent_node(
        joint_module, graph, 
        [torch.ops.aten.detach,]
    )

    # pass: constant reduction
    pass_constant_folding(joint_module, graph)

    # pass: strength reduction
    pass_stength_reduction(joint_module, graph)
    graph.eliminate_dead_code()
    joint_module.recompile()
    # pass_print_graph(joint_module, "./xmlcnn_tvm.svg")

# tvm
import tvm
from tvm import relay, auto_scheduler
from tvm.contrib import graph_executor
import os

def tvm_compile_fn(fx_module: torch.fx.GraphModule, _):
    graph = fx_module.graph
    for node in graph.nodes:
        if node.op == "output":
            print(node.args)
            node.args = ([node.args[0][0], node.args[0][1]])

    graph.eliminate_dead_code()
    fx_module.recompile()
    jit_mod = torch.jit.script(fx_module)
    primal_1 = torch.randn(size=(670208//2,), dtype=torch.float16, device="cuda")
    primal_2 = torch.randn(size=(670208//2, 512), dtype=torch.float16, device="cuda")
    primal_3 = torch.randn(size=(2048, 512), dtype=torch.float16, device="cuda")
    primal_4 = torch.randn(size=(2048, 670208//2), dtype=torch.float16, device="cuda")
    tangents_1 = torch.randn(size=(1,), dtype=torch.float16, device="cuda")
    example_inputs = [primal_1, primal_2, primal_3, primal_4, tangents_1]
    shape_list = [(f"inp_{idx}", i.shape) for idx, i in enumerate(example_inputs)]
    mod, params = relay.frontend.from_pytorch(jit_mod, shape_list, default_dtype="float16")
    target = tvm.target.Target("cuda -arch=sm_80")
    print(mod.astext())
    mod, params = relay.optimize(mod, target, params)

    # print(mod.astext())

    tasks, task_weights = auto_scheduler.extract_tasks(mod["main"], params, target, include_simple_tasks=False)
    tuning_logfile = "./subgraph_tuning_log"
    log_file = f"{tuning_logfile}.json"
    dev = tvm.cuda(0)

    # if len(tasks) != 0:
    #     tuner = auto_scheduler.TaskScheduler(tasks, task_weights)
    #     tune_option = auto_scheduler.TuningOptions(
    #         num_measure_trials=64*8,
    #         measure_callbacks=[auto_scheduler.RecordToFile(log_file)],
    #         # early_stopping=1000,
    #         # verbose=2,
    #     )
    #     tuner.tune(tune_option)
    assert os.path.exists(log_file)
    with auto_scheduler.ApplyHistoryBest(log_file):
        with tvm.transform.PassContext(
            opt_level=3, config={"relay.backend.use_auto_scheduler": True}
        ):
            lib = relay.build(mod, target=target, params=params)
    m = graph_executor.GraphModule(lib["default"](dev))
    def exec_tvm(*args):
        for idx, arg in enumerate(args, 0):
            if arg.dim() != 0:
                m.set_input(
                    f"inp_{idx}",
                    tvm.nd.from_dlpack(torch.utils.dlpack.to_dlpack(arg.contiguous())),
                )
        m.run()
        outs = [
            torch.utils.dlpack.from_dlpack(m.get_output(i).to_dlpack())
            for i in range(m.get_num_outputs())
        ]

        return outs + [None, None, None]
    return exec_tvm
    # return fx_module

model_tvm = aot_module(
    model, compiler_fn, tvm_compile_fn, 
    partial(
        partition_func, 
        pre_partition_optimization=pre_partition_optimization_tvm))

model_uturn = aot_module(
    model, compiler_fn, compiler_fn, 
    partial(
        partition_func, 
        pre_partition_optimization=pre_partition_optimization_uturn))

for i in range(30):
    with nvtx.annotate("uturn"):
        loss = model_uturn(input, y)
        loss.backward()

for i in range(30):
    with nvtx.annotate("gtl"):
        loss = model_gtl(input, y)
        loss.backward()

for i in range(30):
    with nvtx.annotate("tvm"):
        loss = model_tvm(input, y)
        loss.backward()
