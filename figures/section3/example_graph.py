import torch.nn as nn
import torch
import operator
from torch.fx import symbolic_trace
from torch.fx.passes.shape_prop import ShapeProp
import sys
sys.path.append("/workspace/sparseTraining/Codegen/compiler")
from passes import pass_gemm_fusion, pass_print_graph, pass_mark_epilogue_permutations
from functorch._src.compilers import ts_compile, tensorexpr_compile, tvm_compile
import nvtx
import logging

b = 128
m = 1024
n = 1024
k = 256
logging.basicConfig(level=getattr(logging, "DEBUG"))

class ExampleGraph(nn.Module):
    def __init__(self) -> None:
        super().__init__()
    
    def forward(self, X, weight, bias):
        X_permute = torch.ops.aten.permute(X, [1, 0, 2])
        bmm_out = torch.ops.aten.bmm(X_permute, weight)
        relu = torch.ops.aten.relu(bmm_out)
        add = torch.ops.aten.add(relu, bias)
        relu_permute = torch.ops.aten.permute(add, [1, 0, 2])
        view = torch.ops.aten.view(relu_permute, [4, m//4, b, n])
        reduce = torch.ops.aten.sum(view, [0,1,2])
        
        return view, relu, reduce


module = ExampleGraph()
module_reference = ExampleGraph()

symbolic_traced : torch.fx.GraphModule = symbolic_trace(module)
X = torch.randn((m, b, k), dtype=torch.float16, device="cuda")
W = torch.randn((b, k, n), dtype=torch.float16, device="cuda")
bias = torch.randn((b, 1, n), dtype=torch.float16, device="cuda")

ShapeProp(symbolic_traced).propagate(X, W, bias)

pass_print_graph(symbolic_traced, "./gemm.svg")


pass_mark_epilogue_permutations(symbolic_traced, symbolic_traced.graph)
pass_gemm_fusion(symbolic_traced, symbolic_traced.graph)
symbolic_traced.recompile()
pass_print_graph(symbolic_traced, "./gemm_optimized.svg")





# # try tvm
import tvm
from tvm import relay, auto_scheduler
from tvm.contrib import graph_executor
import os

# describe the graph manually to enable fusion
xr = relay.var("inp_0", relay.TensorType((m, b, k), "float16"))
wr = relay.var("inp_1", relay.TensorType((b, k, n), "float16"))
br = relay.var("inp_2", relay.TensorType((b, 1, n), "float16"))
x_permute_r = relay.transpose(xr, axes=[1, 0, 2])
bmm_out_r = relay.nn.batch_matmul(x_permute_r, wr, transpose_a=False, transpose_b=False)
relu_r = relay.nn.relu(bmm_out_r)
add_r = relay.add(relu_r, br)
relu_permute_r = relay.transpose(add_r, [1, 0, 2])
reshape_r = relay.reshape(relu_permute_r, newshape=[4, m//4, b, n])
sum_r = relay.sum(reshape_r, axis=[0,1,2])
out = relay.Tuple([reshape_r, relu_r, sum_r])

func = relay.Function([xr, wr, br], out)

target = tvm.target.Target("cuda -arch=sm_80")
dev = tvm.cuda(0)
jit_mod = torch.jit.script(symbolic_traced)
example_inputs = [X, W, bias]
shape_list = [(f"inp_{idx}", i.shape) for idx, i in enumerate(example_inputs)]
mod, params = relay.frontend.from_pytorch(jit_mod, shape_list, default_dtype="float16")

mod = tvm.IRModule.from_expr(func)

mod, params = relay.optimize(mod, target, params)

print(mod.astext())

# tasks, task_weights = auto_scheduler.extract_tasks(mod["main"], params, target, include_simple_tasks=False)

# tuning_logfile = "./tuning_log"
# log_file = f"{tuning_logfile}.json"
    
# # if len(tasks) != 0:
# #     tuner = auto_scheduler.TaskScheduler(tasks, task_weights)
# #     tune_option = auto_scheduler.TuningOptions(
# #         num_measure_trials=64*5,
# #         measure_callbacks=[auto_scheduler.RecordToFile(log_file)],
# #         # early_stopping=1000,
# #         # verbose=2,
# #     )
# #     tuner.tune(tune_option)

# assert os.path.exists(log_file)
# with auto_scheduler.ApplyHistoryBest(log_file):
#     with tvm.transform.PassContext(
#         opt_level=4, config={"relay.backend.use_auto_scheduler": True}
#     ):
#         lib = relay.build(mod, target=target, params=params)
# m = graph_executor.GraphModule(lib["default"](dev))
# def exec_tvm(*args):
#     for idx, arg in enumerate(args, 0):
#         if arg.dim() != 0:
#             m.set_input(
#                 f"inp_{idx}",
#                 tvm.nd.from_dlpack(torch.utils.dlpack.to_dlpack(arg.contiguous())),
#             )
#     m.run()
#     outs = [
#         torch.utils.dlpack.from_dlpack(m.get_output(i).to_dlpack())
#         for i in range(m.get_num_outputs())
#     ]
#     return outs
# symbolic_traced = exec_tvm


for i in range(10):
    symbolic_traced(X, W, bias)

for i in range(10):
    with nvtx.annotate("ours"):
        symbolic_traced(X, W, bias)
