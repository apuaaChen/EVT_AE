import torch
import torch.nn as nn
import sys
sys.path.append("/workspace/bert")
sys.path.append("/workspace/gtl/sparseTraining/Codegen/compiler")
from passes import *
from functorch.compile import aot_module
import logging
from functools import partial
import nvtx
import pycutlass
from pycutlass import *
pycutlass.get_memory_pool(manager="torch")
pycutlass.compiler.nvcc()
from resnet_modeling import conv3x3
from apex import amp
# from lamb_amp_opt.fused_lamb import FusedLAMBAMP

logging.basicConfig(level=logging.DEBUG)
logging.basicConfig(format='%(message)s')

class ConvBnReLU(nn.Module):
    def __init__(self, inplanes, outplanes) -> None:
        super().__init__()
        self.conv = conv3x3(inplanes, outplanes)
        self.bn1 = nn.BatchNorm2d(outplanes, track_running_stats=False)
        self.relu = nn.ReLU()
    
    def forward(self, x:Tensor) -> Tensor:
        out = self.conv(x)
        out = self.bn1(out)
        out = self.relu(out)
        return out

batch_size = 128
inplanes = 128
outplanes = 128
H = W = 28

model = ConvBnReLU(inplanes, outplanes).to("cuda")

param_optimizer = list(model.named_parameters())
no_decay = ['bias', 'gamma', 'beta']

optimizer_grouped_parameters = [
    {'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)], 'weight_decay': 0.0},
    {'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}]

optimizer = torch.optim.SGD(optimizer_grouped_parameters, 1e-6)
# optimizer = FusedLAMBAMP(optimizer_grouped_parameters,
#                             lr=1e-6)

# optimizer.setup_fp32_params()

model, optimizer = amp.initialize(
        model, optimizer, 
        cast_model_outputs=torch.float16, 
        opt_level="O2", keep_batchnorm_fp32=False, 
        loss_scale="dynamic", verbosity=0
    )

input = torch.randn(size=(batch_size, inplanes, H, W), dtype=torch.float16, device="cuda")

def gtl_compile_fn(fx_module: torch.fx.GraphModule, _):
    pass_print_graph(fx_module, "./self_attention.svg")
    graph = fx_module.graph
    pass_suffix_elimination(fx_module, graph)
    pass_eliminate_transparent_node(
        fx_module, graph, 
        [torch.ops.aten.detach, torch.ops.aten.as_strided_.default]
    )
    pass_composed_op_breakdown(fx_module, graph)
    pass_remove_duplicated_node(fx_module, graph)
    pass_merge_common_factor(fx_module, graph)
    pass_update_attributes(fx_module, graph)
    pass_constant_folding(fx_module, graph)
    pass_stength_reduction(fx_module, graph)
    pass_batchnorm_preprocessing(fx_module, graph)
    pass_conv_fusion(fx_module, graph)
    pass_gemm_fusion(fx_module, graph)
    fx_module.recompile()
    pass_print_graph(fx_module, "./self_attention_chimera.svg")
    return fx_module

def bw_compiler_fn(fx_module: torch.fx.GraphModule, _):
    logging.debug("============Optimized Source Code============")
    logging.debug(fx_module.code)
    return fx_module


model_gtl = aot_module(model, gtl_compile_fn, bw_compiler_fn)

model_ref = ConvBnReLU(inplanes, outplanes).to("cuda")
param_optimizer_ref = list(model.named_parameters())
no_decay_ref = ['bias', 'gamma', 'beta']

optimizer_grouped_parameters_ref = [
    {'params': [p for n, p in param_optimizer_ref if not any(nd in n for nd in no_decay_ref)], 'weight_decay': 0.0},
    {'params': [p for n, p in param_optimizer_ref if any(nd in n for nd in no_decay_ref)], 'weight_decay': 0.0}]

optimizer_ref = torch.optim.SGD(optimizer_grouped_parameters, 1e-6)
# optimizer_ref = FusedLAMBAMP(optimizer_grouped_parameters_ref,
#                             lr=1e-6)

# optimizer_ref.setup_fp32_params()

model_ref, optimizer_ref = amp.initialize(
        model_ref, optimizer_ref, 
        cast_model_outputs=torch.float16, 
        opt_level="O2", keep_batchnorm_fp32=True, 
        loss_scale="dynamic", verbosity=0
    )
model_ref.to(memory_format=torch.channels_last)

for i in range(10):
    with nvtx.annotate("torch"):
        model_ref(input)

for i in range(10):
    with nvtx.annotate("gtl"):
        model_gtl(input)


# # Try TVM
# import tvm
# from tvm import relay, auto_scheduler
# from tvm.contrib import graph_executor
# import os

# def tvm_compile_fn(fx_module: torch.fx.GraphModule, _):
#     graph = fx_module.graph
#     pass_suffix_elimination(fx_module, graph)
#     pass_tvm_preprocessing(fx_module, graph)
#     fx_module.recompile()
#     jit_mod = torch.jit.script(fx_module)
#     primal_1 = torch.randn(size=(128,), dtype=torch.float16, device="cuda")
#     primal_2 = torch.randn(size=(128,), dtype=torch.float16, device="cuda")
#     primal_3 = torch.randn(size=(128, 128, 3, 3), dtype=torch.float16, device="cuda")
#     primal_4 = torch.zeros(size=(1,), dtype=torch.int64, device="cuda")
#     primal_5 = torch.randn(size=(128,), dtype=torch.float16, device="cuda")
#     primal_6 = torch.randn(size=(128,), dtype=torch.float16, device="cuda")
#     primal_7 = torch.randn(size=(128,128,28,28), dtype=torch.float16, device="cuda")
#     example_inputs = [primal_1, primal_2, primal_3, primal_4, primal_5, primal_6, primal_7]
#     shape_list = [(f"inp_{idx}", i.shape) for idx, i in enumerate(example_inputs)]
#     mod, params = relay.frontend.from_pytorch(jit_mod, shape_list, default_dtype="float16")
#     target = tvm.target.Target("cuda -arch=sm_80")
#     mod, params = relay.optimize(mod, target, params)

#     # print(mod.astext())

#     tasks, task_weights = auto_scheduler.extract_tasks(mod["main"], params, target, include_simple_tasks=False)
#     tuning_logfile = "./subgraph_tuning_log"
#     log_file = f"{tuning_logfile}.json"
#     dev = tvm.cuda(0)

#     # if len(tasks) != 0:
#     #     tuner = auto_scheduler.TaskScheduler(tasks, task_weights)
#     #     tune_option = auto_scheduler.TuningOptions(
#     #         num_measure_trials=64*32,
#     #         measure_callbacks=[auto_scheduler.RecordToFile(log_file)],
#     #         # early_stopping=1000,
#     #         # verbose=2,
#     #     )
#     #     tuner.tune(tune_option)
#     assert os.path.exists(log_file)
#     with auto_scheduler.ApplyHistoryBest(log_file):
#         with tvm.transform.PassContext(
#             opt_level=3, config={"relay.backend.use_auto_scheduler": True}
#         ):
#             lib = relay.build(mod, target=target, params=params)
#     m = graph_executor.GraphModule(lib["default"](dev))
#     def exec_tvm(*args):
#         for idx, arg in enumerate(args, 0):
#             if arg.dim() != 0:
#                 m.set_input(
#                     f"inp_{idx}",
#                     tvm.nd.from_dlpack(torch.utils.dlpack.to_dlpack(arg.contiguous())),
#                 )
#         m.run()
#         outs = [
#             torch.utils.dlpack.from_dlpack(m.get_output(i).to_dlpack())
#             for i in range(m.get_num_outputs())
#         ]
#         return outs
#     return exec_tvm
#     # return fx_module

# model_tvm = aot_module(model, tvm_compile_fn, bw_compiler_fn)

# for i in range(10):
#     with nvtx.annotate("tvm"):
#         model_tvm(input)