import torch
import sys
sys.path.append("/workspace/gtl/sparseTraining/thirdparty/DeepLearningExample/PyTorch/LanguageModeling/BERT")
from modeling import BertSelfAttention, BertConfig
from torch.fx import symbolic_trace
from torch.fx.passes.shape_prop import ShapeProp
sys.path.append("/workspace/gtl/sparseTraining/Codegen/compiler")
from passes import *
from functorch.compile import aot_module
import logging
import nvtx

batch_size = 32
seq_len = 512

config_file = "./large.json"
config = BertConfig.from_json_file(config_file)

model = BertSelfAttention(config).to("cuda").to(torch.float16)

def chimera_compile_fn(fx_module: torch.fx.GraphModule, _):
    pass_print_graph(fx_module, "./self_attention.svg")
    graph = fx_module.graph

    pass_suffix_elimination(fx_module, graph)

    pass_eliminate_transparent_node(
        fx_module, graph, 
        [torch.ops.aten.detach,]
    )
    pass_composed_op_breakdown(fx_module, graph)
    pass_remove_duplicated_node(fx_module, graph)
    pass_merge_common_factor(fx_module, graph)
    pass_update_attributes(fx_module, graph)
    pass_constant_folding(fx_module, graph)
    pass_trans_2_permute(fx_module, graph)
    pass_update_attributes(fx_module, graph)
    pass_stength_reduction(fx_module, graph)
    pass_mark_epilogue_permutations(fx_module, graph)
    pass_layernorm_preprocessing(fx_module, graph)
    pass_gemm_fusion(fx_module, graph)
    fx_module.recompile()

    pass_print_graph(fx_module, "./self_attention_chimera.svg")
    return fx_module


def bw_compiler_fn(fx_module: torch.fx.GraphModule, _):
    logging.debug("============Optimized Source Code============")
    logging.debug(fx_module.code)
    return fx_module

# symbolic_traced : torch.fx.GraphModule = symbolic_trace(model)

hidden_state = torch.randn(size=(seq_len, batch_size, config.hidden_size), dtype=torch.float16, device="cuda")
attention_mask = torch.ones(size=(batch_size, 1, 1, seq_len), dtype=torch.float16, device="cuda")

# ShapeProp(symbolic_traced).propagate(hidden_state, attention_mask)
model_chimera = aot_module(model, chimera_compile_fn, bw_compiler_fn)

for i in range(10):
    with nvtx.annotate("torch"):
        model(hidden_state, attention_mask)

for i in range(10):
    with nvtx.annotate("chimera"):
        model_chimera(hidden_state, attention_mask)

# try tvm
import tvm
from tvm import relay, auto_scheduler
from tvm.contrib import graph_executor
import os

def tvm_compile_fn(fx_module: torch.fx.GraphModule, _):
    graph = fx_module.graph
    pass_suffix_elimination(fx_module, graph)
    pass_tvm_preprocessing(fx_module, graph)
    fx_module.recompile()
    jit_mod = torch.jit.script(fx_module)
    primal_1 = torch.randn(size=(1024,), dtype=torch.float16, device="cuda")
    primal_2 = torch.randn(size=(1024, 1024), dtype=torch.float16, device="cuda")
    primal_3 = torch.randn(size=(1024,), dtype=torch.float16, device="cuda")
    primal_4 = torch.randn(size=(1024, 1024), dtype=torch.float16, device="cuda")
    primal_5 = torch.randn(size=(1024,), dtype=torch.float16, device="cuda")
    primal_6 = torch.randn(size=(1024, 1024), dtype=torch.float16, device="cuda")
    primal_7 = hidden_state
    primal_8 = attention_mask
    example_inputs = [primal_1, primal_2, primal_3, primal_4, primal_5, primal_6, primal_7, primal_8]
    shape_list = [(f"inp_{idx}", i.shape) for idx, i in enumerate(example_inputs)]
    mod, params = relay.frontend.from_pytorch(jit_mod, shape_list, default_dtype="float16")
    target = tvm.target.Target("cuda -arch=sm_80")
    mod, params = relay.optimize(mod, target, params)

    print(mod.astext())

    tasks, task_weights = auto_scheduler.extract_tasks(mod["main"], params, target, include_simple_tasks=False)
    tuning_logfile = "./subgraph_tuning_log"
    log_file = f"{tuning_logfile}.json"
    dev = tvm.cuda(0)

    # if len(tasks) != 0:
    #     tuner = auto_scheduler.TaskScheduler(tasks, task_weights)
    #     tune_option = auto_scheduler.TuningOptions(
    #         num_measure_trials=64*32,
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
        return outs
    return exec_tvm

model_tvm = aot_module(model, tvm_compile_fn, bw_compiler_fn)

for i in range(10):
    with nvtx.annotate("tvm"):
        model_tvm(hidden_state, attention_mask)

# from functorch._src.compilers import tensorexpr_compile
# model_tc = aot_module(model, tensorexpr_compile, bw_compiler_fn).to("cuda")
# for i in range(10):
#     with nvtx.annotate("tensorexpr"):
#         model_tc(hidden_state, attention_mask)

# ################################################################################
# # AI Template based Implementation
# from aitemplate.frontend import nn, Tensor
# from aitemplate.compiler import compile_model, ops
# from aitemplate.compiler.ops.common.epilogue import FuncEnum
# from aitemplate.testing import detect_target

# # create graph
# primals_7 = Tensor(shape=(512, 32, 1024), dtype="float16", name="primals_7", is_input=True)
# view_default = ops.reshape()(primals_7, shape=[16384, 1024])
# primals_3 = Tensor(shape=(1024,), dtype="float16", name="primals_3", is_input=True)
# primals_4 = Tensor(shape=(1024, 1024), dtype="float16", name="primals_4", is_input=True)
# addmm_default = ops.gemm_rcr_bias(view_default, primals_4, primals_3)
# primals_2 = Tensor(shape=(1024, 1024), dtype="float16", name="primals_2", is_input=True)
# primals_1 = Tensor(shape=(1024,), dtype="float16", name="primals_1", is_input=True)
# addmm_default_1 = ops.gemm_rcr_bias(view_default, primals_2, primals_1)
# view_default_1 = ops.reshape()(addmm_default, shape=[512, 32, 1024])
# view_default_6 = ops.reshape()(view_default_1, shape=[512, 512, 64])
# transpose_int = ops.permute102()(view_default_6)
# view_default_3 = ops.reshape()(addmm_default_1, shape=[512, 32, 1024])
# view_default_7 = ops.reshape()(view_default_3, shape=[512, 512, 64])
# permute_default = ops.permute



