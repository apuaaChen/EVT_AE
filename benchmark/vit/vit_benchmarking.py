################################################################################
# Vit Benchmarking
################################################################################
# Dependencies
import sys
sys.path.append("/workspace/sparseTraining/Codegen/compiler")
sys.path.append("/workspace/bert")
import torch
from vit_modeling import ViT, Attention, TransformerLayer
from apex import amp
from lamb_amp_opt.fused_lamb import FusedLAMBAMP
from aot_helper import compiler_fn, partition_func, compiler_fn_nvfuser
from functorch._src.compilers import ts_compile, tensorexpr_compile, tvm_compile
import pycutlass
from pycutlass import *
pycutlass.get_memory_pool(manager="torch")
pycutlass.compiler.nvcc()
import nvtx
import argparse
import logging
from pycutlass.test.profiler import GpuTimer
from functools import partial
import unittest

################################################################################
# parse args
parser = argparse.ArgumentParser(description="Vit End-to-End Training with CUDA Graph")
parser.add_argument('--iter', '-it', type=int, default=50, help="Profiling Iterations")
# Hyper-parameter that defines the model size
parser.add_argument('--batch_size', '-b', type=int, default=128, help="Training batch size per GPU")
# method
parser.add_argument('--method', '-mt', type=str, default="torch", choices=["torch", "east", "nvfuser", "flash", "lightseq"])
parser.add_argument('--passes', '-ps', nargs='+', default=[], help="passes enabled in EAST")
parser.add_argument('--cuda_graph', '-cg', action="store_true", help="using cuda graph")
# logging
parser.add_argument('--log', '-lg', type=str, default="INFO", choices=["INFO", "DEBUG"])
# unittest
parser.add_argument('--unittest', '-ut', action="store_true", help="run unittest")
args = parser.parse_args()

################################################################################
# logging setting
if args.unittest: logging.disable(logging.INFO)
logging.basicConfig(level=getattr(logging, args.log))
logging.basicConfig(format='%(message)s')

logging.info(f"Vit[batch size: {args.batch_size}]")
logging.info(f"Compiler: {args.method}")
if args.cuda_graph:
    logging.info(f"CUDA Graph: ON")
else:
    logging.info(f"CUDA Graph: OFF")

class Config:
    def __init__(self) -> None:
        self.hidden_size=768
        self.num_attention_heads=12
        self.attention_probs_dropout_prob=1e-16
        self.max_position_embeddings = 256
        self.intermediate_size = 3072
        self.hidden_dropout_prob=1e-16
        self.hidden_act = "gelu"

config = Config()

################################################################################
# Initiate inputs
x = torch.randn(size=(args.batch_size, 3, 224, 224), dtype=torch.float16, device="cuda")
y = torch.randint(low=0, high=1000, size=(args.batch_size,), dtype=torch.int64, device="cuda")

learning_rate = 6e-3

def prepare_model_and_optimizer(device, reference=None):
    model = ViT(
        image_size=224,
        patch_size=14,
        num_classes=1000,
        dim=768,
        depth=12,
        heads=12,
        mlp_dim=3072,
        dropout=1e-16,
        emb_dropout=1e-16
    )
    if reference is not None:
        reference_embedding_state_dict = reference.state_dict()
        model.load_state_dict(reference_embedding_state_dict)
    model.to(device)

    param_optimizer = list(model.named_parameters())
    no_decay = ['bias', 'gamma', 'beta']

    optimizer_grouped_parameters = [
        {'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)], 'weight_decay': 0.0},
        {'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}]

    optimizer = FusedLAMBAMP(optimizer_grouped_parameters,
                             lr=learning_rate)
    
    optimizer.setup_fp32_params()

    return model, optimizer


################################################################################
# Slapo passes

from vit_modeling_slapo import xFlashAttention, LSBertEmbeddings, LSBertLayer

# pass 1: using flash attention
def pass_using_flash_attention(sch):
    for subsch in sch.child:
        if isinstance(sch[subsch].mod, Attention):
            new_mod = xFlashAttention(config)
            sch[subsch].replace(new_mod)
        else:
            pass_using_flash_attention(sch[subsch])

# pass 2: using lightseq2 embedding
# def pass_using_lightseq2_embedding(sch):
#     for subsch in sch.child:
#         if isinstance(sch[subsch].mod, modeling.BertEmbeddings):
#             new_mod = LSBertEmbeddings(config)
#             sch[subsch].replace(new_mod)
#         else:
#             pass_using_lightseq2_embedding(sch[subsch])

# pass 3: using lightseq2 encoder layer
def pass_using_lightseq2_encoder_layer(sch):
    for subsch in sch.child:
        if isinstance(sch[subsch].mod, TransformerLayer):
            new_mod = LSBertLayer(config)
            sch[subsch].replace(new_mod)
        else:
            pass_using_lightseq2_encoder_layer(sch[subsch])


def benchmarking():
    ## create model and optimizer
    model, optimizer = prepare_model_and_optimizer("cuda")

    logging.info("AMP: ON")
    model, optimizer = amp.initialize(
        model, optimizer, 
        cast_model_outputs=torch.float16, 
        opt_level="O2", keep_batchnorm_fp32=False, 
        loss_scale="dynamic", verbosity=0
    )

    model.train()

    if args.method == "east":
        model.aot_optimize(compiler_fn, compiler_fn, partial(partition_func, enabled_passes=args.passes))
    elif args.method == "nvfuser":
        model.aot_optimize(ts_compile, compiler_fn_nvfuser)
    elif args.method == "tvm":
        raise NotImplementedError("TVM has issue supporting various ops like layernorm_backward. Additional effort is required to register it to relay")
        tvm_compiler = tvm_compile(target="cuda -arch=sm_80", use_ansor_tuning=True, tuning_logfile="./tuning_log")
        model.aot_optimize(tvm_compiler, tvm_compiler)
    elif args.method == "flash":
        print("using flash")
        import slapo

        sch = slapo.create_schedule(model, group=None)
        pass_using_flash_attention(sch)
        # pass_using_lightseq2_embedding(sch)

        model, _ = slapo.build(sch)
        model.to(torch.float16).to("cuda")
    elif args.method == "lightseq":
        raise NotImplementedError("LightSeq only support up to 16384 batch token")
        import slapo
        sch = slapo.create_schedule(model, group=None)
        # pass_using_lightseq2_embedding(sch)
        pass_using_lightseq2_encoder_layer(sch)
        model, _ = slapo.build(sch)
        model.to(torch.float16).to("cuda")
    
    timer = GpuTimer()

    if args.cuda_graph:
        model.capture_graph((args.batch_size, 3, 224, 224), optimizer=optimizer) 
        
    ## profiling
    s = torch.cuda.Stream(priority=-1)
    s.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(s):
        if args.cuda_graph:
            for i in range(10):
                model.train_with_graph(x, y)
            torch.cuda.synchronize()
            timer.start(stream=s.cuda_stream)
            with nvtx.annotate("40 iter"):
                for i in range(args.iter):
                    with nvtx.annotate("1 iter"):
                        model.train_with_graph(x, y)
            timer.stop_and_wait(stream=s.cuda_stream)
        else:
            for i in range(10):
                loss = model(x, y) * 1e+2
                loss.backward()
            # torch.cuda.synchronize()
            s.synchronize()
            timer.start(stream=s.cuda_stream)
            with nvtx.annotate("40 iter"):
                for i in range(args.iter):
                    with nvtx.annotate("1 iter"):
                        loss = model(x, y)
                        loss.backward()
            timer.stop_and_wait(stream=s.cuda_stream)


    iter_time = timer.duration(args.iter)
    throughput = args.batch_size / iter_time * 1000
    logging.info(f"Throughput: {throughput} img/sec, Iter time: {iter_time} ms")


class VitTest(unittest.TestCase):
    def test_vit_14(self):
        model, optimizer = prepare_model_and_optimizer("cuda")
        model, optimizer = amp.initialize(
            model, optimizer, 
            cast_model_outputs=torch.float16, 
            opt_level="O2", keep_batchnorm_fp32=True, 
            loss_scale="dynamic", verbosity=0
        )
        model.train()
        optimizer.zero_grad()
        loss = model(x, y) * 1e+2
        loss.backward()

        model_fused, optimizer_fused = prepare_model_and_optimizer("cuda", reference=model)
        model_fused, optimizer_fused = amp.initialize(
            model_fused, optimizer_fused, 
            cast_model_outputs=torch.float16, 
            opt_level="O2", keep_batchnorm_fp32=False, 
            loss_scale="dynamic", verbosity=0
        )

        model_fused.train()
        model_fused.aot_optimize(compiler_fn, compiler_fn, partition_func)
        model_fused.capture_graph((args.batch_size, 3, 224, 224), optimizer_fused)

        optimizer_fused.zero_grad()
        model_fused.train_with_graph(x, y)
        
        for param1, param2 in zip(
            list(model.named_parameters()), 
            list(model_fused.named_parameters())):
            # print(torch.sum(torch.isclose(param1[1].grad, param2[1].grad, rtol=1e-1)) / param2[1].grad.numel())
            self.assertTrue(
                (torch.sum(torch.isclose(param1[1].grad, param2[1].grad, rtol=1e-1)) / param2[1].grad.numel() > 0.9) or 
                "bias" in param1[0])
                # torch.allclose(param1[1].grad, param2[1].grad, atol=1e-3))



if __name__ == '__main__':
    if args.unittest: 
        pass
        runner = unittest.TextTestRunner()
        itersuite = unittest.TestLoader().loadTestsFromTestCase(VitTest)
        runner.run(itersuite)
    else: 
        benchmarking()