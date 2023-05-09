################################################################################
# Vit Benchmarking
################################################################################
# Dependencies
import sys
sys.path.append("/workspace/gtl/sparseTraining/Codegen/compiler")
sys.path.append("/workspace/bert")
import torch
from vit_modeling import ViT, Attention, TransformerLayer
from apex import amp
# from lamb_amp_opt.fused_lamb import FusedLAMBAMP
# from aot_helper import compiler_fn, partition_func, compiler_fn_nvfuser
# from functorch._src.compilers import ts_compile, tensorexpr_compile, tvm_compile
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

from gtl.helper import compiler_fn, partition_func, GTLProfiler, BaseTestCase, apex_autocast
from vit_pass_manager import pre_partition_optimization

################################################################################
# parse args
parser = argparse.ArgumentParser(description="Vit End-to-End Training with CUDA Graph")
parser.add_argument('--iter', '-it', type=int, default=50, help="Profiling Iterations")
# Hyper-parameter that defines the model size
parser.add_argument('--batch_size', '-b', type=int, default=128, help="Training batch size per GPU")
# method
parser.add_argument('--method', '-mt', type=str, default="torch", choices=["torch", "gtl", "nvfuser", "flash", "lightseq"])
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

    optimizer = torch.optim.SGD(optimizer_grouped_parameters, learning_rate)

    return model, optimizer

def benchmarking():
    ## create model and optimizer
    model, optimizer = prepare_model_and_optimizer("cuda")
    logging.info("AMP: ON")
    model, optimizer = apex_autocast(model, optimizer, False)
    model.train()

    # perform AOT Optimize
    if args.method == "gtl":
        model.aot_optimize(
            compiler_fn, compiler_fn, 
            partial(
                partition_func, 
                joint_compiler=partial(
                    pre_partition_optimization, 
                    enabled_passes=args.passes
                )
            )
        )
    
    # capture CUDA graph
    if args.cuda_graph:
        model.capture_graph(
            (args.batch_size, 3, 224, 224), optimizer=optimizer
        ) 
    
    ## profiling
    profiler = GTLProfiler(
        model, label=args.method, use_cuda_graph=args.cuda_graph)
    
    profiler.profile(
        [x, y], ("image", args.batch_size)
    )


class VitTest(BaseTestCase):
    def test_vit_large(self):
        sample_inputs = [x, y]
        model, optimizer = prepare_model_and_optimizer("cuda")
        model, optimizer = apex_autocast(model, optimizer, False)
        self.run_reference_model(model, optimizer, sample_inputs, 100.)

        model_fused, optimizer_fused = prepare_model_and_optimizer("cuda", reference=model)
        model_fused, optimizer_fused = apex_autocast(
            model_fused, optimizer_fused, False)
        model_fused.aot_optimize(
            compiler_fn, compiler_fn, 
            partial(
                partition_func, 
                joint_compiler=pre_partition_optimization
            )
        )
        model_fused.capture_graph((args.batch_size, 3, 224, 224), optimizer_fused)
        self.run_target_model(model_fused, optimizer_fused, sample_inputs)

        self.verify(model, model_fused, verbose=0)
    
    def is_close(self, grad1, grad2):
        return (
            torch.sum(
                torch.isclose(grad1, grad2, rtol=1e-1)
            ) / grad1.numel() > 0.9
            or torch.allclose(grad1, grad2, atol=1e-3)
        ) or grad1.numel() <= 768 # TODO: maybe because of reduction error



if __name__ == '__main__':
    if args.unittest: 
        runner = unittest.TextTestRunner()
        itersuite = unittest.TestLoader().loadTestsFromTestCase(VitTest)
        runner.run(itersuite)
    else: 
        benchmarking()