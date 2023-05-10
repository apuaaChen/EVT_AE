################################################################################
# Bert Benchmarking
################################################################################
# Dependencies
import sys
sys.path.append("/workspace/gtl/sparseTraining/Codegen/compiler")
sys.path.append("/workspace/bert")
import torch
from resnet_modeling import ResNet, BasicBlock, Bottleneck
from apex import amp
# from lamb_amp_opt.fused_lamb import FusedLAMBAMP
# from aot_helper import compiler_fn, partition_func
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
from functorch.compile import ts_compile#, tensorexpr_compile, tvm_compile
from resnet_pass_manager import pre_partition_optimization
from gtl.helper import compiler_fn, partition_func, GTLProfiler, BaseTestCase, apex_autocast

import argparse
parser = argparse.ArgumentParser(description="ResNet End-to-End Training with CUDA Graph")
parser.add_argument('--iter', '-it', type=int, default=50, help="Profiling Iterations")
# Hyper-parameter that defines the model size
parser.add_argument('--batch_size', '-b', type=int, default=256, help="Training batch size per GPU")
parser.add_argument('--depth', '-d', type=int, default=50, help="depth of the resnet")
parser.add_argument('--layout', '-l', type=str, default='nhwc', choices=['nhwc', 'nchw'])
parser.add_argument('--method', '-mt', type=str, default="torch", choices=["torch", "nvfuser", "gtl", "tvm"])
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

logging.info(f"ResNet{args.depth} [batch size: {args.batch_size}]")
logging.info(f"Compiler: {args.method}")
if args.cuda_graph:
    logging.info(f"CUDA Graph: ON")
else:
    logging.info(f"CUDA Graph: OFF")

################################################################################
# Initiate inputs
x = torch.randn(size=(args.batch_size, 4, 224, 224), dtype=torch.float16, device="cuda")
y = torch.randint(low=0, high=1000, size=(args.batch_size,), dtype=torch.int64, device="cuda")

learning_rate = 6e-3

def prepare_model_and_optimizer(device, depth, reference=None):
    if depth == 18:
        model = ResNet(block=BasicBlock, layers=[2, 2, 2, 2])
    elif depth == 50:
        model = ResNet(block=Bottleneck, layers=[3, 4, 6, 3])
    elif depth == 101:
        model = ResNet(block=Bottleneck, layers=[3, 4, 23, 3])
    else:
        raise NotImplementedError()
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
    model, optimizer = prepare_model_and_optimizer("cuda", args.depth)
    if args.layout == "nhwc":
        if args.method == "gtl":
            model.to_channels_last()
        else:
            model = model.to(memory_format=torch.channels_last)
    
    if args.method == "gtl":
        keep_batchnorm_fp32=False
    else:
        keep_batchnorm_fp32=True
    
    model, optimizer = apex_autocast(model, optimizer, keep_batchnorm_fp32)

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
    
    if args.cuda_graph:
        model.capture_graph((args.batch_size, 4, 224, 224), optimizer)
    
    ## profiling
    profiler = GTLProfiler(
        model, label=args.method, use_cuda_graph=args.cuda_graph
    )

    profiler.profile(
        [x, y], 
        ("image", args.batch_size)
    )

class ResNetTest(BaseTestCase):
    def test_resnet_50(self):
        sample_inputs = [x, y]
        model, optimizer = prepare_model_and_optimizer("cuda", 18)
        model, optimizer = apex_autocast(model, optimizer, False)

        self.run_reference_model(model, optimizer, sample_inputs, 100.)

        model_fused, optimizer_fused = prepare_model_and_optimizer("cuda", 18, reference=model)
        model_fused, optimizer_fused = apex_autocast(model_fused, optimizer_fused, False)
        model_fused.train()
        model_fused.to_channels_last()

        model_fused.aot_optimize(
            compiler_fn, compiler_fn, 
            partial(
                partition_func, 
                joint_compiler=pre_partition_optimization
            )
        )

        model_fused.capture_graph((args.batch_size, 4, 224, 224), optimizer_fused)
        self.run_target_model(model_fused, optimizer_fused, sample_inputs)

        self.verify(model, model_fused, verbose=0, rtol=3e-1)
    
    def grad_preprocess(self, grad):
        if len(grad.size()) == 4:
            K, C ,R, S = grad.size()
            grad = grad.view(K, R, S, C).permute(0, 3, 1, 2).contiguous().to(torch.float16)
        else:
            grad = grad.contiguous().to(torch.float16)
        return grad
    
    def is_close(self, grad1, grad2):
        return torch.sum(torch.isclose(grad1, grad2, rtol=3e-1)) / grad1.numel() > 0.6
        

if __name__ == '__main__':
    if args.unittest: 
        runner = unittest.TextTestRunner()
        itersuite = unittest.TestLoader().loadTestsFromTestCase(ResNetTest)
        runner.run(itersuite)
    else: 
        benchmarking()