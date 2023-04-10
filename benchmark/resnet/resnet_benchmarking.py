################################################################################
# Bert Benchmarking
################################################################################
# Dependencies
import sys
sys.path.append("/workspace/sparseTraining/Codegen/compiler")
sys.path.append("/workspace/bert")
import torch
from resnet_modeling import ResNet, BasicBlock, Bottleneck
from apex import amp
from lamb_amp_opt.fused_lamb import FusedLAMBAMP
from aot_helper import compiler_fn, partition_func
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
from functorch._src.compilers import ts_compile, tensorexpr_compile, tvm_compile

import argparse
parser = argparse.ArgumentParser(description="ResNet End-to-End Training with CUDA Graph")
parser.add_argument('--iter', '-it', type=int, default=50, help="Profiling Iterations")
# Hyper-parameter that defines the model size
parser.add_argument('--batch_size', '-b', type=int, default=128, help="Training batch size per GPU")
parser.add_argument('--depth', '-d', type=int, default=18, help="depth of the resnet")
parser.add_argument('--layout', '-l', type=str, default='nhwc', choices=['nhwc', 'nchw'])
parser.add_argument('--method', '-mt', type=str, default="torch", choices=["torch", "nvfuser", "east", "tvm"])
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

    optimizer = FusedLAMBAMP(optimizer_grouped_parameters,
                             lr=learning_rate)
    
    optimizer.setup_fp32_params()

    return model, optimizer

def benchmarking():
    model, optimizer = prepare_model_and_optimizer("cuda", args.depth)
    if args.layout == "nhwc":
        if args.method == "east":
            model.to_channels_last()
        else:
            model = model.to(memory_format=torch.channels_last)

    if args.method == "east":
        keep_batchnorm_fp32=False
    else:
        keep_batchnorm_fp32=True
    model, optimizer = amp.initialize(
        model, optimizer, 
        cast_model_outputs=torch.float16, 
        opt_level="O2", keep_batchnorm_fp32=keep_batchnorm_fp32, 
        loss_scale="dynamic", verbosity=0
    )
    model.train()

    if args.method == "east":
        model.aot_optimize(compiler_fn, compiler_fn, partial(partition_func, enabled_passes=args.passes))
    elif args.method == "nvfuser":
        model.aot_optimize(ts_compile, ts_compile)
    elif args.method == "tvm":
        raise NotImplementedError("TVM has issue supporting various ops like layernorm_backward. Additional effort is required to register it to relay")
        tvm_compiler = tvm_compile(target="cuda -arch=sm_80", use_ansor_tuning=True, tuning_logfile="./tuning_log")
        model.aot_optimize(tvm_compiler, tvm_compiler)
    
    timer = GpuTimer()
    if args.cuda_graph:
        model.capture_graph((args.batch_size, 4, 224, 224), optimizer)
    

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
                        loss = model(x, y) * 1e+2
                        loss.backward()
            timer.stop_and_wait(stream=s.cuda_stream)


    iter_time = timer.duration(args.iter)
    throughput = args.batch_size / iter_time * 1000
    logging.info(f"Throughput: {throughput} img/sec, Iter time: {iter_time} ms")


class ResNetTest(unittest.TestCase):
    def test_resnet18(self):
        # reference model
        model, optimizer = prepare_model_and_optimizer("cuda", 18)
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

        model_fused, optimizer_fused = prepare_model_and_optimizer(
            "cuda", 18, reference=model)
        model_fused, optimizer_fused = amp.initialize(
            model_fused, optimizer_fused, 
            cast_model_outputs=torch.float16, 
            opt_level="O2", keep_batchnorm_fp32=False, 
            loss_scale="dynamic", verbosity=0
        )

        model_fused.train()
        model_fused.to_channels_last()
        model_fused.aot_optimize(compiler_fn, compiler_fn, partition_func)
        model_fused.capture_graph((args.batch_size, 4, 224, 224), optimizer_fused)
        optimizer_fused.zero_grad()
        model_fused.train_with_graph(x, y)

        for param1, param2 in zip(list(model.named_parameters()), list(model_fused.named_parameters())):
            grad_origin = param1[1].grad.to(torch.float16)
            if len(grad_origin.size()) == 4:
                K, C, R, S = grad_origin.size()
                grad_fused = param2[1].grad.view(K, R, S, C).permute(0, 3, 1, 2).contiguous()
            else:
                grad_fused = param2[1].grad.contiguous()
            # print(torch.sum(torch.isclose(grad_origin, grad_fused, rtol=3e-1)) / grad_origin.numel())
            self.assertTrue(torch.sum(torch.isclose(grad_origin, grad_fused, rtol=3e-1)) / grad_origin.numel() > 0.7)


if __name__ == '__main__':
    if args.unittest: 
        pass
        runner = unittest.TextTestRunner()
        itersuite = unittest.TestLoader().loadTestsFromTestCase(ResNetTest)
        runner.run(itersuite)
    else: 
        benchmarking()