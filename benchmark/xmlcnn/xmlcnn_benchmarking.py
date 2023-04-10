################################################################################
# XML-CNN Benchmarking
################################################################################
# Dependencies
import sys
sys.path.append("/workspace/sparseTraining/Codegen/compiler")
sys.path.append("/workspace/bert")
from xmlcnn_modeling import xmlCNN, Params
from apex import amp
from lamb_amp_opt.fused_lamb import FusedLAMBAMP
from amp_helper import scale_loss
from aot_helper import compiler_fn, partition_func
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

import argparse
parser = argparse.ArgumentParser(description="XML-CNN End-to-End Training with CUDA Graph")
parser.add_argument('--iter', '-it', type=int, default=50, help="Profiling Iterations")
# method
parser.add_argument('--method', '-mt', type=str, default="torch", choices=["torch", "east", "nvfuser"])
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

logging.info(f"XML-CNN")
logging.info(f"Compiler: {args.method}")
if args.cuda_graph:
    logging.info(f"CUDA Graph: ON")
else:
    logging.info(f"CUDA Graph: OFF")

################################################################################
# Model Configuration
params = Params(
    embedding_dim=304,  # alignment of 8
    filter_sizes=[2, 4, 8],
    sequence_length=512,
    batch_size=2048,
    num_filters=32,
    y_dim=670208,
    hidden_dims=512,
    pooling_chunks=8
)

################################################################################
# Initiate inputs
e_emb = torch.randn(
    size=(params.batch_size, params.sequence_length, params.embedding_dim), 
    dtype=torch.float16, device="cuda")

y = torch.empty(
    size=(params.batch_size, params.y_dim), 
    dtype=torch.float16, device="cuda").random_(2)

learning_rate=6e-3

def prepare_model_and_optimizer(params, device, reference=None):
    model = xmlCNN(params)
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
    ## create model and optimizer
    model, optimizer = prepare_model_and_optimizer(params, "cuda")
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
        model.aot_optimize(ts_compile, ts_compile)
    
    timer = GpuTimer()

    if args.cuda_graph:
        model.capture_graph(
            params.batch_size, params.sequence_length, params.embedding_dim, 
            params.y_dim, optimizer=optimizer) 
        
    ## profiling
    s = torch.cuda.Stream(priority=-1)
    s.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(s):
        if args.cuda_graph:
            for i in range(10):
                model.training_with_graph(e_emb, y)
            torch.cuda.synchronize()
            timer.start(stream=s.cuda_stream)
            with nvtx.annotate("40 iter"):
                for i in range(args.iter):
                    with nvtx.annotate("1 iter"):
                        model.training_with_graph(e_emb, y)
            timer.stop_and_wait(stream=s.cuda_stream)
        else:
            for i in range(10):
                loss = model(e_emb, y)
                loss.backward()
            # torch.cuda.synchronize()
            s.synchronize()
            timer.start(stream=s.cuda_stream)
            with nvtx.annotate("40 iter"):
                for i in range(args.iter):
                    with nvtx.annotate("1 iter"):
                        loss = model(e_emb, y)
                        loss.backward()
            timer.stop_and_wait(stream=s.cuda_stream)


    iter_time = timer.duration(args.iter)
    throughput = params.batch_size / iter_time * 1000
    logging.info(f"Throughput: {throughput} seqs/sec, Iter time: {iter_time} ms")

class XmlCnnTest(unittest.TestCase):
    def test_xml_cnn(self):
        model, optimizer = prepare_model_and_optimizer(params, "cuda")
        model, optimizer = amp.initialize(
            model, optimizer, 
            cast_model_outputs=torch.float16, 
            opt_level="O2", keep_batchnorm_fp32=False, 
            loss_scale="dynamic", verbosity=0
        )
        model.train()
        optimizer.zero_grad()
        loss = model(e_emb, y)
        loss.backward()

        model_fused, optimizer_fused = prepare_model_and_optimizer(params, "cuda", reference=model)
        model_fused, optimizer_fused = amp.initialize(
            model_fused, optimizer_fused, 
            cast_model_outputs=torch.float16, 
            opt_level="O2", keep_batchnorm_fp32=False, 
            loss_scale="dynamic", verbosity=0
        )

        model_fused.train()
        model_fused.aot_optimize(compiler_fn, compiler_fn, partition_func)
        model_fused.capture_graph(
            params.batch_size, params.sequence_length, params.embedding_dim, 
            params.y_dim, optimizer=optimizer_fused)
        optimizer_fused.zero_grad()
        s = torch.cuda.Stream(priority=-1)
        s.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(s):
            model_fused.training_with_graph(e_emb, y)

        torch.cuda.synchronize()
        
        for param1, param2 in zip(list(model.named_parameters()), list(model_fused.named_parameters())):
            # print(torch.sum(torch.isclose(param1[1].grad, param2[1].grad, rtol=1e-1)) / param2[1].grad.numel())
            self.assertTrue(torch.sum(torch.isclose(param1[1].grad, param2[1].grad, rtol=1e-1)) / param2[1].grad.numel() > 0.9)

if __name__ == '__main__':
    if args.unittest: 
        pass
        runner = unittest.TextTestRunner()
        itersuite = unittest.TestLoader().loadTestsFromTestCase(XmlCnnTest)
        runner.run(itersuite)
    else: 
        benchmarking()