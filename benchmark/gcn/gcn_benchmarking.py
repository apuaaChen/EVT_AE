################################################################################
# GCN Benchmarking
################################################################################
# Dependencies
import sys
sys.path.append("/workspace/sparseTraining/Codegen/compiler")
sys.path.append("/workspace/bert")
import torch
from ogb.nodeproppred import DglNodePropPredDataset
import dgl
from gcn_modeling import GCN, DGLGCN_
import torch.nn.functional as F
from lamb_amp_opt.fused_lamb import FusedLAMBAMP
from aot_helper import compiler_fn, partition_func
from functorch._src.compilers import ts_compile, tensorexpr_compile, tvm_compile
from apex import amp
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
parser = argparse.ArgumentParser(description="GCN End-to-End Training with CUDA Graph")
parser.add_argument('--iter', '-it', type=int, default=50, help="Profiling Iterations")
# method
parser.add_argument('--method', '-mt', type=str, default="torch", choices=["torch", "east", "nvfuser", "apex"])
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
# logging.basicConfig(level=getattr(logging, args.log))
logging.basicConfig(level=logging.INFO)
logging.basicConfig(format='%(message)s')

logging.info(f"GCN OGBN-MAG")
logging.info(f"Compiler: {args.method}")
if args.cuda_graph:
    logging.info(f"CUDA Graph: ON")
else:
    logging.info(f"CUDA Graph: OFF")

################################################################################
# Initiate inputs
dataset = DglNodePropPredDataset(name = "ogbn-mag")

split_idx = dataset.get_idx_split()
train_idx, valid_idx, test_idx = split_idx["train"], split_idx["valid"], split_idx["test"]
g, labels = dataset[0]
g = g.to("cuda")

features = g.ndata["feat"]["paper"]
labels = labels["paper"]
labels = labels.to("cuda").squeeze()

d_i = torch.pow(g.out_degrees(etype='cites').float() + 1, -0.5)
d_j = torch.pow(g.in_degrees(etype='cites').float() + 1, -0.5)
e = dgl.ops.u_mul_v(g, d_i, d_j)[2].to(torch.float16)

row_idx, col_idx, e_idx = g.adj_sparse(fmt="csr", etype='cites')
csr = torch.sparse_csr_tensor(row_idx, col_idx, e)

csc_ = csr.transpose(0, 1)
csc = torch.sparse_csr_tensor(csc_.ccol_indices(), csc_.row_indices(), csc_.values())

in_feats = features.shape[1]
n_classes = ((dataset.num_classes + 7) // 8) * 8
features = features.to(torch.float16)

def prepare_model_and_optimizer(f32_loss=True, embedding=64, depth=2, reference=None, apex_loss=False):
    model = GCN(in_feats, embedding, n_classes, depth, F.relu, 1e-16, f32_loss, apex_loss)
    model.set_graph((csr, csc))
    if reference is not None:
        reference_embedding_state_dict = reference.state_dict()
        model.load_state_dict(reference_embedding_state_dict)
    model = model.to("cuda")

    param_optimizer = list(model.named_parameters())
    no_decay = ['bias', 'gamma', 'beta']

    optimizer_grouped_parameters = [
        {'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)], 'weight_decay': 0.0},
        {'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}]

    optimizer = FusedLAMBAMP(optimizer_grouped_parameters,
                             lr=1e-2)
    
    optimizer.setup_fp32_params()

    return model, optimizer


def benchmarking():
    ## create model and optimizer
    if args.method in ["torch", "nvfuser"]:
        model, optimizer = prepare_model_and_optimizer(embedding=64, depth=2, apex_loss=False, f32_loss=True)
    elif args.method in ["nvfuser", "apex"]:
        model, optimizer = prepare_model_and_optimizer(embedding=64, depth=2, apex_loss=True, f32_loss=False)
    elif args.method == "east":
        model, optimizer = prepare_model_and_optimizer(
            f32_loss=False, embedding=64, depth=2)
    
    model, optimizer = amp.initialize(
        model, optimizer,
        cast_model_outputs=torch.float16, 
        opt_level="O2", keep_batchnorm_fp32=True, 
        loss_scale="dynamic", verbosity=0
    )
    model.train()

    if args.method == "east":
        model.aot_optimize(compiler_fn, compiler_fn, partial(partition_func, enabled_passes=args.passes))
    elif args.method == "nvfuser":
        model.aot_optimize(ts_compile, ts_compile)
    
    timer = GpuTimer()
    if args.cuda_graph:
        model.capture_graph(features, labels, optimizer)
        model.set_features(features, labels)
    
    ## profiling
    s = torch.cuda.Stream(priority=-1)
    s.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(s):
        if args.cuda_graph:
            for i in range(10):
                model.train_with_graph()
            torch.cuda.synchronize()
            timer.start(stream=s.cuda_stream)
            with nvtx.annotate("40 iter"):
                for i in range(args.iter):
                    with nvtx.annotate("1 iter"):
                        model.train_with_graph()
            timer.stop_and_wait(stream=s.cuda_stream)
        else:
            for i in range(10):
                loss = model(features, labels) * 1e+3
                loss.backward()
            # torch.cuda.synchronize()
            s.synchronize()
            timer.start(stream=s.cuda_stream)
            with nvtx.annotate("40 iter"):
                for i in range(args.iter):
                    with nvtx.annotate("1 iter"):
                        loss = model(features, labels) * 1e+3
                        loss.backward()
            timer.stop_and_wait(stream=s.cuda_stream)
    
    iter_time = timer.duration(args.iter)
    throughput = 1. / iter_time * 1000
    print(f"Throughput: {throughput} epochs/sec, Iter time: {iter_time} ms")
    # print("hhhh")

class GCNTest(unittest.TestCase):
    def test_gcn(self):
        model, optimizer = prepare_model_and_optimizer(embedding=64, depth=2, apex_loss=True)
        model, optimizer = amp.initialize(
            model, optimizer,
            cast_model_outputs=torch.float16, 
            opt_level="O2", keep_batchnorm_fp32=True, 
            loss_scale="dynamic", verbosity=0
        )
        model.train()

        optimizer.zero_grad()
        loss = model(features, labels) * 1e+3
        loss.backward()

        model_fused, optimizer_fused = prepare_model_and_optimizer(
            f32_loss=False, apex_loss=False, embedding=64, depth=2, reference=model)
        model_fused, optimizer_fused = amp.initialize(
            model_fused, optimizer_fused,
            cast_model_outputs=torch.float16, 
            opt_level="O2", keep_batchnorm_fp32=True, 
            loss_scale="dynamic", verbosity=0
        )
        model_fused.train()
        model_fused.aot_optimize(compiler_fn, compiler_fn, partition_func)
        model_fused.capture_graph(features, labels, optimizer_fused)
        optimizer_fused.zero_grad()
        model_fused.set_features(features, labels)
        model_fused.train_with_graph()
        for param1, param2 in zip(list(model.named_parameters()), list(model_fused.named_parameters())):
            grad_origin = param1[1].grad
            grad_fused = param2[1].grad
            # print(torch.sum(torch.isclose(grad_origin, grad_fused, rtol=1e-1)) / grad_origin.numel())
            self.assertTrue(
                torch.sum(torch.isclose(grad_origin, grad_fused, rtol=1e-1)) / grad_origin.numel() > 0.9
            )


if __name__ == '__main__':
    if args.unittest: 
        pass
        runner = unittest.TextTestRunner()
        itersuite = unittest.TestLoader().loadTestsFromTestCase(GCNTest)
        runner.run(itersuite)
    else: 
        benchmarking()