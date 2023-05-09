################################################################################
# GCN Benchmarking
################################################################################
# Dependencies
import sys
# sys.path.append("/workspace/sparseTraining/Codegen/compiler")
sys.path.append("/workspace/bert")
import torch
import dgl
import ogb
from ogb.nodeproppred import DglNodePropPredDataset
from gcn_modeling import GCN, DGLGCN_
import torch.nn.functional as F
# from lamb_amp_opt.fused_lamb import FusedLAMBAMP
# from aot_helper import compiler_fn, partition_func
# from functorch._src.compilers import ts_compile, tensorexpr_compile, tvm_compile
from gtl.helper import compiler_fn, partition_func, GTLProfiler, BaseTestCase, apex_autocast
from gat_pass_manager import pre_partition_optimization
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
parser.add_argument('--method', '-mt', type=str, default="torch", choices=["torch", "gtl", "nvfuser", "apex"])
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

row_idx, col_idx, e_idx = g.adj(etype='cites').csr()
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

    optimizer = torch.optim.SGD(optimizer_grouped_parameters, 1e-2)

    return model, optimizer


def benchmarking():
    ## create model and optimizer
    if args.method in ["torch", "nvfuser"]:
        model, optimizer = prepare_model_and_optimizer(embedding=64, depth=2, apex_loss=False, f32_loss=True)
    elif args.method in ["nvfuser", "apex"]:
        model, optimizer = prepare_model_and_optimizer(embedding=64, depth=2, apex_loss=True, f32_loss=False)
    elif args.method == "gtl":
        model, optimizer = prepare_model_and_optimizer(
            f32_loss=False, embedding=64, depth=2)
    
    logging.info("AMP: ON")
    model, optimizer = apex_autocast(model, optimizer, False)

    model.train()

    # perform AOT Optimization
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
    elif args.method == "nvfuser":
        model.aot_optimize(ts_compile, ts_compile)
    
    # timer = GpuTimer()
    if args.cuda_graph:
        model.capture_graph(csr, csc, features, labels, optimizer)
        model.set_features(features, labels)
        sample_inputs = []
    else:
        sample_inputs = [csr, csc, features, labels]
    
    ## profiling
    profiler = GTLProfiler(
        model, label=args.method, use_cuda_graph=args.cuda_graph
    )

    profiler.profile(
        sample_inputs,
        ("epoch", 1.)
    )


class GCNTest(BaseTestCase):
    def test_gcn(self):
        sample_inputs = [csr, csc, features, labels]

        model, optimizer = prepare_model_and_optimizer(embedding=64, depth=2, apex_loss=False)
        model, optimizer = apex_autocast(
            model, optimizer, True
        )

        self.run_reference_model(model, optimizer, sample_inputs, 1e+3)

        model_fused, optimizer_fused = prepare_model_and_optimizer(
            f32_loss=False, apex_loss=False, embedding=64, depth=2, reference=model)
        model_fused, optimizer_fused = apex_autocast(
            model_fused, optimizer_fused, False)
    
        model_fused.aot_optimize(
            compiler_fn, compiler_fn, 
            partial(
                partition_func, 
                joint_compiler=pre_partition_optimization
            )
        )
        model_fused.train()
        model_fused.capture_graph(csr, csc, features, labels, optimizer_fused)
        model_fused.set_features(features, labels)

        self.run_target_model(model_fused, optimizer_fused, [])

        self.verify(model, model_fused, verbose=0)
    
    def is_close(self, grad1, grad2):
        return (
            torch.sum(
                torch.isclose(grad1, grad2, rtol=1e-1)
            ) / grad1.numel() > 0.9 
        )


if __name__ == '__main__':
    if args.unittest: 
        pass
        runner = unittest.TextTestRunner()
        itersuite = unittest.TestLoader().loadTestsFromTestCase(GCNTest)
        runner.run(itersuite)
    else: 
        benchmarking()