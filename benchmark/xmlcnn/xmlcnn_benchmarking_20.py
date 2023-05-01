################################################################################
# XML-CNN Benchmarking
################################################################################
# Dependencies
import sys
sys.path.append("/workspace/gtl/sparseTraining/Codegen/compiler")
sys.path.append("/workspace/bert")
from xmlcnn_modeling import xmlCNN, Params
# from apex import amp
# from lamb_amp_opt.fused_lamb import FusedLAMBAMP
# from amp_helper import scale_loss
# from aot_helper import compiler_fn, partition_func
# from functorch._src.compilers import ts_compile, tensorexpr_compile, tvm_compile
import pycutlass
from pycutlass import *
pycutlass.get_memory_pool(manager="torch")
pycutlass.compiler.nvcc()
# import nvtx
import argparse
import logging
# from pycutlass.test.profiler import GpuTimer
from functools import partial
import unittest
from gtl.helper import compiler_fn, partition_func, GTLProfiler, BaseTestCase, apex_autocast
from xmlcnn_pass_manager import pre_partition_optimization

import argparse
parser = argparse.ArgumentParser(description="XML-CNN End-to-End Training with CUDA Graph")
parser.add_argument('--iter', '-it', type=int, default=50, help="Profiling Iterations")
# method
parser.add_argument('--method', '-mt', type=str, default="torch", choices=["torch", "gtl", "nvfuser"])
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

    optimizer = torch.optim.SGD(optimizer_grouped_parameters, learning_rate)
    # optimizer = FusedLAMBAMP(optimizer_grouped_parameters,
    #                          lr=learning_rate)
    
    # optimizer.setup_fp32_params()

    return model, optimizer

def benchmarking():
    ## create model and optimizer
    model, optimizer = prepare_model_and_optimizer(params, "cuda")
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
    
    if args.cuda_graph:
        model.capture_graph(
            params.batch_size, params.sequence_length, params.embedding_dim, 
            params.y_dim, optimizer=optimizer) 
    
     ## profiling
    profiler = GTLProfiler(
        model, label=args.method, use_cuda_graph=args.cuda_graph)
    
    profiler.profile(
        [e_emb, y], ("seq", params.batch_size)
    )

class XMLCNNTest(BaseTestCase):
    def test_xmlcnn(self):
        sample_inputs = [e_emb, y]

        model, optimizer = prepare_model_and_optimizer(params, "cuda")
        model, optimizer = apex_autocast(model, optimizer, False)

        self.run_reference_model(model, optimizer, sample_inputs, 1)

        model_fused, optimizer_fused = prepare_model_and_optimizer(
            params, "cuda", reference=model)
        model_fused, optimizer_fused = apex_autocast(
            model_fused, optimizer_fused, False)
        
        model_fused.capture_graph(
            params.batch_size, params.sequence_length, params.embedding_dim, 
            params.y_dim, optimizer=optimizer_fused)

        self.run_target_model(model_fused, optimizer_fused, sample_inputs)

        self.verify(model, model_fused, verbose=0)
    
    def is_close(self, grad1, grad2):
        return torch.sum(
            torch.isclose(grad1, grad2, rtol=1e-1)) / grad1.numel() > 0.9


if __name__ == '__main__':
    if args.unittest: 
        runner = unittest.TextTestRunner()
        itersuite = unittest.TestLoader().loadTestsFromTestCase(XMLCNNTest)
        runner.run(itersuite)
    else: 
        benchmarking()