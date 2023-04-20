################################################################################
# Bert Benchmarking
################################################################################
# Dependencies
import sys
sys.path.append("/workspace/gtl/sparseTraining/Codegen/compiler")
sys.path.append("/workspace/bert")
import torch
import bert_modeling
import modeling
from apex import amp
import functorch
# from lamb_amp_opt.fused_lamb import FusedLAMBAMP
from amp_helper import scale_loss
# from aot_helper import compiler_fn, partition_func
from gtl.helper import compiler_fn, partition_func, GTLProfiler, BaseTestCase, apex_autocast
from bert_pass_manager import pre_partition_optimization
from bert_pass_manager import *
from functorch.compile import ts_compile
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


from torch.utils._python_dispatch import _pop_mode_temporarily, _disable_current_modes

################################################################################
# parse args
parser = argparse.ArgumentParser(description="Bert End-to-End Training with CUDA Graph")
parser.add_argument('--iter', '-it', type=int, default=50, help="Profiling Iterations")
# Hyper-parameter that defines the model size
parser.add_argument('--batch_size', '-b', type=int, default=32, help="Training batch size per GPU")
parser.add_argument('--seq_len', '-l', type=int, default=512, help="Sequence length")
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

logging.info(f"BertLarge[batch size: {args.batch_size}, seq len: {args.seq_len}]")
logging.info(f"Compiler: {args.method}")
if args.cuda_graph:
    logging.info(f"CUDA Graph: ON")
else:
    logging.info(f"CUDA Graph: OFF")

################################################################################
# Initiate inputs
input_ids = torch.randint(low=101, high=29858, size=(args.batch_size, args.seq_len), dtype=torch.int64, device="cuda")
token_type_ids = torch.randint(low=0, high=2, size=(args.batch_size, args.seq_len), dtype=torch.int64, device="cuda")
attention_mask = torch.ones(size=(args.batch_size, args.seq_len), dtype=torch.float16, device="cuda")
next_sentence_labels = torch.randint(low=0, high=2, size=(args.batch_size,), dtype=torch.int64, device="cuda")
labels = torch.randint(low=-1, high=26555, size=(args.batch_size, args.seq_len), dtype=torch.int64, device="cuda")

config_file = "./large.json"
config = modeling.BertConfig.from_json_file(config_file)
learning_rate=6e-3

def prepare_model_and_optimizer(device, sequence_output_is_dense, reference=None):

    # Padding for divisibility by 8
    if config.vocab_size % 8 != 0:
        config.vocab_size += 8 - (config.vocab_size % 8)

    model = bert_modeling.Bert(config, sequence_output_is_dense)

    if reference is not None:
        reference_embedding_state_dict = reference.state_dict()
        model.load_state_dict(reference_embedding_state_dict)

    model.to(device)

    param_optimizer = list(model.named_parameters())
    no_decay = ['bias', 'gamma', 'beta', 'LayerNorm']

    optimizer_grouped_parameters = [
        {'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)], 'weight_decay': 0.01},
        {'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}]

    optimizer = torch.optim.SGD(optimizer_grouped_parameters, learning_rate)

    model.checkpoint_activations(False)

    return model, optimizer


def benchmarking():
    ## create model and optimizer
    model, optimizer = prepare_model_and_optimizer("cuda", sequence_output_is_dense=False)
    
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
            batch=args.batch_size, sequence_length=args.seq_len, optimizer=optimizer) 
    
    ## profiling
    profiler = GTLProfiler(
        model, label=args.method, use_cuda_graph=args.cuda_graph)
    
    profiler.profile(
        [
            input_ids, token_type_ids, attention_mask, 
            labels, labels, next_sentence_labels
        ], 
        ("token", args.batch_size * args.seq_len)
    )


class BertTest(BaseTestCase):
    def test_bert_large(self):
        sample_inputs = [input_ids, token_type_ids, attention_mask, 
            labels, labels, next_sentence_labels]
        model, optimizer = prepare_model_and_optimizer(
            "cuda", sequence_output_is_dense=False)
        model, optimizer = amp.initialize(
            model, optimizer, 
            cast_model_outputs=torch.float16, 
            opt_level="O2", keep_batchnorm_fp32=False, 
            loss_scale="dynamic", verbosity=0
        )
        self.run_reference_model(model, optimizer, sample_inputs, 4096.)
        
        model_fused, optimizer_fused = prepare_model_and_optimizer(
            "cuda", sequence_output_is_dense=False, reference=model)
        model_fused, optimizer_fused = amp.initialize(
            model_fused, optimizer_fused, 
            cast_model_outputs=torch.float16, 
            opt_level="O2", keep_batchnorm_fp32=False, 
            loss_scale="dynamic", verbosity=0
        )
        model_fused.capture_graph(
            batch=args.batch_size, 
            sequence_length=args.seq_len, optimizer=optimizer_fused)
        self.run_target_model(model_fused, optimizer_fused, sample_inputs)

        self.verify(model, model_fused, verbose=0)

    def is_close(self, grad1, grad2):
        return (
            torch.sum(
                torch.isclose(grad1, grad2, rtol=1e-1)
            ) / grad1.numel() > 0.9 
            or torch.allclose(grad1, grad2, atol=1e-3)
        )

if __name__ == '__main__':
    if args.unittest: 
        runner = unittest.TextTestRunner()
        itersuite = unittest.TestLoader().loadTestsFromTestCase(BertTest)
        runner.run(itersuite)
    else: 
        benchmarking()