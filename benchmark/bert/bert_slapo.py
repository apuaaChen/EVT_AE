################################################################################
# Bert Benchmarking
################################################################################
# Dependencies
import sys
sys.path.append("/workspace/sparseTraining/Codegen/compiler")
sys.path.append("/workspace/bert")
import torch
import bert_modeling
import modeling
from apex import amp
from lamb_amp_opt.fused_lamb import FusedLAMBAMP
from bert_modeling_slapo import xFlashAttention, LSBertEmbeddings
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
import nvtx



################################################################################
# parse args
parser = argparse.ArgumentParser(description="Bert End-to-End Training with CUDA Graph")
parser.add_argument('--iter', '-it', type=int, default=50, help="Profiling Iterations")
# Hyper-parameter that defines the model size
parser.add_argument('--batch_size', '-b', type=int, default=32, help="Training batch size per GPU")
parser.add_argument('--seq_len', '-l', type=int, default=512, help="Sequence length")
# method
parser.add_argument('--method', '-mt', type=str, default="torch", choices=["torch", "east", "nvfuser", "tvm", "te"])
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
learning_rate=6e-3
config = modeling.BertConfig.from_json_file(config_file)

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

    optimizer = FusedLAMBAMP(optimizer_grouped_parameters,
                             lr=learning_rate)

    model.checkpoint_activations(False)

    optimizer.setup_fp32_params()

    return model, optimizer

model, optimizer = prepare_model_and_optimizer(
    "cuda", sequence_output_is_dense=False)

import slapo


sch = slapo.create_schedule(model.to(torch.float16), group=None)
# TODO: apply different passes
subsch = sch["encoder.model.bert.encoder.layer.0.attention"]


import torch.nn as nn



# pass 1: using flash attention
def pass_using_flash_attention(sch):
    for subsch in sch.child:
        if isinstance(sch[subsch].mod, modeling.BertSelfAttention):
            new_mod = xFlashAttention(config)
            sch[subsch].replace(new_mod)
        else:
            pass_using_flash_attention(sch[subsch])

# pass 2: using lightseq2 embedding
def pass_using_lightseq2_embedding(sch):
    for subsch in sch.child:
        if isinstance(sch[subsch].mod, modeling.BertEmbeddings):
            new_mod = LSBertEmbeddings(config)
            sch[subsch].replace(new_mod)
        else:
            pass_using_flash_attention(sch[subsch])

# pass 3: fuse bias gelu:
# can use nvfuser to process

pass_using_flash_attention(sch)
pass_using_lightseq2_embedding(sch)


# print(sch["encoder.model.bert.encoder.layer.0.attention.self"].mod)
# new_mod = SelfAttention(config)
# sch["encoder.model.bert.encoder.layer.0.attention.self"].replace(new_mod)


model, _ = slapo.build(sch)
model.to(torch.float16).to("cuda")

from pycutlass.test.profiler import GpuTimer

timer = GpuTimer()
s = torch.cuda.Stream(priority=-1)
s.wait_stream(torch.cuda.current_stream())
with torch.cuda.stream(s):
    # if args.cuda_graph:
    #     for i in range(10):
    #         model.training_with_graph(
    #             input_ids, token_type_ids, attention_mask, 
    #             labels, labels, next_sentence_labels)
    #     torch.cuda.synchronize()
    #     timer.start(stream=s.cuda_stream)
    #     with nvtx.annotate("40 iter"):
    #         for i in range(args.iter):
    #             with nvtx.annotate("1 iter"):
    #                 model.training_with_graph(
    #                     input_ids, token_type_ids, attention_mask, 
    #                     labels, labels, next_sentence_labels)
    #     timer.stop_and_wait(stream=s.cuda_stream)
    # else:
    for i in range(10):
        loss = model(
            input_ids, token_type_ids, attention_mask, 
            labels, labels, next_sentence_labels)
        loss.backward()
    # torch.cuda.synchronize()
    s.synchronize()
    timer.start(stream=s.cuda_stream)
    with nvtx.annotate("40 iter"):
        for i in range(args.iter):
            with nvtx.annotate("1 iter"):
                loss = model(
                    input_ids, token_type_ids, attention_mask, 
                    labels, labels, next_sentence_labels)
                loss.backward()
    timer.stop_and_wait(stream=s.cuda_stream)


iter_time = timer.duration(args.iter)
throughput = args.batch_size * args.seq_len / iter_time * 1000

print(f"Throughput: {throughput} token/sec, Iter time: {iter_time} ms")
