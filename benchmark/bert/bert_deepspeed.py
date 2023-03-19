import sys
sys.path.append("/workspace/sparseTraining/Codegen/compiler")
sys.path.append("/workspace/bert")
import torch
import bert_modeling
import modeling
from apex import amp
from lamb_amp_opt.fused_lamb import FusedLAMBAMP
from amp_helper import scale_loss
from aot_helper import compiler_fn, partition_func
import pycutlass
from pycutlass import *
pycutlass.get_memory_pool(manager="torch")
pycutlass.compiler.nvcc()
import nvtx
import deepspeed

# DeepSpeed requires a distributed environment even when only one process is used
# https://huggingface.co/transformers/v4.10.1/main_classes/deepspeed.html
# This emulates a launcher
import os
os.environ['MASTER_ADDR'] = 'localhost'
os.environ['MASTER_PORT'] = '9994' # modify if RuntimeError: Address already in use
os.environ['RANK'] = "0"
os.environ['LOCAL_RANK'] = "0"
os.environ['WORLD_SIZE'] = "1"


import argparse
parser = argparse.ArgumentParser(description="Bert End-to-End Training with CUDA Graph")
# Hyper-parameter that defines the model size
parser.add_argument('--batch_size', '-b', type=int, default=32, help="Training batch size per GPU")
parser.add_argument('--seq_len', '-l', type=int, default=512, help="Sequence length")
# Modes
parser.add_argument('--mode', '-m', type=str, default="verify", choices=["verify", "profile"])
parser = deepspeed.add_config_arguments(parser)
args = parser.parse_args()

# creating inputs

input_ids = torch.randint(low=101, high=29858, size=(args.batch_size, args.seq_len), dtype=torch.int64, device="cuda")
token_type_ids = torch.randint(low=0, high=2, size=(args.batch_size, args.seq_len), dtype=torch.int64, device="cuda")
attention_mask = torch.ones(size=(args.batch_size, args.seq_len), dtype=torch.float16, device="cuda")
next_sentence_labels = torch.randint(low=0, high=2, size=(args.batch_size,), dtype=torch.int64, device="cuda")
labels = torch.randint(low=-1, high=26555, size=(args.batch_size, args.seq_len), dtype=torch.int64, device="cuda")

# Also supports loading real data loaded from dataset
# i = 2
# input_ids = torch.load("/workspace/bert/batch/input_ids_iter%d.pt"%i).unsqueeze(0).expand([4, 8, 512]).contiguous().view([-1, 512]).to("cuda")
# token_type_ids = torch.load("/workspace/bert/batch/token_type_ids_iter%d.pt"%i).unsqueeze(0).expand([4, 8, 512]).contiguous().view([-1, 512]).to("cuda")
# attention_mask = torch.load("/workspace/bert/batch/attention_mask_iter%d.pt"%i).to(torch.float16).unsqueeze(0).expand([4, 8, 512]).contiguous().view([-1, 512]).to("cuda")
# next_sentence_labels = torch.load("/workspace/bert/batch/next_sentence_labels_iter%d.pt"%i).unsqueeze(0).expand([4, 8]).contiguous().view([-1]).to("cuda")
# labels = torch.load("/workspace/bert/batch/labels_iter%d.pt"%i).unsqueeze(0).expand([4, 8, 512]).contiguous().view([-1, 512]).to("cuda")

batch = {
    "input_ids": input_ids,
    "token_type_ids": token_type_ids,
    "attention_mask": attention_mask,
    "next_sentence_labels": next_sentence_labels,
    "labels": labels
}

## args
config_file = "./large.json" 
learning_rate=6e-3


def prepare_model_and_optimizer(device, sequence_output_is_dense, reference=None):

    # Prepare model
    config = modeling.BertConfig.from_json_file(config_file)

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

## create model and optimizer
model, optimizer = prepare_model_and_optimizer("cuda", sequence_output_is_dense=False)


## amp initialize
model, optimizer = amp.initialize(
    model, optimizer, 
    cast_model_outputs=torch.float16, 
    opt_level="O2", keep_batchnorm_fp32=False, 
    loss_scale="dynamic"
)

# 
model.train()


optimizer.zero_grad()
loss = model(input_ids, token_type_ids, attention_mask, labels, labels, next_sentence_labels)
with scale_loss(loss, optimizer) as scaled_loss:
    scaled_loss.backward()

if args.mode == "profile":
    for i in range(10):
        loss = model(input_ids, token_type_ids, attention_mask, labels, labels, next_sentence_labels)
        with scale_loss(loss, optimizer) as scaled_loss:
            scaled_loss.backward()

    with nvtx.annotate("pytorch 40 iter"):
        for i in range(40):
            with nvtx.annotate("torch"):
                loss = model(input_ids, token_type_ids, attention_mask, labels, labels, next_sentence_labels)
                with scale_loss(loss, optimizer) as scaled_loss:
                    scaled_loss.backward()

    torch.cuda.synchronize()

model_fused, optimizer_fused = prepare_model_and_optimizer("cuda", sequence_output_is_dense=False, reference=model)

if args.mode == "profile":
    del model
    del optimizer

model_fused, optimizer_fused = amp.initialize(
    model_fused, optimizer_fused, 
    cast_model_outputs=torch.float16, 
    opt_level="O2", keep_batchnorm_fp32=False, 
    loss_scale="dynamic"
)

model_fused, optimizer_fused, _, _ = deepspeed.initialize(
    args=args, model=model_fused, optimizer=optimizer_fused
)

loss = model_fused(input_ids, token_type_ids, attention_mask, labels, labels, next_sentence_labels)
with scale_loss(loss, optimizer_fused) as scaled_loss:
    model_fused.backward(scaled_loss)


# model_fused.train()
# model_fused.aot_optimize(compiler_fn, compiler_fn, partition_func)
# model_fused.capture_graph(batch=args.batch_size, sequence_length=args.seq_len, optimizer=optimizer_fused)


# optimizer_fused.zero_grad()
# s = torch.cuda.Stream(priority=-1)
# s.wait_stream(torch.cuda.current_stream())
# with torch.cuda.stream(s):
#     model_fused.training_with_graph(input_ids, token_type_ids, attention_mask, labels, labels, next_sentence_labels)

if args.mode == "profile":
    s = torch.cuda.Stream(priority=-1)
    s.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(s):
        for i in range(10):
            loss = model_fused(input_ids, token_type_ids, attention_mask, labels, labels, next_sentence_labels)
            with scale_loss(loss, optimizer_fused) as scaled_loss:
                model_fused.backward(scaled_loss)
            # model_fused.training_with_graph(input_ids, token_type_ids, attention_mask, labels, labels, next_sentence_labels)
    
        with nvtx.annotate("ds 40 iter"):
            for i in range(40):
                with nvtx.annotate("ds"):
                    loss = model_fused(input_ids, token_type_ids, attention_mask, labels, labels, next_sentence_labels)
                    with scale_loss(loss, optimizer_fused) as scaled_loss:
                        model_fused.backward(scaled_loss)
                    # model_fused.training_with_graph(input_ids, token_type_ids, attention_mask, labels, labels, next_sentence_labels)

# if args.mode == "verify":
#     for param1, param2 in zip(list(model.named_parameters()), list(model_fused.named_parameters())):
#         print(torch.sum(torch.isclose(param1[1].grad, param2[1].grad, rtol=1e-1)) / param2[1].grad.numel())
#         try:
#             assert torch.sum(torch.isclose(param1[1].grad, param2[1].grad, rtol=1e-1)) / param2[1].grad.numel() > 0.9
#         except:
#             print(param1[0])
#             print(param1[1].grad)
#             print(param2[1].grad)
#             print(param1[1].grad.size())