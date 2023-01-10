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


i = 2
input_ids = torch.load("/workspace/bert/batch/input_ids_iter%d.pt"%i).unsqueeze(0).expand([4, 8, 512]).contiguous().view([-1, 512]).to("cuda")
token_type_ids = torch.load("/workspace/bert/batch/token_type_ids_iter%d.pt"%i).unsqueeze(0).expand([4, 8, 512]).contiguous().view([-1, 512]).to("cuda")
attention_mask = torch.load("/workspace/bert/batch/attention_mask_iter%d.pt"%i).to(torch.float16).unsqueeze(0).expand([4, 8, 512]).contiguous().view([-1, 512]).to("cuda")
next_sentence_labels = torch.load("/workspace/bert/batch/next_sentence_labels_iter%d.pt"%i).unsqueeze(0).expand([4, 8]).contiguous().view([-1]).to("cuda")
labels = torch.load("/workspace/bert/batch/labels_iter%d.pt"%i).unsqueeze(0).expand([4, 8, 512]).contiguous().view([-1, 512]).to("cuda")

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
model_fused, optimizer_fused = prepare_model_and_optimizer("cuda", sequence_output_is_dense=False, reference=model)

## amp initialize
model, optimizer = amp.initialize(
    model, optimizer, 
    cast_model_outputs=torch.float16, 
    opt_level="O2", keep_batchnorm_fp32=False, 
    loss_scale="dynamic"
)

model_fused, optimizer_fused = amp.initialize(
    model_fused, optimizer_fused, 
    cast_model_outputs=torch.float16, 
    opt_level="O2", keep_batchnorm_fp32=False, 
    loss_scale="dynamic"
)

# 
model.train()
model_fused.train()

model_fused.aot_optimize(compiler_fn, compiler_fn, partition_func)
# loss = model_fused(input_ids, token_type_ids, attention_mask, labels, labels, next_sentence_labels)
# with scale_loss(loss, optimizer_fused) as scaled_loss:
#     scaled_loss.backward()
model_fused.capture_graph(batch=32, sequence_length=512, optimizer=optimizer_fused)


## Functional verification
optimizer.zero_grad()
loss = model(input_ids, token_type_ids, attention_mask, labels, labels, next_sentence_labels)
with scale_loss(loss, optimizer) as scaled_loss:
    scaled_loss.backward()

optimizer_fused.zero_grad()
# loss = model_fused(input_ids, token_type_ids, attention_mask, labels, labels, next_sentence_labels)
# with scale_loss(loss, optimizer_fused) as scaled_loss:
#     scaled_loss.backward()
model_fused.training_with_graph(input_ids, token_type_ids, attention_mask, labels, labels, next_sentence_labels)

for param1, param2 in zip(list(model.named_parameters()), list(model_fused.named_parameters())):
    print(torch.sum(torch.isclose(param1[1].grad, param2[1].grad, rtol=1e-1)) / param2[1].grad.numel())
    try:
        assert torch.sum(torch.isclose(param1[1].grad, param2[1].grad, rtol=1e-1)) / param2[1].grad.numel() > 0.9
    except:
        print(param1[0])
        print(param1[1].grad)
        print(param2[1].grad)
        print(param1[1].grad.size())

# for i in range(40):
#     with nvtx.annotate("torch"):
#         loss = model(input_ids, token_type_ids, attention_mask, labels, labels, next_sentence_labels)
#         with scale_loss(loss, optimizer) as scaled_loss:
#             scaled_loss.backward()

# s = torch.cuda.Stream(priority=-1)
# s.wait_stream(torch.cuda.current_stream())
# with torch.cuda.stream(s):
#     for i in range(40):
#         with nvtx.annotate("fused"):
#             model_fused.training_with_graph(input_ids, token_type_ids, attention_mask, labels, labels, next_sentence_labels)