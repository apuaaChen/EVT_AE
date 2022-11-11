import sys
from turtle import forward
sys.path.append("/workspace/sparseTraining/Codegen/compiler")
sys.path.append("/workspace/bert")
import torch
import modeling
from schedulers import PolyWarmUpScheduler
from lamb_amp_opt.fused_lamb import FusedLAMBAMP
from typing import Final
import nvtx
from apex import amp
from apex.amp._amp_state import _amp_state
import contextlib

import pycutlass
from pycutlass import *
pycutlass.get_memory_pool(init_pool_size=2**10, max_pool_size=2**10)
pycutlass.compiler.nvcc()
import torch.fx as fx

from functorch.compile import aot_function, aot_module, compiled_function
from functorch import make_functional_with_buffers
from functorch._src.partitioners import _is_primal, _extract_fwd_bwd_outputs, _extract_graph_with_inputs_outputs, _extract_graph_with_inputs_outputs, _extract_fwd_bwd_modules
from passes import *
from bert_pass_manager import *


def partition_func(joint_module: fx.GraphModule, _joint_inputs):

    pre_partition_optimization(joint_module)

    primal_inputs = list(filter(_is_primal, joint_module.graph.nodes))
    fwd_outputs, bwd_outputs = _extract_fwd_bwd_outputs(joint_module)

    forward_only_graph = _extract_graph_with_inputs_outputs(joint_module.graph, primal_inputs, fwd_outputs)
    forward_node_names = set([node.name for node in forward_only_graph.nodes if node.op != 'output'])

    def node_saved(node):
        return node.name in forward_node_names and 'tensor_meta' in node.meta
    saved_values = [node for node in joint_module.graph.nodes if node_saved(node)]
    return _extract_fwd_bwd_modules(joint_module, saved_values)

# The compiler_fn is called after the forward and backward graphs are extracted.
# Here, we just print the code in the compiler_fn. Return of this function is a callable.
def compiler_fn(fx_module: torch.fx.GraphModule, _):
    # print(fx_module.code)
    return fx_module

# for i in range(10):
#     input_ids = torch.load("./batch/input_ids_iter%d.pt"%i)
#     token_type_ids = torch.load("./batch/token_type_ids_iter%d.pt"%i)
#     attention_mask = torch.load("./batch/attention_mask_iter%d.pt"%i)
#     next_sentence_labels = torch.load("./batch/next_sentence_labels_iter%d.pt"%i)
#     labels = torch.load("./batch/labels_iter%d.pt"%i)

#     print(input_ids.shape)
#     print(token_type_ids.shape)
#     print(attention_mask.shape)
#     print(next_sentence_labels.shape)
#     print(labels.shape)

# Step 1: get training data
i = 2
input_ids = torch.load("/workspace/bert/batch/input_ids_iter%d.pt"%i).unsqueeze(0).expand([4, 8, 512]).contiguous().view([-1, 512])
token_type_ids = torch.load("/workspace/bert/batch/token_type_ids_iter%d.pt"%i).unsqueeze(0).expand([4, 8, 512]).contiguous().view([-1, 512])
attention_mask = torch.load("/workspace/bert/batch/attention_mask_iter%d.pt"%i).to(torch.float16).unsqueeze(0).expand([4, 8, 512]).contiguous().view([-1, 512])
next_sentence_labels = torch.load("/workspace/bert/batch/next_sentence_labels_iter%d.pt"%i).unsqueeze(0).expand([4, 8]).contiguous().view([-1])
labels = torch.load("/workspace/bert/batch/labels_iter%d.pt"%i).unsqueeze(0).expand([4, 8, 512]).contiguous().view([-1, 512])

batch = {
    "input_ids": input_ids,
    "token_type_ids": token_type_ids,
    "attention_mask": attention_mask,
    "next_sentence_labels": next_sentence_labels,
    "labels": labels
}

class BertPretrainingCriterion(torch.nn.Module):

    sequence_output_is_dense: Final[bool]

    def __init__(self, vocab_size, sequence_output_is_dense=False):
        super(BertPretrainingCriterion, self).__init__()
        self.loss_fn = torch.nn.CrossEntropyLoss(ignore_index=-1)
        self.vocab_size = vocab_size
        self.sequence_output_is_dense = sequence_output_is_dense

    def forward(self, prediction_scores, seq_relationship_score, masked_lm_labels, next_sentence_labels):
        with nvtx.annotate("criterion"):
            if self.sequence_output_is_dense:
                # prediction_scores are already dense
                masked_lm_labels_flat = masked_lm_labels.view(-1)
                mlm_labels = masked_lm_labels_flat[masked_lm_labels_flat != -1]
                masked_lm_loss = self.loss_fn(prediction_scores.view(-1, self.vocab_size), mlm_labels.view(-1))
            else:
                masked_lm_loss = self.loss_fn(prediction_scores.view(-1, self.vocab_size), masked_lm_labels.view(-1))
            next_sentence_loss = self.loss_fn(seq_relationship_score.view(-1, 2), next_sentence_labels.view(-1))
            total_loss = masked_lm_loss + next_sentence_loss
        return total_loss

# args
config_file = "./large.json" #"/workspace/bert/bert_configs/large.json"
disable_jit_fusions = True
warmup_proportion=0.128
max_steps=40
learning_rate=6e-3
init_loss_scale=2048


class Bert(torch.nn.Module):
    def __init__(self, config, sequence_output_is_dense) -> None:
        super().__init__()
        self.model = modeling.BertForPreTraining(config, sequence_output_is_dense=sequence_output_is_dense)
        self.criterion = BertPretrainingCriterion(config.vocab_size, sequence_output_is_dense=sequence_output_is_dense)
    
    def forward(self, input_ids, token_type_ids, attention_mask, masked_lm_labels, labels, next_sentence_labels):
        prediction_scores, seq_relationship_score = self.model(input_ids, token_type_ids, attention_mask, masked_lm_labels)
        loss = self.criterion(prediction_scores, seq_relationship_score, labels, next_sentence_labels)
        return loss
    
    def checkpoint_activations(self, check):
        self.model.checkpoint_activations(check)


def prepare_model_and_optimizer(device, sequence_output_is_dense, reference=None):

    # Prepare model
    config = modeling.BertConfig.from_json_file(config_file)

    # Padding for divisibility by 8
    if config.vocab_size % 8 != 0:
        config.vocab_size += 8 - (config.vocab_size % 8)

    model = Bert(config, sequence_output_is_dense=sequence_output_is_dense)
    if reference is not None:
        reference_state_dict = reference.state_dict()
        model.load_state_dict(reference_state_dict)
        # print(list(model.named_parameters())[0])
        # print(list(reference.named_parameters())[0])

    model.to(device)

    if not disable_jit_fusions :
        model = torch.jit.script(model)

    param_optimizer = list(model.named_parameters())
    no_decay = ['bias', 'gamma', 'beta', 'LayerNorm']

    optimizer_grouped_parameters = [
        {'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)], 'weight_decay': 0.01},
        {'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}]

    optimizer = FusedLAMBAMP(optimizer_grouped_parameters,
                             lr=learning_rate)

    model.checkpoint_activations(False)

    optimizer.setup_fp32_params()

    # criterion = BertPretrainingCriterion(config.vocab_size, sequence_output_is_dense=sequence_output_is_dense)

    return model, optimizer

model, optimizer = prepare_model_and_optimizer("cuda", sequence_output_is_dense=True)
model_fused, optimizer_fused = prepare_model_and_optimizer("cuda", sequence_output_is_dense=True, reference=model)

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

model.train()
model_fused.train()
# print(model)
model_fused = aot_module(model_fused, fw_compiler=compiler_fn, bw_compiler=compiler_fn, partition_fn=partition_func)

@contextlib.contextmanager
def scale_loss(loss,
               optimizer,
               loss_id=0):

    loss_scaler = _amp_state.loss_scalers[loss_id]

    if not optimizer._amp_stash.params_have_scaled_gradients:
        optimizer._prepare_amp_backward()

    yield loss.float()

    loss_scaler.clear_overflow_state()

    optimizer._post_amp_backward(loss_scaler)
    optimizer._amp_stash.params_have_scaled_gradients = False

    # Probably ok to skip this if not delay_unscale
    if _amp_state.opt_properties.patch_torch_functions:
        _amp_state.handle._clear_cache()


batch = {k: v.to("cuda", non_blocking=True) for k, v in batch.items()}

def take_training_step(model, batch, optimizer):
    optimizer.zero_grad()
    with nvtx.annotate("forward"):
        loss = model(input_ids=batch['input_ids'], token_type_ids=batch['token_type_ids'], attention_mask=batch['attention_mask'], masked_lm_labels=batch['labels'], labels=batch['labels'], next_sentence_labels=batch['next_sentence_labels'])
    with nvtx.annotate("backward"):
        with scale_loss(loss, optimizer) as scaled_loss:
            scaled_loss.backward()
    with nvtx.annotate("update"):
        optimizer.step()

take_training_step(model_fused, batch, optimizer_fused)
take_training_step(model, batch, optimizer)


with nvtx.annotate("Compare"):
    for i in range(40):
        with nvtx.annotate("fused"):
            take_training_step(model_fused, batch, optimizer_fused)

    for i in range(40):
        with nvtx.annotate("unfused"):
            take_training_step(model, batch, optimizer)



# # #########################
# # # Functional Verification
# # #########################

# with nvtx.annotate("unfused"):
#     take_training_step(model, batch, optimizer)

# with nvtx.annotate("fused"):
#     take_training_step(model_fused, batch, optimizer_fused)


# for param1, param2 in zip(list(model.named_parameters()), list(model_fused.orig_module.named_parameters())):
#     print(torch.sum(torch.isclose(param1[1].grad, param2[1].grad, rtol=1e-1)) / param2[1].grad.numel())
#     try:
#         assert torch.sum(torch.isclose(param1[1].grad, param2[1].grad, rtol=1e-1)) / param2[1].grad.numel() > 0.95
#     except:
#         print(param1[1].grad)
#         print(param2[1].grad)