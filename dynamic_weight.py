import contextlib
import nvtx
import torch.nn as nn
from bert_config import BertConfig, allclose
import torch
from apex.optimizers import FusedAdam
from apex.contrib.sparsity import ASP
from apex import amp

from sptrain.meta import bdense2sparse_gold
from sparse_ops_weight import Linear, SpLinear
from apex.amp._amp_state import _amp_state


config_file = "./BERT/BERT/bert_configs/large.json"
config = BertConfig.from_json_file(config_file)



class BertSelfOutput(nn.Module):
    def __init__(self, config):
        super(BertSelfOutput, self).__init__()
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        self.LayerNorm = nn.LayerNorm(config.hidden_size, eps=1e-12)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

    def forward(self, hidden_states, input_tensor):
        hidden_states = self.dense(hidden_states)
        # hidden_states = self.dropout(hidden_states)
        hidden_states = self.LayerNorm(hidden_states + input_tensor)
        return hidden_states


class SparseBertSelfOutput(nn.Module):
    def __init__(self, config):
        super(SparseBertSelfOutput, self).__init__()
        self.dense = SpLinear(config.hidden_size, config.hidden_size)
        self.LayerNorm = nn.LayerNorm(config.hidden_size, eps=1e-12)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

    def forward(self, hidden_states, input_tensor):
        hidden_states = self.dense(hidden_states)
        # hidden_states = self.dropout(hidden_states)
        hidden_states = self.LayerNorm(hidden_states + input_tensor)
        return hidden_states

model = BertSelfOutput(config).to("cuda")
model_sparse = SparseBertSelfOutput(config).to("cuda")

# Prepare optimizer
param_optimizer = list(model.named_parameters())
param_optimizer = [n for n in param_optimizer if 'pooler' not in n[0]]
no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
optimizer_grouped_parameters = [
        {'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)], 'weight_decay': 0.01},
        {'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
    ]

param_optimizer_sparse = list(model_sparse.named_parameters())
param_optimizer_sparse = [n for n in param_optimizer_sparse if 'pooler' not in n[0]]
optimizer_grouped_parameters_sparse = [
        {'params': [p for n, p in param_optimizer_sparse if not any(nd in n for nd in no_decay)], 'weight_decay': 0.01},
        {'params': [p for n, p in param_optimizer_sparse if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
    ]

optimizer = FusedAdam(optimizer_grouped_parameters, lr=0.1, bias_correction=False)
optimizer_sparse = FusedAdam(optimizer_grouped_parameters_sparse, lr=0.1, bias_correction=False)

# Something to do with master weights
model, optimizer = amp.initialize(model, optimizer, opt_level="O2", keep_batchnorm_fp32=False, loss_scale="dynamic")
model_sparse, optimizer_sparse = amp.initialize(model_sparse, optimizer_sparse, opt_level="O2", keep_batchnorm_fp32=False, loss_scale="dynamic")

model_sparse.dense.set_parameters(model.dense.weight, model.dense.bias)

## Create the inputs
batch_size = 4
sequence_length = 2048
hidden = config.hidden_size

hidden_states = torch.randn(size=(sequence_length, batch_size, hidden), dtype=torch.float16, device="cuda", requires_grad=True)
input_tensor = torch.randn(size=(sequence_length, batch_size, hidden), dtype=torch.float16, device="cuda", requires_grad=True)

hidden_states_sparse = hidden_states.detach().clone().requires_grad_(True)
input_tensor_sparse = input_tensor.detach().clone().requires_grad_(True)

grad_output = torch.randn_like(hidden_states)

# try NVIDIA ASP
ASP.init_model_for_pruning(model, "m4n2_1d", whitelist=[torch.nn.Linear], allow_recompute_mask=True)
ASP.init_optimizer_for_pruning(optimizer)


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


############################################################
# Functional verification

ASP.compute_sparse_masks()

optimizer.zero_grad()

# forward pass
output = model(hidden_states, input_tensor)
output_sparse = model_sparse(hidden_states_sparse, input_tensor_sparse)

with scale_loss(output, optimizer) as scaled_loss:
    scaled_loss.backward(grad_output)

with scale_loss(output_sparse, optimizer_sparse) as scaled_loss:
    scaled_loss.backward(grad_output)

optimizer.step()

# output_sparse.backward()
# print(output)

allclose(output_sparse, output, 5e-2, 5e-2)
allclose(model_sparse.dense.get_dense_weight_grad(), model.dense.weight.grad, 5e-2, 5e-2)
allclose(model_sparse.dense.bias.grad, model.dense.bias.grad, 5e-2, 5e-2)
allclose(hidden_states_sparse.grad, hidden_states.grad, 5e-2, 1e-1)
allclose(input_tensor_sparse.grad, input_tensor.grad, 5e-2, 5e-2)


# profling
for i in range(10):
    with nvtx.annotate("asp"):
        with nvtx.annotate("forward"):
            output = model(hidden_states, input_tensor)
        with nvtx.annotate("backward"):
            with scale_loss(output, optimizer) as scaled_loss:
                scaled_loss.backward(grad_output)
        with nvtx.annotate("update"):
            optimizer.step()


for i in range(10):
    with nvtx.annotate("dsp"):
        with nvtx.annotate("forward"):
            output_sparse = model_sparse(hidden_states_sparse, input_tensor_sparse)
        with nvtx.annotate("backward"):
            with scale_loss(output_sparse, optimizer_sparse) as scaled_loss:
                scaled_loss.backward(grad_output)
        with nvtx.annotate("update"):
            optimizer_sparse.step()




# print(output_2)


# def train_step():
#     optimizer.zero_grad()
#     with nvtx.annotate("forward"):
#         output = model(hidden_states, input_tensor)
#     with nvtx.annotate("backward"):
#         output.backward(grad_output)
#     with nvtx.annotate("step"):
#         optimizer.step()

# # # train with dense model
# # for i in range(10):
# #     with nvtx.annotate("dense"):
# #         train_step()

# # simulate sparsity by inserting zeros into existing dense weights
# ASP.compute_sparse_masks()
# print(model.dense.weight)

# # # train with sparse mask
# # for i in range(10):
# #     with nvtx.annotate("sparse"):
# #         train_step()

# print(model_sparse.dense.weigh)
