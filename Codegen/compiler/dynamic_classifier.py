from importlib.machinery import PathFinder
from importlib.util import module_from_spec
import torch.nn as nn
import torch
from apex.optimizers import FusedAdam
from apex import amp
import nvtx
from apex.amp._amp_state import _amp_state
import contextlib
import torch.nn.functional as F
import torch.fx as fx


from functorch.compile import aot_function, aot_module, compiled_function
from functorch import make_functional_with_buffers
from functorch._src.aot_autograd import _is_primal, _extract_fwd_bwd_outputs, _extract_graph_with_inputs_outputs, _extract_graph_with_inputs_outputs, _extract_fwd_bwd_modules
from passes import *


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
    print(fx_module.code)
    return fx_module


class Classifier(nn.Module):
    """
    Fully-connected classifier
    """
    def __init__(self, in_features, out_features, init_weight=0.1):
        """
        Constructor for the Classifier.

        :param in_features: number of input features
        :param out_features: number of output features (size of vocabulary)
        :param init_weight: range for the uniform initializer
        """
        super(Classifier, self).__init__()
        self.classifier = nn.Linear(in_features, out_features)
        nn.init.uniform_(self.classifier.weight.data, -init_weight, init_weight)
        nn.init.uniform_(self.classifier.bias.data, -init_weight, init_weight)

    def forward(self, x):
        """
        Execute the classifier.

        :param x: output from decoder
        """
        out = self.classifier(x)
        return out


# Original LabelSmoothing. Gather has wrong auto gradient in the current version
# class LabelSmoothing(nn.Module):
#     """
#     NLL loss with label smoothing.
#     """
#     def __init__(self, padding_idx, smoothing=0.0):
#         """
#         Constructor for the LabelSmoothing module.

#         :param padding_idx: index of the PAD token
#         :param smoothing: label smoothing factor
#         """
#         super(LabelSmoothing, self).__init__()
#         self.padding_idx = padding_idx
#         self.confidence = 1.0 - smoothing
#         self.smoothing = smoothing

#     def forward(self, x, target):
#         logprobs = torch.nn.functional.log_softmax(x, dim=-1,
#                                                    dtype=torch.float32)

#         non_pad_mask = (target != self.padding_idx)
#         nll_loss = -logprobs.gather(dim=-1, index=target.unsqueeze(1))
#         nll_loss = nll_loss.squeeze(1)[non_pad_mask]
#         smooth_loss = -logprobs.mean(dim=-1)[non_pad_mask]
#         loss = self.confidence * nll_loss + self.smoothing * smooth_loss
#         return loss.sum()


class LabelSmoothing(nn.Module):
    """
    NLL loss with label smoothing.
    """
    def __init__(self, padding_idx, smoothing=0.0):
        """
        Constructor for the LabelSmoothing module.

        :param padding_idx: index of the PAD token
        :param smoothing: label smoothing factor
        """
        super(LabelSmoothing, self).__init__()
        self.padding_idx = padding_idx
        self.confidence = 1.0 - smoothing
        self.smoothing = smoothing

    def forward(self, x, target):
        logprobs = torch.nn.functional.log_softmax(x, dim=-1)

        non_pad_mask = (target != self.padding_idx)
        # nll_loss = -logprobs.gather(dim=-1, index=target.unsqueeze(1))
        nll_loss = F.nll_loss(logprobs, target, ignore_index=self.padding_idx, reduction="sum")
        smooth_loss = -logprobs.mean(dim=-1)*non_pad_mask
        loss = self.confidence * nll_loss + self.smoothing * smooth_loss.sum()
        return loss


class ExtremeClassifier(nn.Module):
    def __init__(self, in_features, out_features, padding_idx, smoothing) -> None:
        super(ExtremeClassifier, self).__init__()
        self.classifier = Classifier(in_features, out_features)
        self.criterion = LabelSmoothing(padding_idx, smoothing)
    
    def forward(self, input, target):
        output = self.classifier(input)
        T, B = output.size(0), output.size(1)
        loss = self.criterion(output.view(T * B, -1), target)
        return loss


# configurations
hidden_size = 1024
vocab_size = 32320

# load operand
input = torch.load('/data/datasets/users/zdchen/input.pt').to("cuda").to(torch.float).requires_grad_(True)
input.retain_grad()
target = torch.load('/data/datasets/users/zdchen/target.pt').to("cuda").requires_grad_(False)
input_sparse = input.clone().detach().requires_grad_(True)

model = ExtremeClassifier(hidden_size, vocab_size, 0, 0.1).to("cuda")
model_sparse = ExtremeClassifier(hidden_size, vocab_size, 0, 0.1).to("cuda")

model_sparse.classifier.classifier.weight = torch.nn.Parameter(model.classifier.classifier.weight.clone())
model_sparse.classifier.classifier.bias = torch.nn.Parameter(model.classifier.classifier.bias.clone())

model_sparse = aot_module(model_sparse, fw_compiler=compiler_fn, bw_compiler=compiler_fn, partition_fn=partition_func)

param_optimizer = list(model.named_parameters())
param_optimizer = [n for n in param_optimizer if 'pooler' not in n[0]]
no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
optimizer_grouped_parameters = [
        {'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)], 'weight_decay': 0.01},
        {'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
    ]

param_optimizer_sparse = list(model_sparse.named_parameters())
param_optimizer_sparse = [n for n in param_optimizer_sparse if 'pooler' not in n[0]]
no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
optimizer_grouped_parameters_sparse = [
        {'params': [p for n, p in param_optimizer_sparse if not any(nd in n for nd in no_decay)], 'weight_decay': 0.01},
        {'params': [p for n, p in param_optimizer_sparse if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
    ]

optimizer = FusedAdam(optimizer_grouped_parameters, lr=0.1, bias_correction=False)
optimizer_sparse = FusedAdam(optimizer_grouped_parameters_sparse, lr=0.1, bias_correction=False)

model, optimizer = amp.initialize(
    model, optimizer, 
    cast_model_outputs=torch.float16, 
    opt_level="O2", keep_batchnorm_fp32=False, 
    loss_scale="dynamic")

model_sparse, optimizer_sparse = amp.initialize(
    model_sparse, optimizer_sparse, 
    cast_model_outputs=torch.float16, 
    opt_level="O2", keep_batchnorm_fp32=False, 
    loss_scale="dynamic")

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

#########################
# Functional Verification
#########################

optimizer.zero_grad()
optimizer_sparse.zero_grad()

with nvtx.annotate("forward"):
    loss_ref = model(input, target)
with nvtx.annotate("backward"):
    with scale_loss(loss_ref, optimizer) as scaled_loss:
        scaled_loss.backward()
with nvtx.annotate("update"):
    optimizer.step()

with nvtx.annotate("sp forward"):
    loss_sparse = model_sparse(input_sparse, target)
with nvtx.annotate("sp backward"):
    with scale_loss(loss_sparse, optimizer_sparse) as scaled_loss:
        scaled_loss.backward()
with nvtx.annotate("sp update"):
    optimizer_sparse.step()

# assert torch.allclose(loss_sparse, loss_ref)
assert torch.sum(torch.isclose(model_sparse.orig_module.classifier.classifier.weight.grad, model.classifier.classifier.weight.grad, rtol=1e-2)) / model_sparse.orig_module.classifier.classifier.weight.grad.numel() > 0.95
assert torch.sum(torch.isclose(model_sparse.orig_module.classifier.classifier.bias.grad, model.classifier.classifier.bias.grad, rtol=1e-2)) / model_sparse.orig_module.classifier.classifier.bias.grad.numel() > 0.95
assert torch.sum(torch.isclose(input_sparse.grad, input.grad, rtol=1e-2)) / input_sparse.grad.numel() > 0.95


optimizer.zero_grad()
optimizer_sparse.zero_grad()

with nvtx.annotate("forward"):
    loss_ref = model(input, target)
with nvtx.annotate("backward"):
    with scale_loss(loss_ref, optimizer) as scaled_loss:
        scaled_loss.backward()
with nvtx.annotate("update"):
    optimizer.step()

with nvtx.annotate("sp forward"):
    loss_sparse = model_sparse(input_sparse, target)
with nvtx.annotate("sp backward"):
    with scale_loss(loss_sparse, optimizer_sparse) as scaled_loss:
        scaled_loss.backward()
with nvtx.annotate("sp update"):
    optimizer_sparse.step()


# assert torch.allclose(loss_sparse, loss_ref)
assert torch.sum(torch.isclose(model_sparse.orig_module.classifier.classifier.weight.grad, model.classifier.classifier.weight.grad, rtol=1e-2)) / model_sparse.orig_module.classifier.classifier.weight.grad.numel() > 0.95
assert torch.sum(torch.isclose(model_sparse.orig_module.classifier.classifier.bias.grad, model.classifier.classifier.bias.grad, rtol=1e-2)) / model_sparse.orig_module.classifier.classifier.bias.grad.numel() > 0.95
assert torch.sum(torch.isclose(input_sparse.grad, input.grad, rtol=1e-2)) / input_sparse.grad.numel() > 0.9
