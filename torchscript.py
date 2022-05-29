from cv2 import trace
import torch.nn as nn
import torch
from apex.optimizers import FusedAdam
from apex import amp
import nvtx
from apex.amp._amp_state import _amp_state
import contextlib
import torch.nn.functional as F
from sparse_ops_classification import ExtremeClassifier_Fn, ExtremeClassifier_Ref_Fn, SpExtremeClassifier_Fn, SpExtremeClassifier_Ref_Fn
from bert_config import allclose
import math
import sptrain_compile


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
        # out = self.classifier(x)
        out = torch.matmul(x, self.classifier.weight.t())
        out += self.classifier.bias
        return out


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
        logprobs = torch.nn.functional.log_softmax(x, dim=-1,
                                                   dtype=torch.float32)

        non_pad_mask = (target != self.padding_idx)
        nll_loss = -logprobs.gather(dim=-1, index=target.unsqueeze(1))
        nll_loss = nll_loss.squeeze(1)[non_pad_mask]
        smooth_loss = -logprobs.mean(dim=-1)[non_pad_mask]
        loss = self.confidence * nll_loss + self.smoothing * smooth_loss
        return loss.sum()


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
input = torch.load('./GNMT/input.pt').to("cuda").to(torch.float).requires_grad_(True)
#input.retain_grad()
target = torch.load('./GNMT/target.pt').to("cuda")

model = ExtremeClassifier(hidden_size, vocab_size, 0, 0.1).to("cuda")


param_optimizer = list(model.named_parameters())
param_optimizer = [n for n in param_optimizer if 'pooler' not in n[0]]
no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
optimizer_grouped_parameters = [
        {'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)], 'weight_decay': 0.01},
        {'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
    ]

optimizer = FusedAdam(optimizer_grouped_parameters, lr=0.1, bias_correction=False)

model, optimizer = amp.initialize(
    model, optimizer, 
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
# Torch Script
#########################

traced_model = torch.jit.trace(model, (input, target))
# traced_model = torch.jit.script(model)

print(traced_model.state_dict())

graph = traced_model.graph

print(graph)

torch._C._jit_pass_inline(graph)

print(graph)

# torch._C._jit_pass_fuse_linear(graph)
sptrain_compile.fuse_linear_v2(graph)

print(graph)

traced_model.eval()
print(traced_model.graph_for(input, target))


# graph_inputs = list(graph.inputs())
# print(graph_inputs)



#########################
# Functional Verification
#########################

optimizer.zero_grad()

with nvtx.annotate("traced"):
    with nvtx.annotate("forward"):
        loss = traced_model(input, target)
        print(loss)
    with nvtx.annotate("backward"):
        with scale_loss(loss, optimizer) as scaled_loss:
            scaled_loss.backward()
    with nvtx.annotate("update"):
        optimizer.step()


with nvtx.annotate("traced"):
    with nvtx.annotate("forward"):
        loss = traced_model(input, target)
        print(loss)
    with nvtx.annotate("backward"):
        with scale_loss(loss, optimizer) as scaled_loss:
            scaled_loss.backward()
    with nvtx.annotate("update"):
        optimizer.step()

with nvtx.annotate("untraced"):
    with nvtx.annotate("forward"):
        loss = model(input, target)
        print(loss)
    with nvtx.annotate("backward"):
        with scale_loss(loss, optimizer) as scaled_loss:
            scaled_loss.backward()
    with nvtx.annotate("update"):
        optimizer.step()