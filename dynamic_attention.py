import torch
import nvtx
import json
import torch.nn as nn
import sys
import copy
from apex.optimizers import FusedAdam
from apex import amp
import math
import torch.nn.functional as F
from sparse_ops_attn import sddmm_reference, Sddmm, Softmax, Spmm


config_file = "./BERT/BERT/bert_configs/large.json"

class BertConfig(object):
    """Configuration class to store the configuration of a `BertModel`.
    """
    def __init__(self,
                 vocab_size_or_config_json_file,
                 hidden_size=768,
                 num_hidden_layers=12,
                 num_attention_heads=12,
                 intermediate_size=3072,
                 hidden_act="gelu",
                 hidden_dropout_prob=0.1,
                 attention_probs_dropout_prob=0.1,
                 max_position_embeddings=512,
                 type_vocab_size=2,
                 initializer_range=0.02,
                 output_all_encoded_layers=False):
        """Constructs BertConfig.

        Args:
            vocab_size_or_config_json_file: Vocabulary size of `inputs_ids` in `BertModel`.
            hidden_size: Size of the encoder layers and the pooler layer.
            num_hidden_layers: Number of hidden layers in the Transformer encoder.
            num_attention_heads: Number of attention heads for each attention layer in
                the Transformer encoder.
            intermediate_size: The size of the "intermediate" (i.e., feed-forward)
                layer in the Transformer encoder.
            hidden_act: The non-linear activation function (function or string) in the
                encoder and pooler. If string, "gelu", "relu" and "swish" are supported.
            hidden_dropout_prob: The dropout probabilitiy for all fully connected
                layers in the embeddings, encoder, and pooler.
            attention_probs_dropout_prob: The dropout ratio for the attention
                probabilities.
            max_position_embeddings: The maximum sequence length that this model might
                ever be used with. Typically set this to something large just in case
                (e.g., 512 or 1024 or 2048).
            type_vocab_size: The vocabulary size of the `token_type_ids` passed into
                `BertModel`.
            initializer_range: The sttdev of the truncated_normal_initializer for
                initializing all weight matrices.
        """
        if isinstance(vocab_size_or_config_json_file, str) or (sys.version_info[0] == 2
                        and isinstance(vocab_size_or_config_json_file, unicode)):
            with open(vocab_size_or_config_json_file, "r", encoding='utf-8') as reader:
                json_config = json.loads(reader.read())
            for key, value in json_config.items():
                self.__dict__[key] = value
        elif isinstance(vocab_size_or_config_json_file, int):
            self.vocab_size = vocab_size_or_config_json_file
            self.hidden_size = hidden_size
            self.num_hidden_layers = num_hidden_layers
            self.num_attention_heads = num_attention_heads
            self.hidden_act = hidden_act
            self.intermediate_size = intermediate_size
            self.hidden_dropout_prob = hidden_dropout_prob
            self.attention_probs_dropout_prob = attention_probs_dropout_prob
            self.max_position_embeddings = max_position_embeddings
            self.type_vocab_size = type_vocab_size
            self.initializer_range = initializer_range
            self.output_all_encoded_layers = output_all_encoded_layers
        else:
            raise ValueError("First argument must be either a vocabulary size (int)"
                             "or the path to a pretrained model config file (str)")

    @classmethod
    def from_dict(cls, json_object):
        """Constructs a `BertConfig` from a Python dictionary of parameters."""
        config = BertConfig(vocab_size_or_config_json_file=-1)
        for key, value in json_object.items():
            config.__dict__[key] = value
        return config

    @classmethod
    def from_json_file(cls, json_file):
        """Constructs a `BertConfig` from a json file of parameters."""
        with open(json_file, "r", encoding='utf-8') as reader:
            text = reader.read()
        return cls.from_dict(json.loads(text))

    def __repr__(self):
        return str(self.to_json_string())

    def to_dict(self):
        """Serializes this instance to a Python dictionary."""
        output = copy.deepcopy(self.__dict__)
        return output

    def to_json_string(self):
        """Serializes this instance to a JSON string."""
        return json.dumps(self.to_dict(), indent=2, sort_keys=True) + "\n"

    def to_json_file(self, json_file_path):
        """ Save this instance to a json file."""
        with open(json_file_path, "w", encoding='utf-8') as writer:
            writer.write(self.to_json_string())

config = BertConfig.from_json_file(config_file)

# class BertSelfAttention(nn.Module):
#     def __init__(self, config):
#         super(BertSelfAttention, self).__init__()
#         if config.hidden_size % config.num_attention_heads != 0:
#             raise ValueError(
#                 "The hidden size (%d) is not a multiple of the number of attention "
#                 "heads (%d)" % (config.hidden_size, config.num_attention_heads))
#         self.num_attention_heads = config.num_attention_heads
#         self.attention_head_size = int(config.hidden_size / config.num_attention_heads)
#         self.all_head_size = self.num_attention_heads * self.attention_head_size

#         self.query = nn.Linear(config.hidden_size, self.all_head_size)
#         self.key = nn.Linear(config.hidden_size, self.all_head_size)
#         self.value = nn.Linear(config.hidden_size, self.all_head_size)

#         self.dropout = nn.Dropout(config.attention_probs_dropout_prob)

#     def transpose_for_scores(self, x):
#         # seq: x.size(0), bsz: x.size(0)
#         x = x.view(x.size(0), x.size(1) * self.num_attention_heads, self.attention_head_size).transpose(0, 1)
#         return x

#     def transpose_key_for_scores(self, x):
#         # seq: x.size(0), bsz: x.size(0)
#         x = x.view(x.size(0), x.size(1) * self.num_attention_heads, self.attention_head_size).permute(1, 2, 0)
#         return x

#     def forward(self, hidden_states, attention_mask):
#         # (seq, bsz, hidden)
#         batch_size = hidden_states.size(1)
#         seq_length = hidden_states.size(0)
#         with nvtx.annotate("Q,K,V"):
#             mixed_query_layer = self.query(hidden_states)
#             mixed_key_layer = self.key(hidden_states)
#             mixed_value_layer = self.value(hidden_states)

#             query_layer = self.transpose_for_scores(mixed_query_layer)
#             key_layer = self.transpose_key_for_scores(mixed_key_layer)
#             value_layer = self.transpose_for_scores(mixed_value_layer)

#         with nvtx.annotate("QK^T"):
#             # Take the dot product between "query" and "key" to get the raw attention scores.
#             attention_scores = torch.bmm(query_layer, key_layer)

#             # (bsz, heads, seq, seq)
#             attention_scores = attention_scores.view(batch_size,
#                                                     self.num_attention_heads,
#                                                     seq_length, seq_length)
#         with nvtx.annotate("/sqrt(d)"):
#             attention_scores = attention_scores / math.sqrt(self.attention_head_size)
#         with nvtx.annotate("mask"):
#             # Apply the attention mask is (precomputed for all layers in BertModel forward() function)
#             attention_scores = attention_scores + attention_mask
#         with nvtx.annotate("softmax"):
#             # Normalize the attention scores to probabilities.
#             attention_probs = F.softmax(attention_scores, dim=-1)

#         with nvtx.annotate("dropout"):
#             # This is actually dropping out entire tokens to attend to, which might
#             # seem a bit unusual, but is taken from the original Transformer paper.
#             # (bsz, heads, seq, seq)
#             torch.manual_seed(9999)
#             attention_probs = self.dropout(attention_probs)
#             attention_probs = attention_probs.view(batch_size * self.num_attention_heads,
#                                                 seq_length, seq_length)
#         with nvtx.annotate("AV"):
#             context_layer = torch.bmm(attention_probs, value_layer)
#             context_layer = context_layer.transpose(0, 1).contiguous()
#         # (seq, bsz, hidden)
#         context_layer = context_layer.view(seq_length, batch_size, self.all_head_size)

#         return context_layer


class BertSelfAttention(nn.Module):
    def __init__(self, config):
        super(BertSelfAttention, self).__init__()
        if config.hidden_size % config.num_attention_heads != 0:
            raise ValueError(
                "The hidden size (%d) is not a multiple of the number of attention "
                "heads (%d)" % (config.hidden_size, config.num_attention_heads))
        self.num_attention_heads = config.num_attention_heads
        self.attention_head_size = int(config.hidden_size / config.num_attention_heads)
        self.all_head_size = self.num_attention_heads * self.attention_head_size

        self.query = nn.Linear(config.hidden_size, self.all_head_size)
        self.key = nn.Linear(config.hidden_size, self.all_head_size)
        self.value = nn.Linear(config.hidden_size, self.all_head_size)

        self.dropout = nn.Dropout(config.attention_probs_dropout_prob)

    def transpose_for_scores(self, x):
        # seq: x.size(0), bsz: x.size(0)
        x = x.view(x.size(0), x.size(1) * self.num_attention_heads, self.attention_head_size).transpose(0, 1)
        return x

    def transpose_key_for_scores(self, x):
        # seq: x.size(0), bsz: x.size(0)
        x = x.view(x.size(0), x.size(1) * self.num_attention_heads, self.attention_head_size).permute(1, 2, 0)
        return x

    def forward(self, hidden_states, attention_mask):
        # (seq, bsz, hidden)
        batch_size = hidden_states.size(1)
        seq_length = hidden_states.size(0)
        with nvtx.annotate("Q,K,V"):
            mixed_query_layer = self.query(hidden_states)
            mixed_key_layer = self.key(hidden_states)
            mixed_value_layer = self.value(hidden_states)

            query_layer = self.transpose_for_scores(mixed_query_layer)
            key_layer = self.transpose_key_for_scores(mixed_key_layer)
            value_layer = self.transpose_for_scores(mixed_value_layer)

        with nvtx.annotate("SDDMM"):
            attention_scores = sddmm_reference(
                query_layer, key_layer, attention_mask, 1./math.sqrt(self.attention_head_size))

        with nvtx.annotate("softmax"):
            # Normalize the attention scores to probabilities.
            attention_probs = F.softmax(attention_scores, dim=-1)
        with nvtx.annotate("dropout"):
            # This is actually dropping out entire tokens to attend to, which might
            # seem a bit unusual, but is taken from the original Transformer paper.
            # (bsz, heads, seq, seq)
            torch.manual_seed(9999)
            # attention_probs = self.dropout(attention_probs)
            attention_probs = attention_probs.view(batch_size * self.num_attention_heads,
                                                seq_length, seq_length)
        with nvtx.annotate("AV"):
            context_layer = torch.bmm(attention_probs, value_layer)
            context_layer = context_layer.transpose(0, 1).contiguous()
        # (seq, bsz, hidden)
        context_layer = context_layer.view(seq_length, batch_size, self.all_head_size)
        
        return context_layer


class SparseBertSelfAttention(nn.Module):
    def __init__(self, config):
        super(SparseBertSelfAttention, self).__init__()
        if config.hidden_size % config.num_attention_heads != 0:
            raise ValueError(
                "The hidden size (%d) is not a multiple of the number of attention "
                "heads (%d)" % (config.hidden_size, config.num_attention_heads))
        self.num_attention_heads = config.num_attention_heads
        self.attention_head_size = int(config.hidden_size / config.num_attention_heads)
        self.all_head_size = self.num_attention_heads * self.attention_head_size

        self.query = nn.Linear(config.hidden_size, self.all_head_size)
        self.key = nn.Linear(config.hidden_size, self.all_head_size)
        self.value = nn.Linear(config.hidden_size, self.all_head_size)

        self.dropout = nn.Dropout(config.attention_probs_dropout_prob)

    def transpose_for_scores(self, x):
        # seq: x.size(0), bsz: x.size(0)
        x = x.view(x.size(0), x.size(1) * self.num_attention_heads, self.attention_head_size).transpose(0, 1)
        return x

    def transpose_key_for_scores(self, x):
        # seq: x.size(0), bsz: x.size(0)
        x = x.view(x.size(0), x.size(1) * self.num_attention_heads, self.attention_head_size).permute(1, 2, 0)
        return x

    def forward(self, hidden_states, attention_mask):
        # (seq, bsz, hidden)
        batch_size = hidden_states.size(1)
        seq_length = hidden_states.size(0)
        with nvtx.annotate("Q,K,V"):
            mixed_query_layer = self.query(hidden_states)
            mixed_key_layer = self.key(hidden_states)
            mixed_value_layer = self.value(hidden_states)

            query_layer = self.transpose_for_scores(mixed_query_layer)
            key_layer = self.transpose_key_for_scores(mixed_key_layer)
            value_layer = self.transpose_for_scores(mixed_value_layer)

        # with nvtx.annotate("QK^T"):
        #     # Take the dot product between "query" and "key" to get the raw attention scores.
        #     attention_scores = torch.bmm(query_layer, key_layer)

        #     # (bsz, heads, seq, seq)
        #     attention_scores = attention_scores.view(batch_size,
        #                                             self.num_attention_heads,
        #                                             seq_length, seq_length)
        # with nvtx.annotate("/sqrt(d)"):
        #     attention_scores = attention_scores / math.sqrt(self.attention_head_size)
        # with nvtx.annotate("mask"):
        #     # Apply the attention mask is (precomputed for all layers in BertModel forward() function)
        #     attention_scores = attention_scores + attention_mask
        with nvtx.annotate("sddmm"):
            # attention_scores = sddmm_reference(
            #     query_layer, key_layer, attention_mask, 1./math.sqrt(self.attention_head_size))
            attention_scores, metadata = Sddmm.apply(query_layer, key_layer, attention_mask, 1./ math.sqrt(self.attention_head_size))
        with nvtx.annotate("softmax"):
            # Normalize the attention scores to probabilities.
            attention_probs = F.softmax(attention_scores, dim=-1)
            # attention_probs = Softmax.apply(attention_scores)
        with nvtx.annotate("dropout"):
            # This is actually dropping out entire tokens to attend to, which might
            # seem a bit unusual, but is taken from the original Transformer paper.
            # (bsz, heads, seq, seq)
            torch.manual_seed(9999)
            # attention_probs = self.dropout(attention_probs)
            # attention_probs = attention_probs.view(batch_size * self.num_attention_heads,
            #                                     seq_length, int(seq_length/2))
        with nvtx.annotate("AV"):
            # context_layer = torch.bmm(attention_probs, value_layer)
            context_layer = Spmm.apply(attention_probs, value_layer, metadata)
            context_layer = context_layer.view(batch_size * self.num_attention_heads,
                                                 seq_length, self.attention_head_size)
            context_layer = context_layer.transpose(0, 1).contiguous()
        # (seq, bsz, hidden)
        context_layer = context_layer.view(seq_length, batch_size, self.all_head_size)
        return context_layer


model = BertSelfAttention(config).to("cuda")
model_sparse = SparseBertSelfAttention(config).to("cuda")

# align the parameters

model_sparse.query.weight = torch.nn.Parameter(model.query.weight.clone())
model_sparse.key.weight = torch.nn.Parameter(model.key.weight.clone())
model_sparse.value.weight = torch.nn.Parameter(model.value.weight.clone())

model_sparse.query.bias = torch.nn.Parameter(model.query.bias.clone())
model_sparse.key.bias = torch.nn.Parameter(model.key.bias.clone())
model_sparse.value.bias = torch.nn.Parameter(model.value.bias.clone())

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

model, optimizer = amp.initialize(model, optimizer, opt_level="O2", keep_batchnorm_fp32=False, loss_scale="dynamic")
model_sparse, optimizer_sparse = amp.initialize(model_sparse, optimizer_sparse, opt_level="O2", keep_batchnorm_fp32=False, loss_scale="dynamic")

## Create the inputs
batch_size = 2
sequence_length = 1024
hidden = config.hidden_size

hidden_states = torch.randn(size=(sequence_length, batch_size, hidden), dtype=torch.float16, device="cuda", requires_grad=True)
hidden_states_sparse = hidden_states.detach().clone().requires_grad_(True)

prob = torch.ones(size=(batch_size, 1, 1, sequence_length), dtype=torch.float16, device="cuda") * 0.2
# TODO: the mask scale can be fixed with softmax
mask = torch.bernoulli(prob) * -1e4

## forward pass
output = model(hidden_states, mask)
output_sparse = model_sparse(hidden_states_sparse, mask)

def allclose(tensor, ref, rtol=5e-2, ntol=1e-1):
    error_abs = torch.abs(tensor - ref)
    scale_abs = torch.abs(ref)
    relative_error = error_abs / scale_abs
    n_error = (torch.ge(relative_error, rtol).sum() / tensor.numel()).item()

    print(tensor)
    print(ref)
    assert n_error < ntol, "get %.5f %% errors" % n_error

allclose(output_sparse, output)

grad_output = torch.randn(size=(sequence_length, batch_size, hidden), dtype=torch.float16, device="cuda")

## backward pass
output.backward(grad_output)
output_sparse.backward(grad_output)

# passed
allclose(hidden_states_sparse.grad, hidden_states.grad)
allclose(model_sparse.query.weight.grad, model.query.weight.grad)
allclose(model_sparse.key.weight.grad, model.key.weight.grad)
allclose(model_sparse.value.weight.grad, model.value.weight.grad)
allclose(model_sparse.query.bias.grad, model.query.bias.grad)
allclose(model_sparse.value.bias.grad, model.value.bias.grad)

# unpassed
allclose(model_sparse.key.bias.grad, model.key.bias.grad)



#######################################################
# Profiling
#######################################################
# for i in range(10):
#     with nvtx.annotate("forward"):
#         output = model(hidden_states, mask)
#     with nvtx.annotate("backward"):
#         output.backward(grad_output)
