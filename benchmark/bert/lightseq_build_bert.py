import torch.nn as nn
from lightseq.training import LSTransformerEmbeddingLayer, LSTransformerEncoderLayer
import torch
import sys
sys.path.append("/workspace/sparseTraining/Codegen/compiler")
sys.path.append("/workspace/bert")
from modeling import BertForPreTraining, BertPreTrainingHeads, BertModel, BertPooler
import modeling
from typing import Final
import nvtx


# class LSLinearActivation(nn.Module):
#     r"""Fused Linear and activation Module.
#     """
#     __constants__ = ['bias']

#     def __init__(self, in_features, out_features, act='gelu', bias=True):
#         super(LSLinearActivation, self).__init__()
#         self.in_features = in_features
#         self.out_features = out_features
#         self.bias = None
#         assert act in ACT2FN, "Activation function is not found in activation dictionary."
#         self.act_fn = ACT2FN[act]
#         self.weight = Parameter(torch.Tensor(out_features, in_features))
#         if bias:
#             self.bias = Parameter(torch.Tensor(out_features))
#         else:
#             self.register_parameter('bias', None)
#         self.reset_parameters()

#     def reset_parameters(self):
#         init.kaiming_uniform_(self.weight, a=math.sqrt(5))
#         if self.bias is not None:
#             fan_in, _ = init._calculate_fan_in_and_fan_out(self.weight)
#             bound = 1 / math.sqrt(fan_in)
#             init.uniform_(self.bias, -bound, bound)

#     def forward(self, input):
#         #if not self.bias is None:
#         #    return self.biased_act_fn(self.bias, F.linear(input, self.weight, None))
#         #else:
#         return self.act_fn(F.linear(input, self.weight, self.bias))

#     def extra_repr(self):
#         return 'in_features={}, out_features={}, bias={}'.format(
#             self.in_features, self.out_features, self.bias is not None
#         )


## Corresponds to BertEmbeddings
class LSBertEmbeddings(nn.Module):
    def __init__(self, config) -> None:
        super().__init__()
        word_emb_config = LSTransformerEmbeddingLayer.get_config(
            vocab_size=config.vocab_size,
            embedding_dim = config.hidden_size,
            max_batch_tokens = 2**20,
            max_seq_len=config.max_position_embeddings,
            padding_idx=0,
            dropout=config.hidden_dropout_prob,
            fp16=True,
            local_rank=0
        )
        self.word_embeddings = LSTransformerEmbeddingLayer(word_emb_config)

        position_emb_config = LSTransformerEmbeddingLayer.get_config(
            vocab_size=config.max_position_embeddings,
            embedding_dim = config.hidden_size,
            max_batch_tokens = 2**20,
            max_seq_len=config.max_position_embeddings,
            padding_idx=0,
            dropout=config.hidden_dropout_prob,
            fp16=True,
            local_rank=0
        )

        self.position_embeddings = LSTransformerEmbeddingLayer(position_emb_config)

        token_emb_config = LSTransformerEmbeddingLayer.get_config(
            vocab_size=config.type_vocab_size,
            embedding_dim = config.hidden_size,
            max_batch_tokens = 2**20,
            max_seq_len=config.max_position_embeddings,
            padding_idx=0,
            dropout=config.hidden_dropout_prob,
            fp16=True,
            local_rank=0
        )

        self.token_type_embeddings = LSTransformerEmbeddingLayer(token_emb_config)

        self.LayerNorm = nn.LayerNorm(config.hidden_size, eps=1e-12)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

        self.distillation = getattr(config, 'distillation', False)
        self.distill_config = {'use_embedding_states' : False }

    def forward(self, input_ids, token_type_ids):
        seq_length = input_ids.size(1)
        position_ids = torch.arange(seq_length, dtype=torch.long, device=input_ids.device)
        position_ids = position_ids.unsqueeze(0).expand_as(input_ids)

        words_embeddings = self.word_embeddings(input_ids)
        position_embeddings = self.position_embeddings(position_ids)
        token_type_embeddings = self.token_type_embeddings(token_type_ids)

        embeddings = words_embeddings + position_embeddings + token_type_embeddings
        embeddings = self.LayerNorm(embeddings)
        embeddings = self.dropout(embeddings)

        if self.distillation:
            if self.distill_config["use_embedding_states"]:
                self.distill_state_dict["embedding_states"] = embeddings
        return embeddings


class LSBertLayer(nn.Module):
    def __init__(self, config) -> None:
        super().__init__()
        encoder_config = LSTransformerEncoderLayer.get_config(
            max_batch_tokens=16384,
            max_seq_len=config.max_position_embeddings,
            hidden_size=config.hidden_size,
            intermediate_size=config.intermediate_size,
            nhead=config.num_attention_heads,
            attn_prob_dropout_ratio=config.attention_probs_dropout_prob,
            activation_dropout_ratio=config.hidden_dropout_prob,
            hidden_dropout_ratio=config.hidden_dropout_prob,
            pre_layer_norm=True,
            activation_fn=config.hidden_act,
            fp16=True,
            local_rank=0
        )
        self.encoder_layer = LSTransformerEncoderLayer(encoder_config)
    
    def forward(self, hidden_states, attention_mask):
        return self.encoder_layer(hidden_states, attention_mask)

class LSBertEncoder(nn.Module):
    def __init__(self, config):
        super(LSBertEncoder, self).__init__()
        self.layer = nn.ModuleList([LSBertLayer(config) for _ in range(config.num_hidden_layers)])
        self.output_all_encoded_layers = config.output_all_encoded_layers
        self._checkpoint_activations = False

    def forward(self, hidden_states, attention_mask):
        hidden_states = hidden_states.permute(1, 0, 2).contiguous()
        attention_mask = attention_mask.squeeze()
        all_encoder_layers = []

        if self._checkpoint_activations:
            hidden_states = self.checkpointed_forward(hidden_states, attention_mask)
        else:
            # (bsz, seq, hidden) => (seq, bsz, hidden)
            hidden_states = hidden_states.transpose(0, 1)
            for i, layer_module in enumerate(self.layer):
                hidden_states = layer_module(hidden_states, attention_mask)

                if self.output_all_encoded_layers:
                    all_encoder_layers.append(hidden_states)
            # The hidden states need to be contiguous at this point to enable
            # dense_sequence_output
            # (seq, bsz, hidden) => (bsz, seq, hidden)
            # hidden_states = hidden_states.transpose(0, 1).contiguous()

        if not self.output_all_encoded_layers or self._checkpoint_activations:
            all_encoder_layers.append(hidden_states)
        return all_encoder_layers


class LSBertModelExcludeEmbedding(BertModel):
    def __init__(self, config, word_embeddings_weight) -> None:
        super(BertModel, self).__init__(config)
        self.word_embeddings_weight = word_embeddings_weight
        self.encoder = LSBertEncoder(config)
        self.pooler = BertPooler(config)
        self.apply(self.init_bert_weights)
        self.output_all_encoded_layers = config.output_all_encoded_layers
        self.teacher = False
    
    def forward(self, input_ids, embedding_output, attention_mask):
        # We create a 3D attention mask from a 2D tensor mask.
        # Sizes are [batch_size, 1, 1, to_seq_length]
        # So we can broadcast to [batch_size, num_heads, from_seq_length, to_seq_length]
        # this attention mask is more simple than the triangular masking of causal attention
        # used in OpenAI GPT, we just need to prepare the broadcast dimension here.
        if attention_mask is None:
            attention_mask = torch.ones_like(input_ids)

        extended_attention_mask = attention_mask.unsqueeze(1).unsqueeze(2)

        # Since attention_mask is 1.0 for positions we want to attend and 0.0 for
        # masked positions, this operation will create a tensor which is 0.0 for
        # positions we want to attend and -10000.0 for masked positions.
        # Since we are adding it to the raw scores before the softmax, this is
        # effectively the same as removing these entirely.
        extended_attention_mask = extended_attention_mask.to(dtype=self.word_embeddings_weight.dtype) # fp16 compatibility
        extended_attention_mask = (1.0 - extended_attention_mask) * -10000.0

        # embedding_output = self.embeddings(input_ids, token_type_ids)
        encoded_layers = self.encoder(embedding_output, extended_attention_mask)
        sequence_output = encoded_layers[-1]
        # Use pooler if not running distillation or distill_config["use_pooler"] is set to True
        pooled_output = self.pooler(sequence_output)
        if not self.output_all_encoded_layers:
            encoded_layers = encoded_layers[-1:]
        if not self.teacher:
            return encoded_layers, pooled_output


class LSBertForPreTrainingExcludeEmbedding(BertForPreTraining):
    def __init__(self, config, word_embeddings_weight, sequence_output_is_dense=False) -> None:
        super(BertForPreTraining, self).__init__(config)
        self.bert = LSBertModelExcludeEmbedding(config, word_embeddings_weight)
        self.cls = BertPreTrainingHeads(config, word_embeddings_weight, sequence_output_is_dense)
        self.apply(self.init_bert_weights)

    def forward(self, input_ids, embedding_output, attention_mask, masked_lm_labels):
        # if self.distillation:
        #     self.bert(input_ids, token_type_ids, attention_mask)
        # else:
        encoded_layers, pooled_output = self.bert(input_ids,embedding_output, attention_mask)
        sequence_output = encoded_layers[-1]
        prediction_scores, seq_relationship_score = self.cls(sequence_output, pooled_output, masked_lm_labels)
        return prediction_scores, seq_relationship_score


class BertPretrainingCriterion(torch.nn.Module):

    sequence_output_is_dense: Final[bool]

    def __init__(self, vocab_size, sequence_output_is_dense=False):
        super(BertPretrainingCriterion, self).__init__()
        self.loss_fn = torch.nn.CrossEntropyLoss(ignore_index=-1)
        self.vocab_size = vocab_size
        self.sequence_output_is_dense = sequence_output_is_dense

    def forward(self, prediction_scores, seq_relationship_score, masked_lm_labels, next_sentence_labels):
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

class LSBertExcludeEmbedding(torch.nn.Module):
    def __init__(self, config, word_embedding_weight, sequence_output_is_dense) -> None:
        super().__init__()
        self.model = LSBertForPreTrainingExcludeEmbedding(config, word_embedding_weight, sequence_output_is_dense)
        self.criterion = BertPretrainingCriterion(config.vocab_size, sequence_output_is_dense=sequence_output_is_dense)
    
    def forward(self, input_ids, embedding_output, attention_mask, masked_lm_labels, labels, next_sentence_labels):
        prediction_scores, seq_relationship_score = self.model(input_ids, embedding_output, attention_mask, masked_lm_labels)
        loss = self.criterion(prediction_scores, seq_relationship_score, labels, next_sentence_labels)
        return loss
    
    def checkpoint_activations(self, check):
        self.model.checkpoint_activations(check)

class LSBert(torch.nn.Module):
    def __init__(self, config, sequence_output_is_dense=False) -> None:
        super().__init__()
        self.config = config
        self.embedding = LSBertEmbeddings(config)
        self.encoder = LSBertExcludeEmbedding(config, self.embedding.word_embeddings.embeddings, sequence_output_is_dense)
    
    def forward(self, input_ids, token_type_ids, attention_mask, masked_lm_labels, labels, next_sentence_labels):
        embedding_output = self.embedding(input_ids, token_type_ids)
        return self.encoder(input_ids, embedding_output, attention_mask, masked_lm_labels, labels, next_sentence_labels)
    
    def checkpoint_activations(self, check):
        self.encoder.checkpoint_activations(check)

if __name__ == "__main__":
    i = 2
    input_ids = torch.load("/workspace/bert/batch/input_ids_iter%d.pt"%i).unsqueeze(0).expand([4, 8, 512]).contiguous().view([-1, 512]).to("cuda")
    token_type_ids = torch.load("/workspace/bert/batch/token_type_ids_iter%d.pt"%i).unsqueeze(0).expand([4, 8, 512]).contiguous().view([-1, 512]).to("cuda")
    attention_mask = torch.load("/workspace/bert/batch/attention_mask_iter%d.pt"%i).to(torch.float16).unsqueeze(0).expand([4, 8, 512]).contiguous().view([-1, 512]).to("cuda")
    next_sentence_labels = torch.load("/workspace/bert/batch/next_sentence_labels_iter%d.pt"%i).unsqueeze(0).expand([4, 8]).contiguous().view([-1]).to("cuda")
    labels = torch.load("/workspace/bert/batch/labels_iter%d.pt"%i).unsqueeze(0).expand([4, 8, 512]).contiguous().view([-1, 512]).to("cuda")

    config_file = "/workspace/sparseTraining/benchmark/bert/large.json" 
    config = modeling.BertConfig.from_json_file(config_file)
    if config.vocab_size % 8 != 0:
        config.vocab_size += 8 - (config.vocab_size % 8)

    model = LSBert(config, sequence_output_is_dense=False).to("cuda").to(torch.float16)

    for i in range(40):
        with nvtx.annotate("lightseq"):
            loss = model(input_ids, token_type_ids, attention_mask, labels, labels, next_sentence_labels)
            loss.backward()

