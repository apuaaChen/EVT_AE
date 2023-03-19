import torch
import torch.nn as nn
import nvtx

################################################################################
# Optimized modules from LightSeq2
################################################################################
from lightseq.training import LSTransformerEmbeddingLayer

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



################################################################################
# Optimized modules from xformer
################################################################################
from slapo.op import FlashAttention


class xFlashAttention(nn.Module):
    """A wrapper to align the original BertSelfAttention forward signature."""

    def __init__(self, config):
        super().__init__()
        self.module = FlashAttention(
            hidden_size=config.hidden_size,
            num_attention_heads=config.num_attention_heads, 
            bias=False, fused_qkv=True, attn_pdrop=config.attention_probs_dropout_prob,
            attn_op_name="cutlass")

    def forward(
        self,
        hidden_states,
        attention_mask=None,
        head_mask=None,
        encoder_hidden_states=None,
        encoder_attention_mask=None,
        past_key_value=None,
        output_attentions=False,
    ):
        with nvtx.annotate("Flash Attention"):
            hidden_states = torch.permute(hidden_states, [1, 0, 2])
            outputs = self.module(hidden_states, None, past_key_value)
            # FIXME: The original output is (hidden_states, None) where the None
            # is present_key_value and only used by decoder (e.g., GPT).
            return outputs[0].permute([1, 0, 2])