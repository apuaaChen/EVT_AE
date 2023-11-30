from slapo.op import FlashAttention
import torch.nn as nn
import torch
import sys
from model_zoo.vit.vit_modeling import FeedForward, Attention
from xformers.triton import FusedLayerNorm
from xformers.components.feedforward import FusedMLP
from xformers.components import Activation

################################################################################
# Optimized modules from xformer
################################################################################
class xFlashAttention(nn.Module):
    """A wrapper to align the original BertSelfAttention forward signature."""

    def __init__(self, dim, heads, dim_head):
        super().__init__()
        self.module = FlashAttention(
            hidden_size=dim,
            num_attention_heads=heads, 
            bias=False, fused_qkv=True, attn_pdrop=1e-19,
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
        hidden_states = torch.permute(hidden_states, [1, 0, 2])
        outputs = self.module(hidden_states, None, past_key_value)[0]
        # FIXME: The original output is (hidden_states, None) where the None
        # is present_key_value and only used by decoder (e.g., GPT).
        return outputs.permute([1, 0, 2])
    
class xFusedMLP(nn.Module):
    def __init__(self, dim, hidden_dim):
        super().__init__()
        self.module = FusedMLP(
            dim_model=dim, hidden_layer_multiplier = int(hidden_dim / dim), 
            activation=Activation.GeLU, dropout=1e-19, bias=False)
    
    def forward(self, x):
        return self.module(x)

# class xFusedIntermediate(nn.Module):
#     def __init__(self, config):
#         super().__init__()
#         self.dense_act = FusedLinear(
#             config.hidden_size, config.intermediate_size, 
#             activation=config.hidden_act)
    
#     def forward(self, hidden_states):
#         hidden_states = self.dense_act(hidden_states)
#         return hidden_states

# class xFusedOutput(nn.Module):
#     def __init__(self, config):
#         super().__init__()
#         self.dense = nn.Linear(config.intermediate_size, config.hidden_size)
#         self.LayerNorm = FusedLayerNorm(config.hidden_size)
#         self.dropout = self.dropout = nn.Dropout(config.hidden_dropout_prob)
    
#     def forward(self, hidden_states, input_tensor):
#         hidden_states = self.dense(hidden_states)
#         hidden_states = self.dropout(hidden_states)
#         hidden_states = self.LayerNorm(hidden_states + input_tensor)
#         return hidden_states

    

################################################################################
# Slapo Registration
################################################################################
def pass_slapo(sch):
    for subsch in sch.child:
        if isinstance(sch[subsch].mod, Attention):
            dim = sch[subsch].mod.dim
            heads = sch[subsch].mod.heads
            dim_head = sch[subsch].mod.dim_head
            new_mod = xFlashAttention(dim, heads, dim_head)
            sch[subsch].replace(new_mod)
        elif isinstance(sch[subsch].mod, nn.LayerNorm):
            new_mod = FusedLayerNorm(sch[subsch].mod.normalized_shape)
            sch[subsch].replace(new_mod)
        if isinstance(sch[subsch].mod, FeedForward):
            dim = sch[subsch].mod.dim
            hidden_dim = sch[subsch].mod.hidden_dim
            new_mod = xFusedMLP(dim, hidden_dim)
            sch[subsch].replace(new_mod)
        
        # elif isinstance(sch[subsch].mod, BertIntermediate):
        #     new_mod = xFusedIntermediate(config)
        #     sch[subsch].replace(new_mod)
        # elif isinstance(sch[subsch].mod, BertOutput):
        #     new_mod = xFusedOutput(config)
        #     sch[subsch].replace(new_mod)
        else:
            pass_slapo(sch[subsch])
