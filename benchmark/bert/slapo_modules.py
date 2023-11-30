from slapo.op import FlashAttention
import torch.nn as nn
import torch
import sys
sys.path.append("/workspace/gtl/sparseTraining/thirdparty/DeepLearningExample/PyTorch/LanguageModeling/BERT")
from modeling import BertSelfAttention, BertConfig, BertAttention, BertIntermediate, BertOutput
from xformers.triton import FusedLayerNorm, FusedLinear

################################################################################
# Optimized modules from xformer
################################################################################
class xFlashAttention(nn.Module):
    """A wrapper to align the original BertSelfAttention forward signature."""

    def __init__(self, config):
        super().__init__()
        self.module = FlashAttention(
            hidden_size=config.hidden_size,
            num_attention_heads=config.num_attention_heads, 
            bias=False, fused_qkv=True, attn_pdrop=config.attention_probs_dropout_prob,
            attn_op_name="cutlass")
        # self.LayerNorm = nn.LayerNorm(config.hidden_size, eps=1e-12)
        self.LayerNorm= FusedLayerNorm(config.hidden_size)

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
        outputs = self.module(hidden_states, None, past_key_value)
        outputs = self.LayerNorm(outputs[0] + hidden_states)
        # FIXME: The original output is (hidden_states, None) where the None
        # is present_key_value and only used by decoder (e.g., GPT).
        return outputs.permute([1, 0, 2])
    

class xFusedIntermediate(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.dense_act = FusedLinear(
            config.hidden_size, config.intermediate_size, 
            activation=config.hidden_act)
    
    def forward(self, hidden_states):
        hidden_states = self.dense_act(hidden_states)
        return hidden_states

class xFusedOutput(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.dense = nn.Linear(config.intermediate_size, config.hidden_size)
        self.LayerNorm = FusedLayerNorm(config.hidden_size)
        self.dropout = self.dropout = nn.Dropout(config.hidden_dropout_prob)
    
    def forward(self, hidden_states, input_tensor):
        hidden_states = self.dense(hidden_states)
        hidden_states = self.dropout(hidden_states)
        hidden_states = self.LayerNorm(hidden_states + input_tensor)
        return hidden_states

    

################################################################################
# Slapo Registration
################################################################################
def pass_slapo(sch, config):
    for subsch in sch.child:
        if isinstance(sch[subsch].mod, BertAttention):
            new_mod = xFlashAttention(config)
            sch[subsch].replace(new_mod)
        elif isinstance(sch[subsch].mod, BertIntermediate):
            new_mod = xFusedIntermediate(config)
            sch[subsch].replace(new_mod)
        elif isinstance(sch[subsch].mod, BertOutput):
            new_mod = xFusedOutput(config)
            sch[subsch].replace(new_mod)
        else:
            pass_slapo(sch[subsch], config)
