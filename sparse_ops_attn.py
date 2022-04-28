from torch import autograd, nn
import torch.nn.functional as F
import math
import torch

# class Sddmm(autograd.Function):
#     """prune the unimportant attention weights"""

#     @staticmethod
#     def forward(ctx, query_layer, key_layer):
#         attention_scores = torch.


def sddmm_reference(query_layer, key_layer, mask, alpha):
    batch_size = mask.size(0)
    num_attention_heads = int(query_layer.size(0) / batch_size)
    seq_length = query_layer.size(1)

    attention_scores = torch.bmm(query_layer, key_layer)
    attention_scores = attention_scores.view(
        batch_size, num_attention_heads, seq_length, seq_length)
    
    ## /sqrt(d)
    attention_scores = attention_scores * alpha

    ## mask
    attention_scores = attention_scores + mask

    # print(attention_scores[0][0][0][0:32])

    attention_scores = attention_scores.view(
        batch_size * num_attention_heads, seq_length, seq_length)

    # prune the attention scores
    max_matrix_scores, indices = F.max_pool1d(attention_scores, kernel_size=4, stride=4, return_indices=True)
    base = torch.empty_like(attention_scores).fill_(-math.inf)
    base = base.scatter_(2, indices, max_matrix_scores)
    attention_scores = attention_scores.scatter(2, indices, -math.inf)
    max_matrix_scores, indices = F.max_pool1d(attention_scores, kernel_size=4, stride=4, return_indices=True)
    attention_scores = base.scatter(2, indices, max_matrix_scores)

    attention_scores = attention_scores.view(
        batch_size, num_attention_heads, seq_length, seq_length)
    
    # print(attention_scores[0][0][0][0:32])

    return attention_scores


class Sddmm(autograd.Function):
    """prune the unimportant attention weight"""

    @staticmethod
    def forward(ctx, query_layer, key_layer, mask, alpha):
        batch_size = mask.size(0)
        num_attention_heads = int(query_layer.size(0) / batch_size)
        seq_length = query_layer.size(1)

        attention_scores = torch.bmm(query_layer, key_layer)
        attention_scores = attention_scores.view(
            batch_size, num_attention_heads, seq_length, seq_length)
        
        ## /sqrt(d)
        attention_scores = attention_scores * alpha

        ## mask
        attention_scores = attention_scores + mask

        ctx.seq_length = seq_length
        ctx.batch_size = batch_size
        ctx.num_attention_heads = num_attention_heads
        ctx.alpha = alpha

        # in a different way: generate the binary mask to be applied
        attention_scores = attention_scores.view(
            batch_size * num_attention_heads, seq_length, seq_length)
        max_matrix_scores, indices = F.max_pool1d(attention_scores, kernel_size=4, stride=4, return_indices=True)
        
        base = torch.empty_like(attention_scores).fill_(-math.inf)
        base = base.scatter_(2, indices, max_matrix_scores)
        attention_scores = attention_scores.scatter(2, indices, -math.inf)
        max_matrix_scores, indices = F.max_pool1d(attention_scores, kernel_size=4, stride=4, return_indices=True)
        attention_scores = base.scatter(2, indices, max_matrix_scores)
        ctx.save_for_backward(query_layer, key_layer)
        
        attention_scores = attention_scores.view(
            batch_size, num_attention_heads, seq_length, seq_length)

        return attention_scores
    
    @staticmethod
    def backward(ctx, grad_attention_scores):
        grad_attn_score = grad_attention_scores * ctx.alpha

        query_layer, key_layer = ctx.saved_tensors

        # Notably, we don't need the backpropagation code for the pruning. As the grad_attention_scores is naturally sparse
        grad_attn_score = grad_attn_score.view(ctx.batch_size * ctx.num_attention_heads, ctx.seq_length, ctx.seq_length)
        grad_query = torch.bmm(grad_attn_score, key_layer.transpose(1, 2))
        grad_key = torch.bmm(query_layer.transpose(1, 2), grad_attn_score)
        
        return grad_query, grad_key, None, None
