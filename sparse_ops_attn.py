from torch import autograd, nn
import torch.nn.functional as F
from sptrain.sddmm import sddmm_f16_ntn
from sptrain.spmm import spmmv2_f16_nnn, spmmv2_f16_ntn
from sptrain.spmmt import spmmt_f16_ntn, spmmt_f16_ntt, spmmt_f16_nnn, spmmt_f16_nnt
from sptrain.meta import bdense2sparse
import math
import torch
import nvtx

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
    # attention_scores = attention_scores * alpha

    ## mask
    attention_scores = (attention_scores + mask) * alpha

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

    return attention_scores


class Sddmm(autograd.Function):
    """prune the unimportant attention weight"""

    # TODO: there is some issue with the key bias gradient

    @staticmethod
    def forward(ctx, query_layer, key_layer, mask, alpha):
        batch_size = mask.size(0)
        num_attention_heads = int(query_layer.size(0) / batch_size)
        seq_length = query_layer.size(1)

        nonzeros, metadata = sddmm_f16_ntn(query_layer.contiguous(), key_layer.contiguous(), mask, alpha)

        ctx.seq_length = seq_length
        ctx.batch_size = batch_size
        ctx.num_attention_heads = num_attention_heads
        ctx.alpha = alpha

        ctx.save_for_backward(query_layer, key_layer, metadata)

        attention_scores = nonzeros.view(batch_size, num_attention_heads, seq_length, int(seq_length / 2))

        return attention_scores, metadata
    
    @staticmethod
    def backward(ctx, grad_attention_scores, grad_metadata):
        with nvtx.annotate("Grad SDDMM"):
            query_layer, key_layer, metadata = ctx.saved_tensors

            grad_query = spmmv2_f16_nnn(grad_attention_scores, key_layer.contiguous(), metadata, ctx.alpha)
            grad_key = spmmt_f16_nnn(grad_attention_scores, query_layer.contiguous(), metadata, ctx.alpha)
            
            return grad_query, grad_key, None, None


class Softmax(autograd.Function):

    @staticmethod
    def forward(ctx, x):
        out = F.softmax(x, dim=-1)
        ctx.save_for_backward(out, x)
        return out
    
    @staticmethod
    def backward(ctx, grad_out):
        out, x = ctx.saved_tensors
        # TODO: There is still inf->nan bug in this code. But the sparsity comes from *out and *exp(x). Both of which have 0s
        # at the masked out elements. In another word, we can use a compact grad_out under the N:M sparsity
        # the bug is caused by 0 / 0
        grad_x = grad_out * out - torch.sum(grad_out * out * out/ torch.exp(x), dim=-1, keepdim=True) * torch.exp(x)
        return grad_x


class Spmm(autograd.Function):

    @staticmethod
    def forward(ctx, lhs_matrix, rhs_matrix, metadata):
        output = spmmv2_f16_nnn(lhs_matrix, rhs_matrix.contiguous(), metadata, 1.)

        ctx.batch_size = lhs_matrix.size(0)
        ctx.num_attention_heads = lhs_matrix.size(1)
        ctx.seq_length = lhs_matrix.size(2)
        ctx.save_for_backward(lhs_matrix, rhs_matrix, metadata)

        return output
    
    @staticmethod
    def backward(ctx, grad_out):
        with nvtx.annotate("Grad Spmm"):
            eyes = []
            for i in range(ctx.batch_size * ctx.num_attention_heads):
                eyes.append(torch.eye(n=ctx.seq_length, dtype=torch.float16, device="cuda").unsqueeze(0))

            eyes = torch.cat(eyes, dim=0)
            lhs_matrix, rhs_matrix, metadata = ctx.saved_tensors
            

            # The naive implementation of the kernel
            # lhs_matrix_dense = spmmv2_f16_nnn(lhs_matrix, eyes, metadata)
            # grad_rhs_matrix = torch.bmm(lhs_matrix_dense.transpose(1, 2).contiguous(), grad_out)

            grad_rhs_matrix = spmmt_f16_nnn(lhs_matrix.contiguous(), grad_out.contiguous(), metadata, 1.)
        
            # TODO: handle this part with a fused kernel
            # prune the lhs_matrix
            grad_lhs_matrix = torch.bmm(grad_out.contiguous(), rhs_matrix.transpose(1, 2).contiguous())
            mask = spmmv2_f16_nnn(torch.ones_like(lhs_matrix), eyes, metadata, 1.)
            grad_lhs_matrix *= mask
            grad_lhs_matrix, _ = bdense2sparse(grad_lhs_matrix, True)

            return grad_lhs_matrix.view(lhs_matrix.size()), grad_rhs_matrix.view(rhs_matrix.size()), None

