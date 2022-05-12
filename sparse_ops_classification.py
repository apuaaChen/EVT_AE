import imp
from os import stat
from numpy import nonzero
from torch import autograd
import torch.nn.functional as F
import torch
from sptrain.sddmm import sddmm_f16_ntn
from sptrain.spmmt import spmmt_f16_nnn
from sptrain.spmm import spmmv2_f16_nnn

class ExtremeClassifier_Fn(autograd.Function):

    @staticmethod
    def forward(ctx, input, weight, bias, target, smoothing):

        vocab_size = weight.size(0)
        out = F.linear(input, weight, bias).view(-1, vocab_size)
        probs = F.softmax(out, dim=-1)
        # logprobs = torch.log(probs)
        non_pad_mask = (target != 0)
        # nll_loss = -logprobs.gather(dim=-1, index=target.unsqueeze(1))
        # nll_loss = nll_loss.squeeze(1)[non_pad_mask]
        # smooth_loss = -logprobs.mean(dim=-1)[non_pad_mask]
        # loss = (1. - smoothing) * nll_loss + smoothing * smooth_loss
        # loss = loss.sum()

        ctx.save_for_backward(input, target, probs, weight, non_pad_mask)
        ctx.vocab_size = vocab_size
        ctx.smoothing = smoothing

        return torch.Tensor([1.]).to("cuda")
    
    @staticmethod
    def backward(ctx, grad_loss):

        input, target, O_softmax, weight, non_pad_mask = ctx.saved_tensors

        vocab_size, hidden_size = weight.size()

        bp_mask = non_pad_mask.unsqueeze(1).to(O_softmax.dtype)

        Y_hat = (F.one_hot(target, vocab_size) * (1. - ctx.smoothing) + ctx.smoothing / vocab_size).to(O_softmax.dtype)

        X = input.view(-1, hidden_size)

        grad_out = (-Y_hat + O_softmax) * bp_mask

        grad_W = torch.matmul(grad_out.t(), X)
        grad_b = torch.sum(grad_out, dim=0)
        grad_input = torch.matmul(grad_out, weight)

        return grad_input.view(input.size()) * grad_loss, grad_W* grad_loss, grad_b* grad_loss, None, None



class SpExtremeClassifier_Fn(autograd.Function):

    @staticmethod
    def forward(ctx, input, weight, bias, target, smoothing):

        vocab_size, hidden_size = weight.size()
        print(vocab_size)
        # out = F.linear(input, weight, bias).view(-1, vocab_size)
        nonzeros, metadata = sddmm_f16_ntn(input.view(-1, hidden_size).contiguous(), weight, bias, 1.).view(-1, vocab_size // 2)
        probs = F.softmax(nonzeros, dim=-1)
        # logprobs = torch.log(probs)
        non_pad_mask = (target != 0)
        # nll_loss = -logprobs.gather(dim=-1, index=target.unsqueeze(1))
        # nll_loss = nll_loss.squeeze(1)[non_pad_mask]
        # smooth_loss = -logprobs.mean(dim=-1)[non_pad_mask]
        # loss = (1. - smoothing) * nll_loss + smoothing * smooth_loss
        # loss = loss.sum()

        ctx.save_for_backward(input, target, probs, weight, non_pad_mask, metadata)
        ctx.vocab_size = vocab_size
        ctx.smoothing = smoothing

        return torch.Tensor([1.]).to("cuda")
    
    @staticmethod
    def backward(ctx, grad_loss):

        input, target, O_softmax, weight, non_pad_mask, metadata = ctx.saved_tensors

        vocab_size, hidden_size = weight.size()

        # bp_mask = non_pad_mask.unsqueeze(1).to(O_softmax.dtype)

        # Y_hat = (F.one_hot(target, vocab_size) * (1. - ctx.smoothing) + ctx.smoothing / vocab_size).to(O_softmax.dtype)

        X = input.view(-1, hidden_size)

        # grad_out = (-Y_hat + O_softmax) * bp_mask

        grad_out = O_softmax

        grad_W = torch.matmul(grad_out.t(), X)
        grad_b = torch.sum(grad_out.view(-1, vocab_size), dim=0)
        grad_input = torch.matmul(grad_out, weight)

        return grad_input.view(input.size()), grad_W, grad_b, None, None