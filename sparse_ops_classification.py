from torch import autograd
import torch.nn.functional as F
import torch
from sptrain.spmmt_reduce import spmmt_reduce_f16_nnn
from sptrain.spmm import spmmv2_f16_nnn
from sptrain.sddmm_target import sddmm_target_f16_ntn
from sptrain.softmax import softmax
import math


class ExtremeClassifier_Ref_Fn(autograd.Function):

    @staticmethod
    def forward(ctx, input, weight, bias, target, smoothing, padding_idx=0):

        vocab_size = weight.size(0)
        out = F.linear(input, weight, bias).view(-1, vocab_size)
        probs = F.softmax(out, dim=-1)
        non_pad_mask = (target != padding_idx)

        ## uncommont this if loss is not required
        # logprobs = torch.log(probs)
        # nll_loss = -logprobs.gather(dim=-1, index=target.unsqueeze(1))
        # nll_loss = nll_loss.squeeze(1)[non_pad_mask]
        # smooth_loss = -logprobs.mean(dim=-1)[non_pad_mask]
        # loss = (1. - smoothing) * nll_loss + smoothing * smooth_loss
        # loss = loss.sum()
        ## end comment

        ctx.save_for_backward(input, target, probs, weight, non_pad_mask)
        ctx.vocab_size = vocab_size
        ctx.smoothing = smoothing

        return torch.empty(size=(1,), device="cuda")

        # return torch.Tensor([1.]).to("cuda")
    
    @staticmethod
    def backward(ctx, grad_loss):

        input, target, O_softmax, weight, non_pad_mask = ctx.saved_tensors

        vocab_size, hidden_size = weight.size()

        bp_mask = non_pad_mask.unsqueeze(1).to(O_softmax.dtype)

        Y_hat = (F.one_hot(target, vocab_size) * (1. - ctx.smoothing) + ctx.smoothing / vocab_size).to(O_softmax.dtype)

        X = input.view(-1, hidden_size)

        grad_out = (-Y_hat + O_softmax) * bp_mask * grad_loss.to(O_softmax.dtype)

        grad_W = torch.matmul(grad_out.t(), X)
        grad_b = torch.sum(grad_out, dim=0)
        grad_input = torch.matmul(grad_out, weight)

        return grad_input.view(input.size()), grad_W, grad_b, None, None, None


class ExtremeClassifier_Fn(autograd.Function):

    @staticmethod
    def forward(ctx, input, weight, bias, target, smoothing, padding_idx=0, training=True):

        vocab_size = weight.size(0)
        out = F.linear(input, weight, bias).view(-1, vocab_size)
        if (training):
            probs = softmax(out, -1, -1.*smoothing / vocab_size, target, smoothing - 1., padding_idx)

            ctx.save_for_backward(input, probs, weight)
            ctx.vocab_size = vocab_size
            ctx.smoothing = smoothing

            return torch.empty(size=(1,), device="cuda")
        
        else:
            non_pad_mask = (target != padding_idx)
            logprobs = F.log_softmax(probs, -1, dtype=torch.float32)
            nll_loss = -logprobs.gather(dim=-1, index=target.unsqueeze(1))
            nll_loss = nll_loss.squeeze(1)[non_pad_mask]
            smooth_loss = -logprobs.mean(dim=-1)[non_pad_mask]
            loss = (1. - smoothing) * nll_loss + smoothing * smooth_loss
            loss = loss.sum()

            return loss
    
    @staticmethod
    def backward(ctx, grad_loss):

        input, O_softmax, weight = ctx.saved_tensors

        hidden_size = weight.size(1)

        X = input.view(-1, hidden_size)

        grad_out = O_softmax * grad_loss.to(O_softmax.dtype)

        grad_W = torch.matmul(grad_out.t(), X)
        grad_b = torch.sum(grad_out, dim=0)
        grad_input = torch.matmul(grad_out, weight)

        return grad_input.view(input.size()), grad_W, grad_b, None, None, None, None


class SpExtremeClassifier_Ref_Fn(autograd.Function):

    @staticmethod
    def forward(ctx, input, weight, bias, target, smoothing, padding_idx=0):

        vocab_size = weight.size(0)
        out = F.linear(input, weight, bias).view(-1, vocab_size)

        Y_hat = F.one_hot(target, vocab_size) * out.max()

        # TODO: prune the kernel here
        group = int(out.numel() / 4)
        prob_tmp = (out + Y_hat).detach().reshape(group, 4).to(torch.float32)
        index = torch.argsort(prob_tmp, dim=1)[:, :int(2)]

        mask = torch.zeros(prob_tmp.shape, device=prob_tmp.device, dtype=out.dtype)
        mask = mask.scatter_(dim=1, index=index, value=-math.inf).reshape(out.shape)

        out = out + mask

        mask_Y_hat = 1. - torch.nan_to_num(mask, nan=1.0, posinf=1.0, neginf=1.0)

        probs = F.softmax(out, dim=-1)

        Y_hat_smooth = (F.one_hot(target, vocab_size) * (1. - smoothing) + 2. * smoothing / vocab_size).to(out.dtype)

        non_pad_mask = (target != padding_idx)

        bp_mask = non_pad_mask.unsqueeze(1).to(out.dtype)

        out_hat = (-Y_hat_smooth + probs) * mask_Y_hat * bp_mask

        ctx.save_for_backward(input, out_hat, weight)
        ctx.vocab_size = vocab_size
        ctx.smoothing = smoothing
        
        return torch.empty(size=(1,), device="cuda")
        # return torch.Tensor([1.]).to("cuda")
    
    @staticmethod
    def backward(ctx, grad_loss):

        input, out_hat, weight = ctx.saved_tensors

        hidden_size = weight.size(1)

        X = input.view(-1, hidden_size)

        grad_out = out_hat * grad_loss.to(out_hat.dtype)

        print(grad_out)

        grad_W = torch.matmul(grad_out.t(), X)
        grad_b = torch.sum(grad_out, dim=0)
        grad_input = torch.matmul(grad_out, weight)

        return grad_input.view(input.size()), grad_W, grad_b, None, None, None



class SpExtremeClassifier_Fn(autograd.Function):

    @staticmethod
    def forward(ctx, input, weight, bias, target, smoothing, padding_idx=0):

        vocab_size, hidden_size = weight.size()
        input_ = input.view(-1, hidden_size)
        target = target.view(-1, 1)

        nonzeros, metadata, target_sp = sddmm_target_f16_ntn(input_, weight, target, bias, 1.)
        probs = softmax(nonzeros.squeeze(), -1, -2.*smoothing / vocab_size, target_sp, smoothing - 1., padding_idx)

        print(probs)


        ctx.save_for_backward(input, probs, weight, metadata)
        ctx.vocab_size = vocab_size
        ctx.smoothing = smoothing

        return torch.empty(size=(1,), device="cuda")
    
    @staticmethod
    def backward(ctx, grad_loss):

        input, out_hat, weight, metadata = ctx.saved_tensors

        hidden_size = weight.size(1)

        X = input.view(-1, hidden_size)

        grad_W, grad_b = spmmt_reduce_f16_nnn(out_hat, X, metadata, grad_loss.item())
        grad_input = spmmv2_f16_nnn(out_hat, weight, metadata, grad_loss.item())

        return grad_input.view(input.size()), grad_W, grad_b, None, None, None