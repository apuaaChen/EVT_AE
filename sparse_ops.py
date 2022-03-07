"""
https://github.com/NM-sparsity/NM-sparsity/blob/main/devkit/sparse_ops/sparse_ops.py
"""

from curses import meta
from random import weibullvariate
from turtle import forward
import torch
from torch import autograd, nn
import torch.nn.functional as F
import nvtx
from sptrain.meta import bdense2sparse
from sptrain.spmm import spmmv2_bf16, spmmv2_bf16_nt


class Sparse(autograd.Function):
    """" Prune the unimprotant weight for the forwards phase but pass the gradient to dense weight using SR-STE in the backwards phase"""

    @staticmethod
    def forward(ctx, weight, N, M, decay = 0.0002):
        ctx.save_for_backward(weight)

        output = weight.clone()
        length = weight.numel()
        group = int(length/M)

        weight_temp = weight.detach().abs().reshape(group, M)
        index = torch.argsort(weight_temp, dim=1)[:, :int(M-N)]

        w_b = torch.ones(weight_temp.shape, device=weight_temp.device, dtype=weight.dtype)
        w_b = w_b.scatter_(dim=1, index=index, value=0).reshape(weight.shape)
        ctx.mask = w_b
        ctx.decay = decay

        return output*w_b


    @staticmethod
    def backward(ctx, grad_output):
        # The STE ignores the derivative of the threshold function and passes on the incoming gradient
        # as if the function was an identity function
        weight, = ctx.saved_tensors
        return grad_output + ctx.decay * (1-ctx.mask) * weight, None, None


class SparseV2(autograd.Function):
    """" Prune the unimprotant weight for the forwards phase but pass the gradient to dense weight using SR-STE in the backwards phase"""

    @staticmethod
    def forward(ctx, weight, N, M, decay = 0.0002):
        ctx.save_for_backward(weight)
        if N == 2 and M == 4:
            _, indices = F.max_pool1d(weight.abs(), kernel_size=4, stride=4, return_indices=True)
            base = torch.empty_like(weight).fill_(0.)
            base = base.scatter_(1, indices, 1)
            weight_ = weight.scatter(1, indices, 0)
            _, indices = F.max_pool1d(weight_.abs(), kernel_size=4, stride=4, return_indices=True)
            mask = base.scatter_(1, indices, 1).squeeze_()
        
        ctx.mask = mask
        ctx.decay = decay
        
        return mask * weight
    
    @staticmethod
    def backward(ctx, grad_output):

        weight, = ctx.saved_tensors
        return grad_output + ctx.decay * (1-ctx.mask) * weight, None, None
    

class SparseV3(autograd.Function):
    """" Prune the unimprotant weight for the forwards phase but pass the gradient to dense weight using SR-STE in the backwards phase"""
    """ Customized CUDA kernels are involved"""

    @staticmethod
    def forward(ctx, weight, N, M, decay = 0.0002):
        ctx.save_for_backward(weight)
        assert N == 2
        assert M == 4

        nonzeros, metadata = bdense2sparse(weight, True)

        ctx.meta = metadata
        ctx.decay = decay

        return nonzeros, metadata, weight # the third parameter weight is simply used to bypass gradient shape checking
    
    @staticmethod
    def backward(ctx, grad_nonzeros, grad_metadata, grad_weight):
        mask_nnz = torch.ones(size=(grad_weight.size(0), int(grad_weight.size(1)/2)), dtype=grad_weight.dtype, device="cuda")
        mask = spmmv2_bf16(mask_nnz, torch.eye(grad_weight.size(1), dtype=grad_weight.dtype, device="cuda"), ctx.meta)

        weight, = ctx.saved_tensors
        return grad_weight + ctx.decay * (1-mask) * weight, None, None



class Sparse_NHWC(autograd.Function):
    """" Prune the unimprotant edges for the forwards phase but pass the gradient to dense weight using STE in the backwards phase"""

    @staticmethod
    def forward(ctx, weight, N, M, decay = 0.0002):

        ctx.save_for_backward(weight)
        output = weight.clone()
        length = weight.numel()
        group = int(length/M)

        weight_temp = weight.detach().abs().permute(0,2,3,1).reshape(group, M)
        index = torch.argsort(weight_temp, dim=1)[:, :int(M-N)]

        w_b = torch.ones(weight_temp.shape, device=weight_temp.device)
        w_b = w_b.scatter_(dim=1, index=index, value=0).reshape(weight.permute(0,2,3,1).shape)
        w_b = w_b.permute(0,3,1,2)

        ctx.mask = w_b
        ctx.decay = decay

        return output*w_b

    @staticmethod
    def backward(ctx, grad_output):

        weight, = ctx.saved_tensors
        return grad_output + ctx.decay * (1-ctx.mask) * weight, None, None


class SparseConv(nn.Conv2d):
    """" implement N:M sparse convolution layer """
    
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, groups=1, bias=True, padding_mode='zeros', N=2, M=4, **kwargs):
        self.N = N
        self.M = M
        super(SparseConv, self).__init__(in_channels, out_channels, kernel_size, stride, padding, dilation, groups, bias, padding_mode, **kwargs)


    def get_sparse_weights(self):

        return Sparse_NHWC.apply(self.weight, self.N, self.M)



    def forward(self, x):

        w = self.get_sparse_weights()
        x = F.conv2d(
            x, w, self.bias, self.stride, self.padding, self.dilation, self.groups
        )
        return x



class SparseLinear(nn.Linear):

    def __init__(self, in_features: int, out_features: int, bias: bool = True, N=2, M=2, decay = 0.0002, **kwargs):
        self.N = N
        self.M = M
        super(SparseLinear, self).__init__(in_features, out_features, bias = bias)


    def get_sparse_weights(self):

        return Sparse.apply(self.weight, self.N, self.M)



    def forward(self, x):
        with nvtx.annotate("DP"):
            w = self.get_sparse_weights()
        with nvtx.annotate("Linear"):
            x = F.linear(x, w, self.bias)
        return x


class LinearV2(autograd.Function):
    """Implement the linear forward function"""

    @staticmethod
    def forward(ctx, input, weight, bias):
        ctx.save_for_backward(input, weight, bias)
        if bias is not None:
            bias = bias.unsqueeze(0)
        if bias is not None:
            return F.linear(input, weight, bias)
        else:
            return torch.matmul(input, weight.t())
    
    @staticmethod
    def backward(ctx, grad_out):
        input, weight, bias = ctx.saved_tensors
        grad_input = torch.matmul(grad_out, weight)
        grad_weight = torch.matmul(grad_out.t(), input)
        if bias is not None:
            grad_bias = torch.sum(grad_out, dim=0).view(bias.size())
        else:
            grad_bias = None
        
        return grad_input, grad_weight, grad_bias


class LinearV3(autograd.Function):
    """Implement the linear forward function with sparsity"""
    
    @staticmethod
    def forward(ctx, input, sp_weight, sp_meta, weight, bias):
        eye = torch.eye(n=sp_weight.size(1) * 2, dtype=sp_weight.dtype, device="cuda")
        ctx.save_for_backward(input, sp_weight, sp_meta, bias)
        if bias is not None:
            bias = bias.unsqueeze(0)
        # output = spmmv2_bf16(sp_weight, input.t().contiguous(), sp_meta).t()
        output = spmmv2_bf16_nt(sp_weight, input, sp_meta).t()
        if bias is not None:
            return output + bias
        else:
            return output
    
    @staticmethod
    def backward(ctx, grad_out):
        input, sp_weight, sp_meta, bias = ctx.saved_tensors
        weight = spmmv2_bf16(sp_weight, torch.eye(n=sp_weight.size(1) * 2, dtype=sp_weight.dtype, device="cuda"), sp_meta)
        grad_input = torch.matmul(grad_out, weight)
        grad_weight = torch.matmul(grad_out.t(), input)
        if bias is not None:
            grad_bias = torch.sum(grad_out, dim=0).view(bias.size())
        else:
            grad_bias = None
        
        return grad_input, None, None, grad_weight, grad_bias


class SparseLinearV2(nn.Linear):
    def __init__(self, in_features: int, out_features: int, bias: bool = True, N=2, M=4, decay = 0.0002, **kwargs):
        self.N = N
        self.M = M
        super(SparseLinearV2, self).__init__(in_features, out_features, bias = bias)

        self.sp_weight = self.get_sparse_weights()

    def get_sparse_weights(self):
        return SparseV2.apply(self.weight, self.N, self.M)

    def forward(self, x):
        with nvtx.annotate("DP"):
            w = self.get_sparse_weights()
        with nvtx.annotate("Linear"):
            x = LinearV2.apply(x, w, self.bias)
        return x


class SparseLinearV3(nn.Linear):
    def __init__(self, in_features: int, out_features: int, bias: bool = True, N=2, M=4, decay = 0.0002, **kwargs):
        super(SparseLinearV3, self).__init__(in_features, out_features, bias)
        self.N = N
        self.M = M
        
        # self.sp_weight, self.sp_meta = self.get_sparse_weights()
    
    def get_sparse_weights(self):
        return SparseV3.apply(self.weight, self.N, self.M)
    
    def forward(self, x):
        sp_weight, sp_meta, weight = self.get_sparse_weights()
        x = LinearV3.apply(x, sp_weight, sp_meta, weight, self.bias)

        return x