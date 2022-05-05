from torch import autograd, nn
import torch.nn.functional as F
import torch
from torch.nn import init
import math
import nvtx

from sptrain.spmm import spmmv2_f16_nnn, spmmv2_f16_ntt
from sptrain.spmmt import spmmt_f16_ntt
from sptrain.sddmm_meta import sddmm_meta_f16_ntn
from sptrain.meta import bdense2sparse


class Linear(autograd.Function):

    @staticmethod
    def forward(ctx, input, weight, bias):
        ctx.save_for_backward(input, weight, bias)

        if bias is not None:
            bias = bias.unsqueeze(0)
            return F.linear(input, weight, bias)
        else:
            return torch.matmul(input, weight.t())
    
    @staticmethod
    def backward(ctx, grad_outputs):
        input, weight, bias = ctx.saved_tensors
        feat_out, feat_in = weight.size()
        grad_input = torch.matmul(grad_outputs, weight)
        grad_weight = torch.matmul(grad_outputs.view(-1, feat_out).t(), input.view(-1, feat_in))
        if bias is not None:
            grad_bias = torch.sum(grad_outputs.view(-1, feat_out), dim=0).view(bias.size())
        else:
            grad_bias = None
        
        return grad_input, grad_weight, grad_bias



class SpLinearFn(autograd.Function):

    @staticmethod
    def forward(ctx, input, weight, metadata, bias):
        ctx.save_for_backward(input, weight, metadata, bias)
        if bias is not None:
            bias = bias.unsqueeze(0)
        
        feat_out, feat_in = weight.size()
        feat_in *= 2

        output_shape = list(input.size())
        output_shape[-1] = feat_out
        with nvtx.annotate("spmm"):
            output = spmmv2_f16_ntt(weight, input.view(-1, feat_in), metadata, 1.).view(tuple(output_shape))

        with nvtx.annotate("bias"):
            if bias is not None:
                return output + bias
            else:
                return output
    
    @staticmethod
    def backward(ctx, grad_out):
        input, weight, metadata, bias = ctx.saved_tensors
        feat_out, feat_in = weight.size()
        feat_in *= 2
        with nvtx.annotate("grad_input"):
            grad_input = spmmt_f16_ntt(weight, grad_out.view(-1, feat_out), metadata, 1.)
        with nvtx.annotate("grad_weight"):
            grad_weight = torch.matmul(grad_out.view(-1, feat_out).t(), input.view(-1, feat_in)).squeeze_()
            mask = spmmv2_f16_nnn(torch.ones_like(weight), torch.eye(n=feat_in, dtype=torch.float16, device="cuda"), metadata, 1.)
            grad_weight *= mask.squeeze_()
            grad_weight, _ = bdense2sparse(grad_weight, True)

        if bias is not None:
            with nvtx.annotate("grad_sum"):
                grad_bias = torch.sum(grad_out.view(-1, feat_out), dim=0).view(bias.size())
        else:
            grad_bias = None

        return grad_input.view(input.size()), grad_weight.view(weight.size()), None, grad_bias



class SpLinear(nn.Module):
    __constants__ = ['in_features', 'out_features']
    in_features: int
    out_features: int
    weight: torch.Tensor

    def __init__(self, in_features: int, out_features: int, bias: bool = True,
                 device=None, dtype=None) -> None:
        factory_kwargs = {'device': device, 'dtype': dtype}
        super(SpLinear, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = nn.Parameter(torch.empty((out_features, in_features // 2), **factory_kwargs))
        self.metadata = torch.empty(size=(out_features, in_features // 16), dtype=torch.int16, device=device)
        if bias:
            self.bias = nn.Parameter(torch.empty(out_features, **factory_kwargs))
        else:
            self.register_parameter('bias', None)
    
    def get_dense_weight(self) -> torch.Tensor:
        return spmmv2_f16_nnn(self.weight, torch.eye(n=self.in_features, dtype=torch.float16, device="cuda"), self.metadata, 1.)
    
    def get_dense_weight_grad(self) -> torch.Tensor:
        return spmmv2_f16_nnn(self.weight.grad, torch.eye(n=self.in_features, dtype=torch.float16, device="cuda"), self.metadata, 1.)
    
    def reset_parameters(self) -> None:
        # Setting a=sqrt(5) in kaiming_uniform is the same as initializing with
        # uniform(-1/sqrt(in_features), 1/sqrt(in_features)). For details, see
        # https://github.com/pytorch/pytorch/issues/57109
        init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        if self.bias is not None:
            fan_in, _ = init._calculate_fan_in_and_fan_out(self.weight)
            fan_in = fan_in * 2
            bound = 1 / math.sqrt(fan_in) if fan_in > 0 else 0
            init.uniform_(self.bias, -bound, bound)
    
    def set_parameters(self, weight, bias):
        self.bias = nn.Parameter(bias.clone())
        nnz, self.metadata = bdense2sparse(weight.detach().clone(), True)
        self.weight = nn.Parameter(nnz)
    
    def forward(self, input: torch.Tensor) -> torch.Tensor:
        return SpLinearFn.apply(input, self.weight, self.metadata, self.bias)



# class SparseLinear(autograd.Function):

#     @staticmethod
#     def forward(ctx, input, weight_nnz, metadata, bias):

