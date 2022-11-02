import enum
import math
from copy import copy
import torch


class IterVar:
    def __init__(self, name, extend):
        self.name = name
        self.extend = extend
    
    def __str__(self):
        return "%s(%d)" % (self.name, self.extend)


class IterFactor:
    def __init__(self, iter_var, factor, mod, stride):
        """
        iter_var: the iterator variable
        factor: the factor multiplied to the iterator variable
        mod: the modular of the iterator factor
        stride: the stride appiled to the current iter factor
        """
        self.iter_var = iter_var
        self.factor = factor
        self.stride = stride
        self.mod = mod
    
    def __str__(self):
        if self.factor >= 1:
            return "(%s x %d) mod %d" % (self.iter_var.name, self.factor, self.mod)
        else:
            return "(%s / %d) mod %d" % (self.iter_var.name, 1/self.factor, self.mod)


class Tensor:
    def __init__(self, iter_vars=None, dims=None) -> None:
        # inject 
        if iter_vars is not None and dims is None:
            self.dims = []
            stride = 1
            for iter_var in reversed(iter_vars):
                self.dims.append(
                    [IterFactor(
                        iter_var=iter_var, factor=1, 
                        mod=iter_var.extend, stride=stride),]
                )
                stride *= iter_var.extend
            self.dims.reverse()
        elif dims is not None and iter_vars is None:
            self.dims = dims
        else:
            raise NotImplementedError()
    
    def __str__(self):
        iter_str = ""
        for dim in self.dims:
            tmp = ""
            for iter_factor in dim:
                tmp += iter_factor.__str__()
            iter_str += tmp + ", "
        iter_str = iter_str[:-2]
        tensor_str = "Tensor[%s]" % iter_str
        return tensor_str
    
    def view(self, new_shape) -> None:
        new_strides = []
        stride = 1
        for dim in reversed(new_shape):
            new_strides.append(stride)
            stride *= dim
        new_strides.reverse()

        # get new dims
        new_dims = []

        for stride, extend in zip(new_strides, new_shape):
            new_dim = []
            for dim in self.dims:
                for iter_factor in dim:
                    if iter_factor.factor < 1:
                        raise NotImplementedError()
                    factor = iter_factor.factor * iter_factor.stride / stride
                    mod = extend
                    # check 1: if iter_var / factor is always < 1
                    if factor * iter_factor.iter_var.extend <= 1:
                        continue
                    # check 2: if factor % mod == 0
                    if factor % mod == 0:
                        continue
                    new_dim.append(
                        IterFactor(
                            iter_var=iter_factor.iter_var,
                            factor=factor,
                            mod=mod, stride=stride)
                    )
            if len(new_dim) > 1:
                raise NotImplementedError("currently do not support multiple iterator in the same dimensionn")
            new_dims.append(new_dim)
        new_tensor = Tensor(dims=new_dims)
        return new_tensor
    
    def debroadcast(self, new_shape) -> 'Tensor':
        new_tensor_dims = []
        new_stride = 1
        for dim, shape in zip(reversed(self.dims), reversed(new_shape)):
            if len(dim) > 1:
                raise NotImplementedError
            if dim[0].mod % shape == 0:
                new_tensor_dims.append(
                    [IterFactor(iter_var=dim[0].iter_var, factor=dim[0].factor, mod=shape, stride=new_stride)]
                )
                new_stride *= shape
            else:
                raise NotImplementedError
        new_tensor_dims.reverse()
        new_tensor = Tensor(dims=new_tensor_dims)
        return new_tensor
    
    def squeeze(self, squeeze_idx) -> 'Tensor':
        new_tensor_dims = []
        for idx, dim in enumerate(self.dims):
            if len(dim) > 1:
                raise NotImplementedError
            if idx == squeeze_idx:
                if dim[0].mod != 1:
                    raise ValueError("cannot squeeze non-unit dimension")
            else:
                new_tensor_dims.append(
                    [IterFactor(iter_var=dim[0].iter_var, factor=dim[0].factor, mod=dim[0].mod, stride=dim[0].stride)]
                )
        
        new_tensor = Tensor(dims=new_tensor_dims)
        return new_tensor
    
    def get_node_tensor_bottom_up(self, node):
        if node.target in [torch.ops.aten.view, torch.ops.aten._unsafe_view]:
            node.meta["tensor"] = self.view(node.args[1])
        elif node.target in [torch.ops.aten.add, torch.ops.aten.sub, torch.ops.aten.mul, torch.ops.aten.div]:
            for arg in node.args:
                if not 'tensor' in arg.meta.keys():
                    debroadcast_tensor = self.debroadcast(list(arg.meta["tensor_meta"].shape))
                    arg.meta['tensor'] = debroadcast_tensor
            node.meta["tensor"] = copy(self)
        elif node.target in [torch.ops.aten.neg]:
            node.meta["tensor"] = copy(self)
        else:
            raise NotImplementedError("unsupported operator")
    
    def get_node_tensor_top_down(self, node):
        # if the node has 'tensor' in meta data
        if node.target in [torch.ops.aten.add, torch.ops.aten.sub, torch.ops.aten.mul, torch.ops.aten.div]:
            for input in node.all_input_nodes:
                if 'tensor' in input.meta.keys(): continue
                input.meta['tensor'] = self.debroadcast(list(input.meta['tensor_meta'].shape))
        elif node.target in [torch.ops.aten.view, torch.ops.aten._unsafe_view]:
            input = node.args[0]
            if 'tensor' in input.meta.keys(): return
            input.meta['tensor'] = self.view(new_shape=list(input.meta['tensor_meta'].shape))
        elif node.target in [torch.ops.aten._to_copy]:
            input = node.args[0]
            if 'tensor' in input.meta.keys(): return
            input.meta['tensor'] = copy(self)
        elif node.target in [torch.ops.aten.unsqueeze]:
            input = node.args[0]
            if 'tensor' in input.meta.keys(): return
            input.meta['tensor'] = self.squeeze(node.args[1])
        elif node.target in [torch.ops.aten.mm, torch.ops.aten.bmm, torch.ops.aten._softmax]:
            raise NotImplementedError()
        else:
            raise NotImplementedError()






# iter_vars = [
#     IterVar("b", 128),
#     IterVar("m", 512),
#     IterVar("n", 512)
# ]

# tensor = Tensor(iter_vars)
# print(tensor)

# tensor.view([8, 16, 512, 512])
# print(tensor)

# mask = tensor.debroadcast([8, 1, 1, 512])
# print(mask)
# mask_squeeze_1 = mask.squeeze(2)
# mask_squeeze_2 = mask_squeeze_1.squeeze(1)
# print(mask_squeeze_2)
# print("============")
# print(tensor.debroadcast([512]))

# iter_vars = [
#     IterVar("m", 4096),
#     IterVar("n", 1024)
# ]

# tensor = Tensor(iter_vars=iter_vars)
# print(tensor)

# reshaped = tensor.view([512, 8, 1024])
# bias = reshaped.debroadcast([1024,])
# new_reshaped = reshaped.view([512, 128, 64])
# print(reshaped)
# print(bias)