################################################################################
# This file contains helper functions to inject different kind of node to graph
# It also infers the data type and shapes without actually running the code
################################################################################
import torch
import torch.fx as fx
from torch.fx.passes.shape_prop import TensorMetadata

def node_equal(node1, node2):
    if node1.op != node2.op:
        return False
    if node1.target != node2.target:
        return False
    if node1.args != node2.args:
        return False
    return True

def inject_get_attr(inject_point, module, graph, tensor, tensor_name):
    # update injection point to maintain topological order
    graph.inserting_after(inject_point)
    # register the tensor in the module
    module.register_buffer(tensor_name, tensor)
    # create get attribute node
    attr_node = graph.get_attr(tensor_name)
    attr_node.meta = {}
    attr_node.meta['tensor_meta'] = TensorMetadata(
                shape=tensor.shape, dtype=tensor.dtype, requires_grad=False, 
                stride=(1,), memory_format=torch.contiguous_format, 
                is_quantized=False, qparams={})
    return attr_node


def inject_softmax(inject_point, graph, parent_node, dim, half_to_float=False, tmp_node=None):
    if tmp_node is None: tmp_node = parent_node
    graph.inserting_after(inject_point)
    softmax_node = graph.call_function(torch.ops.aten._softmax, args=(tmp_node, dim, half_to_float))
    softmax_node.meta = {}
    if half_to_float:
        softmax_node.meta['tensor_meta'] = parent_node.meta['tensor_meta']._replace(dtype=torch.float32)
    else:
        softmax_node.meta['tensor_meta'] = parent_node.meta['tensor_meta']._replace()
    return softmax_node

def inject_log(inject_point, graph, parent_node, tmp_node=None):
    if tmp_node is None: tmp_node = parent_node
    graph.inserting_after(inject_point)
    log_node = graph.call_function(torch.ops.aten.log, args=(tmp_node,))
    log_node.meta = {}
    log_node.meta['tensor_meta'] = parent_node.meta['tensor_meta']._replace()
    return log_node

def inject_onehot(inject_point, graph, num_classes, parent_node, tmp_node=None):
    if tmp_node is None: tmp_node = parent_node
    graph.inserting_after(inject_point)
    one_hot_node = graph.call_function(torch.ops.aten.one_hot, args=(tmp_node,), kwargs={"num_classes": num_classes})
    one_hot_node.meta = {}
    shape = list(parent_node.meta['tensor_meta'].shape)
    shape.append(num_classes)
    one_hot_node.meta['tensor_meta'] = parent_node.meta['tensor_meta']._replace(shape=shape)
    return one_hot_node

def inject_neg(inject_point, graph, parent_node, tmp_node=None):
    if tmp_node is None: tmp_node = parent_node
    graph.inserting_after(inject_point)
    neg_node = graph.call_function(torch.ops.aten.neg, args=(tmp_node,))
    neg_node.meta = {}
    neg_node.meta['tensor_meta'] = parent_node.meta['tensor_meta']._replace()
    return neg_node

def get_shape(node):
    if isinstance(node, fx.Node):
        return node.meta['tensor_meta'].shape
    else:
        return ()
    
def get_broadcast_shape(lhs, rhs):
    if isinstance(lhs, fx.Node):
        lhs_shape = lhs.meta['tensor_meta'].shape
    else:
        lhs_shape = ()
    if isinstance(rhs, fx.Node):
        rhs_shape = rhs.meta['tensor_meta'].shape
    else:
        rhs_shape = ()
    if lhs_shape == () and rhs_shape == ():
        shape = ()
    elif lhs_shape == () and rhs_shape != ():
        shape = rhs_shape
    elif rhs_shape == () and lhs_shape != ():
        shape = lhs_shape
    else:
        if len(lhs_shape) == len(rhs_shape):
            shape = tuple([max(l, r) for l, r in zip(list(lhs_shape), list(rhs_shape))])
        else:
            if len(lhs_shape) < len(rhs_shape):
                short = list(lhs_shape)
                long = list(rhs_shape)
            else:
                short = list(rhs_shape)
                long = list(lhs_shape)
            
            shape = []
            long_idx = 0
            for s in short:
                if s == 1:
                    shape.append(long[long_idx])
                    long_idx += 1
                else:
                    while (s != long[long_idx] and long[long_idx] != 1 and long_idx < len(long)):
                        shape.append(long[long_idx])
                        long_idx += 1
                    if s == long[long_idx]:
                        shape.append(long[long_idx])
                    elif long_idx == 1:
                        shape.append(s)
                    else:
                        raise NotImplementedError()
                    long_idx += 1
    
    return shape

def get_auto_type_conversion(lhs, rhs):
    if isinstance(lhs, fx.Node):
        lhs_dtype = lhs.meta['tensor_meta'].dtype
    else:
        lhs_dtype = None
    if isinstance(rhs, fx.Node):
        rhs_dtype = rhs.meta['tensor_meta'].dtype
    else:
        rhs_dtype = None
    if lhs_dtype is None and rhs_dtype is None:
        dtype = None
    elif lhs_dtype is None and rhs_dtype is not None:
        dtype = rhs_dtype
    elif rhs_dtype is None and lhs_dtype is not None:
        dtype = lhs_dtype
    else:
        if lhs_dtype in [torch.int64, torch.int32, torch.int16]:
            dtype = rhs_dtype
        else:
            dtype = lhs_dtype
    
    return dtype

def inject_mul(inject_point, graph, lhs, rhs, tmp_lhs=None, tmp_rhs=None):
    if tmp_lhs is None: tmp_lhs = lhs
    if tmp_rhs is None: tmp_rhs = rhs

    graph.inserting_after(inject_point)
    mul_node = graph.call_function(torch.ops.aten.mul, args=(tmp_lhs, tmp_rhs))
    shape = get_broadcast_shape(lhs, rhs)
    dtype = get_auto_type_conversion(lhs, rhs)
    mul_node.meta = {}
    mul_node.meta['tensor_meta'] = inject_point.meta['tensor_meta']._replace(shape=shape, dtype=dtype)
    return mul_node

def inject_div(inject_point, graph, lhs, rhs, tmp_lhs=None, tmp_rhs=None):
    if tmp_lhs is None: tmp_lhs = lhs
    if tmp_rhs is None: tmp_rhs = rhs

    graph.inserting_after(inject_point)
    mul_node = graph.call_function(torch.ops.aten.div, args=(tmp_lhs, tmp_rhs))
    shape = get_broadcast_shape(lhs, rhs)
    dtype = get_auto_type_conversion(lhs, rhs)
    mul_node.meta = {}
    mul_node.meta['tensor_meta'] = inject_point.meta['tensor_meta']._replace(shape=shape, dtype=dtype)
    return mul_node

def inject_ne(inject_point, graph, lhs, rhs, tmp_lhs=None, tmp_rhs=None):
    if tmp_lhs is None: tmp_lhs = lhs
    if tmp_rhs is None: tmp_rhs = rhs

    graph.inserting_after(inject_point)
    ne_node = graph.call_function(torch.ops.aten.ne, args=(tmp_lhs, tmp_rhs))
    shape = get_broadcast_shape(lhs, rhs)
    ne_node.meta = {}
    ne_node.meta['tensor_meta'] = inject_point.meta['tensor_meta']._replace(shape=tuple(shape), dtype=torch.bool)
    return ne_node

def inject_unsqueeze(inject_point, graph, parent_node, dim, tmp_node=None):
    if tmp_node is None: tmp_node = parent_node

    graph.inserting_after(inject_point)
    unsqueeze_node = graph.call_function(torch.ops.aten.unsqueeze, args=(tmp_node, dim))
    
    # get shape
    shape = list(parent_node.meta['tensor_meta'].shape)
    if dim < 0:
        dim = dim + len(shape) + 1
    shape.insert(dim, 1)
    unsqueeze_node.meta = {}
    unsqueeze_node.meta['tensor_meta'] = parent_node.meta['tensor_meta']._replace(
        shape=shape
    )
    
    return unsqueeze_node

def inject_sum(inject_point, graph, parent_node, dim, tmp_node=None):
    if tmp_node is None: tmp_node = parent_node

    graph.inserting_after(inject_point)
    if isinstance(dim, int):
        dim = [dim,]
    sum_node = graph.call_function(torch.ops.aten.sum, args=(tmp_node, dim))
    sum_node.meta = {}
    shape = list(parent_node.meta['tensor_meta'].shape)
    for d in dim:
        if d < 0:
            shape.pop(d + len(shape) + 1)
        else:
            shape.pop(d)
    sum_node.meta['tensor_meta'] = parent_node.meta['tensor_meta']._replace(
        shape=tuple(shape)
    )
    
    return sum_node

def inject_mean(inject_point, graph, parent_node, dim, tmp_node=None):
    if tmp_node is None: tmp_node = parent_node

    graph.inserting_after(inject_point)
    mean_node = graph.call_function(torch.ops.aten.mean, args=(tmp_node, [dim,], True))
    mean_node.meta = {}
    shape = list(parent_node.meta['tensor_meta'].shape)
    if dim < 0:
        dim = dim + len(shape)
    print(dim)
    print(shape)
    shape.pop(dim)
    mean_node.meta['tensor_meta'] = parent_node.meta['tensor_meta']._replace(
        shape=tuple(shape)
    )
    
    return mean_node

def inject_std(inject_point, graph, parent_node, dim, tmp_node=None):
    if tmp_node is None: tmp_node = parent_node

    graph.inserting_after(inject_point)
    std_node = graph.call_function(torch.ops.aten.std, args=(tmp_node, [dim,], True))
    std_node.meta = {}
    shape = list(parent_node.meta['tensor_meta'].shape)
    if dim < 0:
        dim = dim + len(shape)
    shape.pop(dim)
    std_node.meta['tensor_meta'] = parent_node.meta['tensor_meta']._replace(
        shape=tuple(shape)
    )
    
    return std_node

def inject_sub(inject_point, graph, lhs, rhs, tmp_lhs=None, tmp_rhs=None):
    if tmp_lhs is None: tmp_lhs = lhs
    if tmp_rhs is None: tmp_rhs = rhs

    graph.inserting_after(inject_point)
    sub_node = graph.call_function(torch.ops.aten.sub, args=(tmp_lhs, tmp_rhs))
    shape = get_broadcast_shape(lhs, rhs)
    dtype = get_auto_type_conversion(lhs, rhs)
    sub_node.meta = {}
    sub_node.meta['tensor_meta'] = inject_point.meta['tensor_meta']._replace(shape=shape, dtype=dtype)
    return sub_node

def inject_add(inject_point, graph, lhs, rhs, tmp_lhs=None, tmp_rhs=None):
    if tmp_lhs is None: tmp_lhs = lhs
    if tmp_rhs is None: tmp_rhs = rhs

    graph.inserting_after(inject_point)
    add_node = graph.call_function(torch.ops.aten.add, args=(tmp_lhs, tmp_rhs))
    shape = get_broadcast_shape(lhs, rhs)
    dtype = get_auto_type_conversion(lhs, rhs)
    add_node.meta = {}
    add_node.meta['tensor_meta'] = inject_point.meta['tensor_meta']._replace(shape=shape, dtype=dtype)
    return add_node

def inject_mm(inject_point, graph, lhs, rhs, tmp_lhs=None, tmp_rhs=None):
    if tmp_lhs is None: tmp_lhs = lhs
    if tmp_rhs is None: tmp_rhs = rhs

    graph.inserting_after(inject_point)
    mm_node = graph.call_function(torch.ops.aten.mm, args=(tmp_lhs, tmp_rhs))
    shape = (lhs.meta['tensor_meta'].shape[0], rhs.meta['tensor_meta'].shape[1])
    dtype = get_auto_type_conversion(lhs, rhs)
    mm_node.meta = {}
    mm_node.meta['tensor_meta'] = inject_point.meta['tensor_meta']._replace(shape=shape, dtype=dtype)
    return mm_node

def inject_permute(inject_point, graph, input, dims, tmp_input=None):
    if tmp_input is None: tmp_input = input

    graph.inserting_after(inject_point)
    permute_node = graph.call_function(torch.ops.aten.permute, args=(input, dims))
    shape = [input.meta['tensor_meta'].shape[i] for i in dims]
    permute_node.meta = {}
    permute_node.meta['tensor_meta'] = inject_point.meta['tensor_meta']._replace(shape=shape)
    return permute_node