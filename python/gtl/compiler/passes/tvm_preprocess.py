import torch
from gtl.compiler.nodes import *
import operator

def pass_tvm_preprocessing(module, graph):
    for node in graph.nodes:
        if node.op == "call_function":
            if node.target == torch.ops.aten._softmax:
                node.target = torch.ops.aten.softmax
                node.args = [node.args[0], node.args[1]]
            elif node.target == torch.ops.aten.convolution:
                node.target = torch.ops.aten._convolution
                node.args = node.args + (True, True, True)
            elif node.target == torch.ops.aten.native_batch_norm:
                node.target = torch.ops.aten.batch_norm
                node.args = node.args + (True,)