import torch
import operator
import torch.fx as fx
from torch._functorch.partitioners import _extract_graph_with_inputs_outputs
from typing import List


class NvfuserParser:
    def __init__(self, node) -> None:
        # assert node.target in [
        #     torch.ops.aten.native_batch_norm.default, 
        #     torch.ops.aten.native_batch_norm_backward,
        #     torch.ops.aten.expand]
        self.node = node

        # self.all_new_users = []
        self.fused_nodes = [self.node]
        self.min_topo_id = self.node.meta['topo_idx']
        self.input_nodes = self.node.all_input_nodes
        self.outputs = []
        self.visit_all_users(self.node)
        for node in self.fused_nodes:
            self.visit_all_inputs(node)
        
        for node in self.fused_nodes:
            self.update_sum_nodes(node)
        self.get_all_input_nodes()

        # self.get_all_inputs()
    
    def dfs(self, node):
        if node in self.fused_nodes:
            return True
        else:
            if node.meta['topo_idx'] < self.min_topo_id:
                return False
            else:
                for input in node.all_input_nodes:
                    if self.dfs(input): return True
                return False
                
    
    def fusable(self, node):
        # a node is fusible if and only if it satisfies the following condition
        # * it's target belong's to the fusible list
        # * all its inputs are fusible or already fused or is a tensor input does not depend on the fused nodes
        if node.target in [operator.getitem, torch.ops.aten.relu, torch.ops.aten.add, torch.ops.aten.mean, torch.ops.aten.view, torch.ops.aten.mul, torch.ops.aten.sum, torch.ops.aten.sub, torch.ops.aten.ne, torch.ops.aten.to]:
            input_list = list(node.all_input_nodes)
            for input in input_list:
                if (
                    input in self.fused_nodes or 
                    self.fusable(input) or (not self.dfs(input))):
                    continue
                else:
                    return False
            return True
        else:
            return False

    def visit_all_users(self, node):
        for user in list(node.users.keys()):
            if user not in self.fused_nodes:
                # check 
                if self.fusable(user):
                    self.fused_nodes.append(user)
                    self.min_topo_id = min(self.min_topo_id, user.meta["topo_idx"])
                    self.visit_all_users(user)
                else:
                    if node not in self.outputs:
                        self.outputs.append(node)
    
    def visit_all_inputs(self, node):
        if input in list(node.all_input_nodes):
            if input not in self.fused_nodes:
                # check
                if self.fusable(input):
                    self.fused_nodes.append(input)
                    self.min_topo_id = min(self.min_topo_id, input.meta["topo_idx"])
                    self.visit_all_inputs(input)
    
    def get_all_input_nodes(self):
        for epilogue_node in self.fused_nodes:
            for arg in epilogue_node.args:
                if arg not in self.fused_nodes:
                    if isinstance(arg, fx.Node):
                        if arg not in self.input_nodes:
                            self.input_nodes.append(arg)
    
    def update_sum_nodes(self, node):
        if node.target == torch.ops.aten.sum:
            if node.args[1] == [0, 2, 3]:
                node.args = (node.args[0], [0, 1, 2])
        elif node.target == torch.ops.aten.mean:
            if len(node.meta["tensor_meta"].shape) == 4:
                node_arg = list(node.args)
                if node_arg[1] == [-1, -2]:
                    node_arg[1] = [-2, -3]
                
                node.args = tuple(node_arg)
        elif node.target == torch.ops.aten.expand:
            if len(node.args[1]) == 4:
                new_shape = [node.args[1][i] for i in [0, 2, 3, 1]]
                node.args = (node.args[0], new_shape)
    
    def extract_sub_module(self, module):

        if self.node.target == torch.ops.aten.native_batch_norm.default:
            # replace batch norm 
            def splitted_batch_norm(
                input: torch.Tensor,
                gamma: torch.Tensor,
                beta: torch.Tensor,
                running_mean: torch.Tensor,
                running_var: torch.Tensor,
                is_training: bool,
                momentum: float,
                eps: float, 
                mean: torch.Tensor,
                D: torch.Tensor
            ):
                var = D - mean * mean
                invstd = 1./torch.sqrt(var + eps)
                output = (input - mean) * invstd * gamma + beta
                running_mean = (1-momentum) * running_mean + momentum * mean
                running_var = (1-momentum) * running_var + momentum * var

                return output.to(torch.float16), mean, invstd
            
            self.node.target = splitted_batch_norm

        elif self.node.target == torch.ops.aten.native_batch_norm_backward:
            # replace batch norm backward
            def splitted_batch_norm_backward(
                y_grad: torch.Tensor,
                x: torch.Tensor,
                gamma: torch.Tensor,
                running_mean: torch.Tensor,
                running_var: torch.Tensor,
                saved_mean: torch.Tensor,
                saved_invstd: torch.Tensor,
                is_training: bool,
                epsilon: float,
                grad_y_sum: torch.Tensor,
                grad_y_xhat_sum: torch.Tensor,
                m: float
            ):
                x_hat = (x - saved_mean) * saved_invstd
                mean_grad_mean = grad_y_sum.squeeze() / m
                var_grad_mean = -grad_y_xhat_sum.squeeze() * x_hat * saved_invstd * gamma / m
                x_grad = (y_grad - mean_grad_mean) * gamma * saved_invstd + var_grad_mean

                return x_grad.to(torch.float16), grad_y_xhat_sum.to(torch.float16).squeeze(), grad_y_sum.to(torch.float16).squeeze() 

            self.node.target = splitted_batch_norm_backward
        
        elif self.node.target == torch.ops.aten.max_pool2d_with_indices_backward:
            # replace max pool2d backward
            def channel_last_max_pool2d_backward(
                grad_output: torch.Tensor,
                input: torch.Tensor,
                kernel_size: List[int],
                stride: List[int],
                padding: List[int],
                dilation: List[int],
                ceil_mode: bool,
                indices: torch.Tensor
            ):
                grad_output_nchw = grad_output.permute(0, 3, 1, 2)
                input_nchw = input.permute(0, 3, 1, 2)
                indices_nchw = indices.permute(0, 3, 1, 2)
                grad_input =  torch.ops.aten.max_pool2d_with_indices_backward(
                    grad_output_nchw, input_nchw, kernel_size, stride, padding, dilation, ceil_mode, indices_nchw
                )
                return grad_input.permute(0, 2, 3, 1) # back to nhwc
            
            self.node.target = channel_last_max_pool2d_backward

        

        subgraph = _extract_graph_with_inputs_outputs(module.graph, self.input_nodes, self.outputs)
        fused_module = torch.fx.GraphModule(module, subgraph)

        return torch.jit.script(fused_module)
        # Note: it is quite strange that the torchscript here returns nan
        # in the batch norm forward. uncomment it fix this issue
        # not sure why it happends
        # return fused_module

