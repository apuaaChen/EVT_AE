################################################################################
# Copyright [yyyy] [name of copyright owner]
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
################################################################################
import torch
from nodes import *

################################################################################
# Graph-level pass to break down log softmax, nll_loss_forward, nll_loss_backward, 
# _log_softmax_backward_data
################################################################################
# TODO: the general substitution doesn't work for these cases

class nhwcWgradOnly:
    __name__ = "torch_nhwc_wgrad"
    def __init__(self) -> None:
        self.stream = None

    def __call__(self, grad_output, input, weight, bias_sizes_opt, stride, padding, dilation, transposed, output_padding, groups, output_mask):
        default_stream = torch.cuda.current_stream()
        if self.stream is not None:
            self.stream.wait_stream(default_stream)
            stream = self.stream
        else:
            stream = torch.cuda.current_stream()
        with torch.cuda.stream(stream):
            k, c, s, r = weight.size()
            weight_permuted = weight.view(k, r, s, c).permute(0, 3, 1, 2)
            weight_grad = torch.ops.aten.convolution_backward(grad_output, input, weight_permuted, bias_sizes_opt, stride, padding, dilation, transposed, output_padding, groups, output_mask)[1]
            weight_grad = weight_grad.permute(0, 2, 3, 1).view(k, c, r, s)

            if self.stream is not None:
                grad_output.record_stream(self.stream)
                input.record_stream(self.stream)
        return weight_grad

def pass_composed_op_breakdown(module, graph, disabled_list=[]):
    softmax_node = None
    for node in graph.nodes:
        if node.op == "call_function":
            if node.target in disabled_list: continue
            if node.target == torch.ops.aten._log_softmax:
                # break it down into softmax and log
                softmax_node = inject_softmax(
                    node, graph, node.args[0], node.args[1], node.args[2])
                log_node = inject_log(softmax_node, graph, softmax_node)
                node.replace_all_uses_with(log_node)
            elif node.target == torch.ops.aten.nll_loss_backward:
                if node.args[4] == 1:
                    reduction = "mean"
                elif node.args[4] == 2:
                    reduction = "sum"
                label_node = node.args[2]
                one_hot_node = inject_onehot(node, graph, node.meta["tensor_meta"].shape[1], label_node)
                neg_node = inject_neg(one_hot_node, graph, one_hot_node)
                mul_node = inject_mul(neg_node, graph, neg_node, node.args[0])
                # if node.args[5] >= 0:
                ne_node = inject_ne(mul_node, graph, label_node, node.args[5])
                unsqueeze_node = inject_unsqueeze(ne_node, graph, ne_node, 1)
                mul_mask_node = inject_mul(unsqueeze_node, graph, mul_node, unsqueeze_node)
                if reduction == "mean":
                    # raise NotImplementedError()
                    # sum_node = inject_sum(ne_node, graph, ne_node, dim=0)
                    # TODO: it actually should be sum of not masked nodes
                    div_node = inject_div(mul_mask_node, graph, mul_mask_node, mul_mask_node.meta["tensor_meta"].shape[0])
                    node.replace_all_uses_with(div_node)
                else:
                    node.replace_all_uses_with(mul_mask_node)
                # else:
                #     if reduction == "mean":
                #         div_node = inject_div(mul_node, graph, mul_node, node.meta["tensor_meta"].shape[0])
                #         node.replace_all_uses_with(div_node)
                #     else:
                #         node.replace_all_uses_with(mul_node)
            elif node.target == torch.ops.aten._log_softmax_backward_data:
                softmax_node = node.args[1].args[0]
                sum_node = inject_sum(node, graph, node.args[0], 1)
                unsqueeze_node = inject_unsqueeze(sum_node, graph, sum_node, 1)
                mul_node = inject_mul(unsqueeze_node, graph, unsqueeze_node, softmax_node)
                sub_node = inject_sub(mul_node, graph, node.args[0], mul_node)
                node.replace_all_uses_with(sub_node)
            elif node.target == torch.ops.aten.native_dropout_backward:
                grad_Y = node.args[0]
                mask = node.args[1]
                mul_node = inject_mul(node, graph, grad_Y, mask)
                mul_node2 = inject_mul(mul_node, graph, mul_node, node.args[2])
                node.replace_all_uses_with(mul_node2)
            elif node.target == torch.ops.aten.threshold_backward:
                grad_Y = node.args[0]
                threshold_output = node.args[1]
                threshold = node.args[2]

                ne_node = inject_ne(node, graph, threshold_output, threshold)
                mul_node = inject_mul(ne_node, graph, grad_Y, ne_node)
                node.replace_all_uses_with(mul_node)
            elif node.target == torch.ops.aten.convolution_backward:
                grad_output, input, weight, bias_sizes_opt, stride, padding, dilation, transposed, output_padding, groups, output_mask = node.args
                n, c, h, w = input.meta["tensor_meta"].shape
                k, c_, r, s = weight.meta["tensor_meta"].shape
                assert c == c_
                # inject dgrad node 
                if output_mask[0] and output_mask[1]:
                    dgrad_output = list(node.users.keys())[0]
                    wgrad_output = list(node.users.keys())[1]
                elif (not output_mask[0]) and output_mask[1]:
                    dgrad_output = None
                    wgrad_output = list(node.users.keys())[0]
                elif output_mask[0] and (not output_mask[1]):
                    dgrad_output = list(node.users.keys())[0]
                    wgrad_output = None
                else:
                    raise NotImplementedError
                
                if dgrad_output is not None:
                    dgrad_args = [
                        (n, c, h, w), weight, grad_output, stride, padding, dilation, groups
                    ]
                    graph.inserting_after(node)
                    dgrad_node = graph.call_function(torch.nn.grad.conv2d_input, args=tuple(dgrad_args))
                    dgrad_node.meta["tensor_meta"] = dgrad_output.meta["tensor_meta"]._replace()
                    dgrad_output.replace_all_uses_with(dgrad_node)
                
                if wgrad_output is not None:
                    if dgrad_output is not None:
                        # wgrad_args = [
                        #     input, (k, c, r, s), grad_output, stride, padding, dilation, groups
                        # ]
                        # graph.inserting_after(node)
                        # wgrad_node = graph.call_function(torch.nn.grad.conv2d_weight, args=tuple(wgrad_args))
                        # wgrad_node.meta["tensor_meta"] = wgrad_output.meta["tensor_meta"]._replace()
                        # wgrad_output.replace_all_uses_with(wgrad_node)
                        
                        # We use the pytorch naive wgrad kernel for now
                        # As there is no fusion requirement in the wgrad kernel
                        # And seems that pytorch's implementation runs faster
                        # Uncomment above if pycutlass implementation is prefered
                        # def nhwc_wgrad_only(grad_output, input, weight, bias_sizes_opt, stride, padding, dilation, transposed, output_padding, groups, output_mask):
                        #     k, c, s, r = weight.size()
                        #     weight_permuted = weight.view(k, r, s, c).permute(0, 3, 1, 2)
                        #     weight_grad = torch.ops.aten.convolution_backward(grad_output, input, weight_permuted, bias_sizes_opt, stride, padding, dilation, transposed, output_padding, groups, output_mask)[1]
                        #     weight_grad = weight_grad.permute(0, 2, 3, 1).view(k, c, r, s)
                        #     return weight_grad
                        
                        torch_wgrad_func = nhwcWgradOnly()

                        graph.inserting_after(node)
                        wgrad_node = graph.call_function(torch_wgrad_func, args=(grad_output, input, weight, bias_sizes_opt, stride, padding, dilation, transposed, output_padding, groups, [False, True, False]))
                        wgrad_node.meta["tensor_meta"] = wgrad_output.meta["tensor_meta"]._replace()
                        wgrad_output.replace_all_uses_with(wgrad_node)
                    else:
                        # transfer the function to nhwc
                        def nhwc_wgrad_only(grad_output, input, weight, bias_sizes_opt, stride, padding, dilation, transposed, output_padding, groups, output_mask):
                            k, c, s, r = weight.size()
                            weight_permuted = weight.view(k, r, s, c).permute(0, 3, 1, 2)
                            weight_grad = torch.ops.aten.convolution_backward(grad_output, input, weight_permuted, bias_sizes_opt, stride, padding, dilation, transposed, output_padding, groups, output_mask)[1]
                            weight_grad = weight_grad.permute(0, 2, 3, 1).view(k, c, r, s)
                            return [None, weight_grad, None]
                        
                        node.target = nhwc_wgrad_only
                
    graph.eliminate_dead_code()
    graph.lint()