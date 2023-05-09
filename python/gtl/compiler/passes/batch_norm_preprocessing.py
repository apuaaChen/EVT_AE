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
from gtl.compiler.nodes import *
import logging


################################################################################
# Graph-level pass to preprocess layer norm for kernel fusion
################################################################################
def pass_batchnorm_preprocessing(module, graph):
    batch_norm_idx = 0
    batch_norm_backward_idx = 0
    for node in graph.nodes:
        if node.op == "call_function":
            if node.target == torch.ops.aten._native_batch_norm_legit.no_stats:
                if "splitted" in node.meta.keys(): continue
                # if batch_norm_idx >=4: continue
                logging.debug(f"====================={node}======================")

                ################################################################
                # inject the epilogue nodes for reduction fusion
                input = node.args[0]
                N, K, P, Q = list(input.meta["tensor_meta"].shape)
                reduction_length = N * P * Q

                # div_node = inject_mul(input, graph, input, 1./reduction_length)
                div_node = inject_div(input, graph, input, reduction_length)
                mean_x_node = inject_sum(div_node, graph, div_node, [0, 2, 3], dtype=torch.float32)
                mul_node = inject_mul(div_node, graph, div_node, input)
                mean_x2_node = inject_sum(mul_node, graph, mul_node, [0, 2, 3], dtype=torch.float32)

                graph.inserting_after(node)
                bn_splitted = graph.call_function(torch.ops.aten._native_batch_norm_legit.no_stats, args=(*node.args, mean_x_node, mean_x2_node))
                bn_splitted.meta["splitted"] = True

                # bn_splitted.meta["tensor_meta"] = node.meta["tensor_meta"]._replace()
                node.replace_all_uses_with(bn_splitted)
                graph.erase_node(node)

                # update the meta data of output saved mean & std
                # print(list(bn_splitted.users.keys()))
                # _, saved_mean, saved_invstd = list(bn_splitted.users.keys())
                saved_mean = list(bn_splitted.users.keys())[1]
                saved_invstd = list(bn_splitted.users.keys())[2]

                saved_mean.meta["tensor_meta"] = saved_mean.meta["tensor_meta"]._replace(shape=(1, K, 1, 1))
                saved_invstd.meta["tensor_meta"] = saved_invstd.meta["tensor_meta"]._replace(shape=(1, K, 1, 1))
                
                batch_norm_idx += 1
            # for batch norm backward
            elif node.target == torch.ops.aten.native_batch_norm_backward:
                logging.debug(f"====================={node}======================")
                # if batch_norm_backward_idx >= 0: continue

                # inject the epilogue nodes for reduction fusion
                y_grad, input, gamma, running_mean, running_var, saved_mean, saved_invstd, is_training, epsilon, input_mask = node.args
                N, C, H, W = list(y_grad.meta["tensor_meta"].shape)
                reduction_length = N * H * W

                sum_grad_y_node = inject_sum(y_grad, graph, y_grad, [0, 2, 3], dtype=torch.float32)
                # get x_hat
                sub_mean_node = inject_sub(sum_grad_y_node, graph, input, saved_mean)
                mul_invstd_node = inject_mul(sub_mean_node, graph, sub_mean_node, saved_invstd)
                # add a cast node to project x_hat to float16, as the code above could be fused into the preceeding kernel
                cast_node = inject_dtype_cast(mul_invstd_node, graph, mul_invstd_node, torch.float16)
                mul_xhat_node = inject_mul(cast_node, graph, y_grad, cast_node)
                sum_grad_y_xhat_node = inject_sum(mul_xhat_node, graph, mul_xhat_node, [0, 2, 3], dtype=torch.float32)

                # update the args of the batch_norm_backward node
                node.args = (y_grad, input, gamma, running_mean, running_var, saved_mean, saved_invstd, is_training, epsilon, sum_grad_y_node, sum_grad_y_xhat_node, float(reduction_length))

                if "tensor_meta" in node.meta.keys():
                    node.meta["tensor_meta"] = node.meta["tensor_meta"][0]
                else:
                    node.meta["tensor_meta"] = input.meta["tensor_meta"]._replace()
                batch_norm_backward_idx += 1
    
    graph.lint()
    graph.eliminate_dead_code()