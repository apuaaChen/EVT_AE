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


################################################################################
# Graph-level pass to preprocess layer norm for kernel fusion
################################################################################
def pass_layernorm_preprocessing(module, graph):
    for node in graph.nodes:
        if node.op == "call_function":
            if node.target == torch.ops.aten.native_layer_norm:
                # check if the input is permute node
                # step 1: identify the backward kernel
                # the heuristic is that the output mean and std has only one user
                output, mean, std = list(node.users.keys())
                mean.meta["unfusible"] = True
                std.meta["unfusible"] = True
                if len(mean.users.keys()) != 1:
                    raise NotImplementedError("mean node has %d users" % len(mean.users.keys()))
                backward_node = list(mean.users.keys())[0]
                assert backward_node.target == torch.ops.aten.native_layer_norm_backward

                # step 1: inject the gammar and beta nodes
                input = node.args[0]
                gamma = node.args[2]
                beta = node.args[3]
                node.replace_input_with(gamma, None)
                node.replace_input_with(beta, None)
                mul_node = inject_mul(output, graph, output, gamma, tmp_lhs=beta)
                add_node = inject_add(mul_node, graph, mul_node, beta)
                output.replace_all_uses_with(add_node)
                mul_node.replace_input_with(beta, output)

                if "tensor_meta" in node.meta.keys():
                    node.meta["tensor_meta"] = node.meta["tensor_meta"][0]
                else:
                    node.meta["tensor_meta"] = output.meta["tensor_meta"]._replace()

                if backward_node.args[0].target == torch.ops.aten.permute:
                    # step 6: update meta
                    backward_node.meta["unfusible"] = True
                    continue
                
                # step 2: replacing x in backward with x_hat
                # backward_node.replace_input_with(input, output)

                # step 4: get gammar & beta grad with reduction
                grad_input, grad_gamma, grad_beta = list(backward_node.users.keys())
                grad_output = backward_node.args[0]

                grad_output_dim = len(grad_output.meta["tensor_meta"].shape)
                reduction_dims = [i for i in range(grad_output_dim - 1)]
                sum_beta_node = inject_sum(grad_output, graph, grad_output, reduction_dims)

                sub_mean_node = inject_sub(grad_output, graph, input, mean)
                mul_node = inject_mul(sub_mean_node, graph, sub_mean_node, std)
                mul_gamma_node = inject_mul(mul_node, graph, grad_output, mul_node)
                sum_gamm_node = inject_sum(mul_gamma_node, graph, mul_gamma_node, reduction_dims)

                grad_beta.replace_all_uses_with(sum_beta_node)
                grad_gamma.replace_all_uses_with(sum_gamm_node)
                graph.erase_node(grad_beta)
                graph.erase_node(grad_gamma)

                # step 6: update meta
                if "tensor_meta" in backward_node.meta.keys():
                    backward_node.meta["tensor_meta"] = backward_node.meta["tensor_meta"][0]
                else:
                    backward_node.meta["tensor_meta"] = grad_input.meta["tensor_meta"]._replace()




