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
import torch.nn.functional as F
from passes.utils.subgraph_rewriter import replace_pattern_


################################################################################
# Registered substitution patterns
################################################################################
# register joint log softmax
# def pattern(input, grad_out):
#     log_softmax = torch.ops.aten._log_softmax(input, -1, False)
#     grad_input = torch.ops.aten._log_softmax_backward_data(grad_out, log_softmax, -1, False)
#     return log_softmax, grad_input

# def replacement(input, grad_out):
#     softmax = torch.ops.aten._softmax(input, -1, False)
#     log_softmax = torch.ops.aten.log(softmax)
#     sum = torch.sum(grad_out, dim=1, keepdim=True)
#     mul = softmax * sum
#     grad_input = grad_out - mul
#     return log_softmax, grad_input

# register nll_loss_backward
# def pattern(grad_output, tensor, target, total_weight):
#     return torch.ops.aten.nll_loss_backward(grad_output, tensor, target, None, 2, 0, total_weight)

# def replacement(grad_output, tensor, target, total_weight):
#     one_hot = torch.ops.aten.one_hot(target, num_classes=tensor.size(1))
#     neg = -one_hot
#     mul = grad_output * neg
#     ne = torch.ops.aten.ne(target, 0)
#     unsqueeze = torch.ops.aten.unsqueeze(ne, -1)
#     return mul * unsqueeze

################################################################################
# Graph-level pass to perform graph substitution
################################################################################
def pass_graph_substitution(module, graph):
    matches = replace_pattern_(module, pattern, replacement)
    print(matches)