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
"""
This file contains the implementation of ResNet
"""
# Dependencies
import torch
from model_zoo.resnet.resnet_modeling import ResNet, BasicBlock, Bottleneck


def prepare_model_and_optimizer(depth, device="cuda", reference=None):
    if depth == 18:
        model = ResNet(block=BasicBlock, layers=[2, 2, 2, 2])
    elif depth == 50:
        model = ResNet(block=Bottleneck, layers=[3, 4, 6, 3])
    elif depth == 101:
        model = ResNet(block=Bottleneck, layers=[3, 4, 23, 3])
    else:
        raise NotImplementedError()
    if reference is not None:
        reference_embedding_state_dict = reference.state_dict()
        model.load_state_dict(reference_embedding_state_dict)
    model.to(device)

    param_optimizer = list(model.named_parameters())
    no_decay = ['bias', 'gamma', 'beta']

    optimizer_grouped_parameters = [
        {'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)], 'weight_decay': 0.0},
        {'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}]

    optimizer = torch.optim.SGD(optimizer_grouped_parameters, 6e-3)

    return model, optimizer

def example_inputs(batch_size, device="cuda"):
    """
    Generate the example inputs
    Args:
        batch_size: int, the batch size
    Returns:
        x, y
    """
    x = torch.randn(size=(batch_size, 4, 224, 224), dtype=torch.float16, device=device)
    y = torch.randint(low=0, high=1000, size=(batch_size,), dtype=torch.int64, device="cuda")

    return x, y