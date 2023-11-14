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
This file contains the implementation of XMLCNN
"""
from model_zoo.xmlcnn import xmlcnn_modeling
import torch
from model_zoo.xmlcnn.xmlcnn_modeling import Params


def prepare_model_and_optimizer(params, device="cuda", learning_rate=6e-3, reference=None):
    model = xmlcnn_modeling.xmlCNN(params)
    if reference is not None:
        reference_embedding_state_dict = reference.state_dict()
        model.load_state_dict(reference_embedding_state_dict)
    
    model.to(device)

    param_optimizer = list(model.named_parameters())
    no_decay = ['bias', 'gamma', 'beta']

    optimizer_grouped_parameters = [
        {'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)], 'weight_decay': 0.0},
        {'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}]

    optimizer = torch.optim.SGD(optimizer_grouped_parameters, learning_rate)
    return model, optimizer


def example_inputs(params, device="cuda"):
    e_emb = torch.randn(
        size=(params.batch_size, params.sequence_length, params.embedding_dim), 
        dtype=torch.float16, device=device)

    y = torch.empty(
        size=(params.batch_size, params.y_dim), 
        dtype=torch.float16, device=device).random_(2)
    
    return e_emb, y
