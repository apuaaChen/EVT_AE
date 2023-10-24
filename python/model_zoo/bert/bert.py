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
This file contains the implementation of bert-large for NVIDIA's deep learning 
example.
"""
# Dependencies
import sys
sys.path.append("/workspace/bert")
import torch
import modeling
from model_zoo.bert import bert_modeling
import os


def prepare_model_and_optimizer(
        sequence_output_is_dense=False,
        reference=None,
        learning_rate=6e-3,
        config_file=None,
        device="cuda"):
    """
    Create the bert model and optimizer
    Args:
        sequence_output_is_dense: is True, compressed format is used, which may
            cause CUDA Graph issue as the tensor size is not static. 
            (Default: False)
        reference: if provided, the created model will load all weights from it
            (Default: None)
        learning_rate: the learning rate of the optimizer. (Default: 6e-3)
        config_file: a single json file contains the bert model. If not provided,
            the default config file in the zoo will be used
        device: the device for the model. (Default: cuda)
    Returns:
        model: the initialized bert model
        optimizer: the bert optimizer
    """
    
    # Load config file
    if config_file is None:
        config_file = os.path.join(
            os.path.dirname(os.path.abspath(__file__)),
            "larg.json"
        )
    config = modeling.BertConfig.from_json_file(config_file)

    # Padding for divisibility by 8
    if config.vocab_size % 8 != 0:
        config.vocab_size += 8 - (config.vocab_size % 8)
    
    model = bert_modeling.Bert(config, sequence_output_is_dense)

    if reference is not None:
        reference_embedding_state_dict = reference.state_dict()
        model.load_state_dict(reference_embedding_state_dict)
    
    model.to(device)

    param_optimizer = list(model.named_parameters())
    no_decay = ['bias', 'gamma', 'beta', 'LayerNorm']

    optimizer_grouped_parameters = [
        {'params': 
         [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)],
         'weight_decay': 0.01
        },
        {'params': 
         [p for n, p in param_optimizer if any(nd in n for nd in no_decay)], 
         'weight_decay': 0.0
        }
    ]

    optimizer = torch.optim.SGD(optimizer_grouped_parameters, learning_rate)
    model.checkpoint_activations(False)

    return model, optimizer

def example_inputs(batch_size, seq_len, device="cuda"):
    """
    Generate the example inputs
    Args:
        batch_size: int, the batch size
        seq_len: the sequence length
    Returns:
        input ids, token_type_ids, attention_mask, next_sentence_lables, labels
    """
    input_ids = torch.randint(
        low=101, high=29858, 
        size=(batch_size, seq_len), 
        dtype=torch.int64, device=device)
    
    token_type_ids = torch.randint(
        low=0, high=2, 
        size=(batch_size, seq_len), 
        dtype=torch.int64, device=device)
    
    attention_mask = torch.ones(
        size=(batch_size, seq_len), 
        dtype=torch.float16, device=device)
    
    next_sentence_labels = torch.randint(
        low=0, high=2, size=(batch_size,), 
        dtype=torch.int64, device=device)
    
    labels = torch.randint(
        low=-1, high=26555, size=(batch_size, seq_len), 
        dtype=torch.int64, device=device)
    
    return input_ids, token_type_ids, attention_mask, labels, labels, next_sentence_labels