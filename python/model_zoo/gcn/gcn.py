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
This file contains the implementation of GCN
"""
# Depdencies
import torch
from ogb.nodeproppred import DglNodePropPredDataset
from model_zoo.gcn.gcn_modeling import GCN
import torch.nn.functional as F
import dgl


def prepare_model_and_optimizer(in_feats, n_classes, csr, csc, f32_loss=True, embedding=64, depth=2, reference=None, apex_loss=False):
    model = GCN(in_feats, embedding, n_classes, depth, F.relu, 1e-16, f32_loss, apex_loss)
    model.set_graph((csr, csc))
    if reference is not None:
        reference_embedding_state_dict = reference.state_dict()
        model.load_state_dict(reference_embedding_state_dict)
    model = model.to("cuda")

    param_optimizer = list(model.named_parameters())
    no_decay = ['bias', 'gamma', 'beta']

    optimizer_grouped_parameters = [
        {'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)], 'weight_decay': 0.0},
        {'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}]

    optimizer = torch.optim.SGD(optimizer_grouped_parameters, 1e-2)

    return model, optimizer

def example_inputs(dataset_name):
    dataset = DglNodePropPredDataset(name=dataset_name)
    # split_idx = dataset.get_idx_split()
    # train_idx, valid_idx, test_idx = split_idx["train"], split_idx["valid"], split_idx["test"]
    g, labels = dataset[0]
    g = g.to("cuda")

    features = g.ndata["feat"]["paper"]
    labels = labels["paper"]
    labels = labels.to("cuda").squeeze()

    d_i = torch.pow(g.out_degrees(etype='cites').float() + 1, -0.5)
    d_j = torch.pow(g.in_degrees(etype='cites').float() + 1, -0.5)
    e = dgl.ops.u_mul_v(g, d_i, d_j)[2].to(torch.float16)

    row_idx, col_idx, e_idx = g.adj(etype='cites').csr()
    csr = torch.sparse_csr_tensor(row_idx, col_idx, e)

    csc_ = csr.transpose(0, 1)
    csc = torch.sparse_csr_tensor(csc_.ccol_indices(), csc_.row_indices(), csc_.values())

    in_feats = features.shape[1]
    n_classes = ((dataset.num_classes + 7) // 8) * 8
    features = features.to(torch.float16)

    return features, labels, csr, csc, in_feats, n_classes
