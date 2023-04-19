import torch
from ogb.nodeproppred import DglNodePropPredDataset
import dgl
from gcn_modeling import GCN
import torch.nn.functional as F
from tqdm import tqdm
import nvtx
import GLCC
import sys
sys.path.append("/workspace/bert")
from lamb_amp_opt.fused_lamb import FusedLAMBAMP
from aot_helper import compiler_fn, partition_func
from apex import amp
import pycutlass
from pycutlass import *
pycutlass.get_memory_pool(manager="torch")
pycutlass.compiler.nvcc()

import argparse
parser = argparse.ArgumentParser(description="GCN End-to-End Training with CUDA Graph")
parser.add_argument('--mode', '-m', type=str, default="verify", choices=["verify", "profile"])
args = parser.parse_args()

dataset = DglNodePropPredDataset(name = "ogbn-mag")

split_idx = dataset.get_idx_split()
train_idx, valid_idx, test_idx = split_idx["train"], split_idx["valid"], split_idx["test"]
g, labels = dataset[0]
g = g.to("cuda")

features = g.ndata["feat"]["paper"]
labels = labels["paper"]
labels = labels.to("cuda").squeeze()

d_i = torch.pow(g.out_degrees(etype='cites').float() + 1, -0.5)
d_j = torch.pow(g.in_degrees(etype='cites').float() + 1, -0.5)
e = dgl.ops.u_mul_v(g, d_i, d_j)[2].to(torch.float16)

row_idx, col_idx, e_idx = g.adj_sparse(fmt="csr", etype='cites')
csr = torch.sparse_csr_tensor(row_idx, col_idx, e)

csc_ = csr.transpose(0, 1)
csc = torch.sparse_csr_tensor(csc_.ccol_indices(), csc_.row_indices(), csc_.values())

in_feats = features.shape[1]
n_classes = ((dataset.num_classes + 7) // 8) * 8

def prepare_model_and_optimizer(f32_loss=True, embedding=64, depth=2, reference=None):
    model = GCN(in_feats, embedding, n_classes, depth, F.relu, 1e-16, f32_loss)
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

    optimizer = FusedLAMBAMP(optimizer_grouped_parameters,
                             lr=1e-2)
    
    optimizer.setup_fp32_params()

    return model, optimizer

features = features.to(torch.float16)

model, optimizer = prepare_model_and_optimizer(embedding=64, depth=2)
model, optimizer = amp.initialize(
    model, optimizer,
    cast_model_outputs=torch.float16, 
    opt_level="O2", keep_batchnorm_fp32=True, 
    loss_scale="dynamic"
)
model.train()

optimizer.zero_grad()
loss = model(features, labels) * 1e+3
loss.backward()

##############################################################################
model_fused, optimizer_fused = prepare_model_and_optimizer(
    f32_loss=False, embedding=64, depth=2, reference=model)
model_fused, optimizer_fused = amp.initialize(
    model_fused, optimizer_fused,
    cast_model_outputs=torch.float16, 
    opt_level="O2", keep_batchnorm_fp32=True, 
    loss_scale="dynamic"
)

model_fused.train()
model_fused.aot_optimize(compiler_fn, compiler_fn, partition_func)
model_fused.capture_graph(features, labels, optimizer_fused)
optimizer_fused.zero_grad()
# loss = model_fused(features, labels) * 1e+3
# loss.backward()
model_fused.set_features(features, labels)
model_fused.train_with_graph()

if args.mode == "verify":
    for param1, param2 in zip(list(model.named_parameters()), list(model_fused.named_parameters())):
        grad_origin = param1[1].grad
        grad_fused = param2[1].grad
        print(torch.sum(torch.isclose(grad_origin, grad_fused, rtol=1e-1)) / grad_origin.numel())
        try:
            assert torch.sum(torch.isclose(grad_origin, grad_fused, rtol=1e-1)) / grad_origin.numel() > 0.9
        except:
            print(param1[0])
            print(grad_origin.view(-1))
            print(grad_fused.view(-1))

if args.mode == "profile":
    for i in range(10):
        loss = model(features, labels)
        loss.backward()
    
    with nvtx.annotate("torch_40"):
        for i in range(40):
            with nvtx.annotate("torch"):
                loss = model(features, labels)
                loss.backward()
    
    s = torch.cuda.Stream(priority=-1)
    s.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(s):
        for i in range(10):
            model_fused.train_with_graph()
    
    with nvtx.annotate("ours_40"):
        for i in range(40):
            with nvtx.annotate("ours"):
                model_fused.train_with_graph()
# """
# model = GCN(in_feats, 64, n_classes, 2, F.relu, 0.5)
# model.set_graph((csr, csc))

# model = model.to("cuda")
# model.train()

# optimizer = torch.optim.Adam(model.parameters(), lr=1e-2, weight_decay=5e-4)
# model.capture_graph(features, labels, optimizer)


# # dry run
# for epoch in tqdm(range(10)):
#     # logits = model(features)
#     with nvtx.annotate("one epoch"):
#         loss = model(features, labels)
#         optimizer.zero_grad()
#         loss.backward()
#         optimizer.step()

# model.set_features(features, labels)
# for epoch in tqdm(range(10)):
#     # logits = model(features)
#     with nvtx.annotate("w/cuda graph"):
#         model.train_with_graph()
#         optimizer.step()


# # try GLCC
# adj_scipy = g.adj(scipy_fmt="csr", etype="cites")
# column_index = torch.IntTensor(adj_scipy.indices).cuda()
# row_pointers = torch.IntTensor(adj_scipy.indptr).cuda()

# h = torch.randn(size=(features.shape[0], 64), dtype=torch.float32, device="cuda")

# Ty, Tx, Py, Px, Vy, Vx = 64, 8, 32, 8, 2, 1
# graph_adj = torch.sparse_csr_tensor(row_pointers, column_index, torch.ones(column_index.size()).cuda())
# result_ref = torch.sparse.mm(graph_adj, h)
# result_act = GLCC.csrspmm_myrow(h, row_pointers, column_index, Ty, Tx, Py, Px, Vy, Vx)[0]
# if not torch.allclose(result_ref, result_act):
#     print(result_ref.view(-1))
#     print(result_act.view(-1))
# for i in range(10):
#     with nvtx.annotate("glcc GCN"):
#         h = GLCC.csrspmm_myrow(h, row_pointers, column_index, Ty, Tx, Py, Px, Vy, Vx)[0]
# """