import torch
import sys
sys.path.append("/workspace/sparseTraining/Codegen/compiler")
sys.path.append("/workspace/bert")
from xmlcnn_modeling import xmlCNN, Params
from apex import amp
from lamb_amp_opt.fused_lamb import FusedLAMBAMP
from amp_helper import scale_loss
from aot_helper import compiler_fn, partition_func
import pycutlass
from pycutlass import *
pycutlass.get_memory_pool(manager="torch")
pycutlass.compiler.nvcc()
import nvtx

import argparse
parser = argparse.ArgumentParser(description="XML-CNN End-to-End Training with CUDA Graph")
# Modes
parser.add_argument("--mode", "-m", type=str, default="verify", choices=["verify", "profile"])
args = parser.parse_args()


params = Params(
    embedding_dim=304,  # alignment of 8
    filter_sizes=[2, 4, 8],
    sequence_length=512,
    batch_size=2048,
    num_filters=32,
    y_dim=670208,
    hidden_dims=512,
    pooling_chunks=8
)

e_emb = torch.randn(size=(params.batch_size, params.sequence_length, params.embedding_dim), dtype=torch.float16, device="cuda")
# y_prob = torch.ones(size=(params.batch_size, params.y_dim), device="cuda") * (5.45 / params.y_dim)
# y = torch.bernoulli(y_prob).to(torch.float32)
y = torch.empty(size=(params.batch_size, params.y_dim), dtype=torch.float16, device="cuda").random_(2)

learning_rate=6e-3

def prepare_model_and_optimizer(params, device, reference=None):
    model = xmlCNN(params)
    if reference is not None:
        reference_embedding_state_dict = reference.state_dict()
        model.load_state_dict(reference_embedding_state_dict)
    
    model.to(device)

    param_optimizer = list(model.named_parameters())
    no_decay = ['bias', 'gamma', 'beta']

    optimizer_grouped_parameters = [
        {'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)], 'weight_decay': 0.0},
        {'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}]

    optimizer = FusedLAMBAMP(optimizer_grouped_parameters,
                             lr=learning_rate)
    
    optimizer.setup_fp32_params()

    return model, optimizer

model, optimizer = prepare_model_and_optimizer(params, "cuda")

model, optimizer = amp.initialize(
    model, optimizer, 
    cast_model_outputs=torch.float16, 
    opt_level="O2", keep_batchnorm_fp32=False, 
    loss_scale="dynamic"
)

model.train()
optimizer.zero_grad()
loss_1 = model(e_emb, y)
loss_1.backward()

if args.mode == "profile":
    for i in range(10):
        loss = model(e_emb, y)
        loss.backward()
    
    with nvtx.annotate("pytorch 40 iter"):
        for i in range(40):
            with nvtx.annotate("torch"):
                loss = model(e_emb, y)
                loss.backward()
    
    torch.cuda.synchronize()
    


##############################################################################
model_fused, optimizer_fused = prepare_model_and_optimizer(params, "cuda", reference=model)

if args.mode == "profile":
    del model
    del optimizer

model_fused, optimizer_fused = amp.initialize(
    model_fused, optimizer_fused, 
    cast_model_outputs=torch.float16, 
    opt_level="O2", keep_batchnorm_fp32=False, 
    loss_scale="dynamic"
)

model_fused.train()
model_fused.aot_optimize(compiler_fn, compiler_fn, partition_func)
model_fused.capture_graph(params.batch_size, params.sequence_length, params.embedding_dim, params.y_dim, optimizer_fused)

optimizer_fused.zero_grad()
s = torch.cuda.Stream(priority=-1)
s.wait_stream(torch.cuda.current_stream())
with torch.cuda.stream(s):
    model_fused.training_with_graph(e_emb, y)

torch.cuda.current_stream().wait_stream(s)

if args.mode == "profile":
    s = torch.cuda.Stream(priority=-1)
    s.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(s):
        for i in range(10):
            model_fused.training_with_graph(e_emb, y)
        
        with nvtx.annotate("ours 40 iter"):
            for i in range(40):
                with nvtx.annotate("fused"):
                    model_fused.training_with_graph(e_emb, y)
    torch.cuda.current_stream().wait_stream(s)

# for i in range(40):
#     with nvtx.annotate("fused"):
#         loss = model_fused(e_emb, y)
#         loss.backward()

if args.mode == "verify":
    for param1, param2 in zip(list(model.named_parameters()), list(model_fused.named_parameters())):
        print(torch.sum(torch.isclose(param1[1].grad, param2[1].grad, rtol=1e-1)) / param2[1].grad.numel())
        try:
            assert torch.sum(torch.isclose(param1[1].grad, param2[1].grad, rtol=1e-1)) / param2[1].grad.numel() > 0.9
        except:
            print(param1[0])
            print(param1[1].grad)
            print(param2[1].grad)
            print(param1[1].grad.size())