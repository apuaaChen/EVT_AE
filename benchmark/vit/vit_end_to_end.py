import torch
from vit_modeling import ViT
from apex import amp
import sys
sys.path.append("/workspace/sparseTraining/Codegen/compiler")
sys.path.append("/workspace/bert")
from lamb_amp_opt.fused_lamb import FusedLAMBAMP
from aot_helper import compiler_fn, partition_func
import pycutlass
from pycutlass import *
pycutlass.get_memory_pool(manager="torch")
pycutlass.compiler.nvcc()
import nvtx

import argparse
parser = argparse.ArgumentParser(description="Vit End-to-End Training with CUDA Graph")
parser.add_argument('--mode', '-m', type=str, default="verify", choices=["verify", "profile"])
args = parser.parse_args()

batch_size = 128

x = torch.randn(size=(batch_size, 3, 224, 224), dtype=torch.float16, device="cuda")
y = torch.randint(low=0, high=1000, size=(batch_size,), dtype=torch.int64, device="cuda")

learning_rate = 6e-3

def prepare_model_and_optimizer(device, reference=None):
    model = ViT(
        image_size=224,
        patch_size=14,
        num_classes=1000,
        dim=768,
        depth=12,
        heads=12,
        mlp_dim=3072,
        dropout=1e-16,
        emb_dropout=1e-16
    )
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

model, optimizer = prepare_model_and_optimizer("cuda")
model, optimizer = amp.initialize(
    model, optimizer, 
    cast_model_outputs=torch.float16, 
    opt_level="O2", keep_batchnorm_fp32=True, 
    loss_scale="dynamic"
)
model.train()

optimizer.zero_grad()
loss = model(x, y) * 1e+2
loss.backward()

##############################################################################
model_fused, optimizer_fused = prepare_model_and_optimizer("cuda", reference=model)
model_fused, optimizer_fused = amp.initialize(
    model_fused, optimizer_fused, 
    cast_model_outputs=torch.float16, 
    opt_level="O2", keep_batchnorm_fp32=False, 
    loss_scale="dynamic"
)

model_fused.train()
model_fused.aot_optimize(compiler_fn, compiler_fn, partition_func)
optimizer_fused.zero_grad()
loss = model_fused(x, y) * 1e+2
loss.backward()

if args.mode == "verify":
    for param1, param2 in zip(list(model.named_parameters()), list(model_fused.named_parameters())):
        grad_origin = param1[1].grad
        grad_fused = param2[1].grad
        print(torch.sum(torch.isclose(grad_origin, grad_fused, rtol=1e-1)) / grad_origin.numel())
        try:
            assert torch.sum(torch.isclose(grad_origin, grad_fused, rtol=1e-1)) / grad_origin.numel() > 0.9
        except:
            print(param1[0])
            # print(grad_origin.contiguous().view(-1))
            # print(grad_fused.contiguous().view(-1))

if args.mode == "profile":
    for i in range(10):
        loss = model(x, y)
        loss.backward()
    
    with nvtx.annotate("torch_40"):
        for i in range(10):
            with nvtx.annotate("torch"):
                loss = model(x, y)
                loss.backward()
    
    s = torch.cuda.Stream(priority=-1)
    s.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(s):
        for i in range(10):
            loss = model_fused(x, y)
            loss.backward()
        
        with nvtx.annotate("torch_40"):
            for i in range(10):
                with nvtx.annotate("torch"):
                    loss = model_fused(x, y)
                    loss.backward()