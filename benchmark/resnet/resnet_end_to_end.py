import torch
from resnet_modeling import ResNet, BasicBlock, Bottleneck
import sys
sys.path.append("/workspace/sparseTraining/Codegen/compiler")
sys.path.append("/workspace/bert")
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
parser = argparse.ArgumentParser(description="ResNet End-to-End Training with CUDA Graph")
parser.add_argument('--depth', '-d', type=int, default=18, help="depth of the resnet")
parser.add_argument('--mode', '-m', type=str, default="verify", choices=["verify", "profile"])
args = parser.parse_args()


batch_size = 128

x = torch.randn(size=(batch_size, 4, 224, 224), dtype=torch.float16, device="cuda")
y = torch.randint(low=0, high=1000, size=(batch_size,), dtype=torch.int64, device="cuda")

learning_rate = 6e-3

def prepare_model_and_optimizer(device, depth, reference=None):
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

    optimizer = FusedLAMBAMP(optimizer_grouped_parameters,
                             lr=learning_rate)
    
    optimizer.setup_fp32_params()

    return model, optimizer

model, optimizer = prepare_model_and_optimizer("cuda", args.depth)
model = model.to(memory_format=torch.channels_last)
model, optimizer = amp.initialize(
    model, optimizer, 
    cast_model_outputs=torch.float16, 
    opt_level="O2", keep_batchnorm_fp32=True, 
    loss_scale="dynamic"
)
model.train()
model.capture_graph((batch_size, 4, 224, 224), optimizer)

model.train_with_graph(x, y)


##############################################################################
model_fused, optimizer_fused = prepare_model_and_optimizer("cuda", args.depth, reference=model)

model_fused, optimizer_fused = amp.initialize(
    model_fused, optimizer_fused, 
    cast_model_outputs=torch.float16, 
    opt_level="O2", keep_batchnorm_fp32=False, 
    loss_scale="dynamic"
)

model_fused.train()
model_fused.to_channels_last()
model_fused.aot_optimize(compiler_fn, compiler_fn, partition_func)
# loss = model_fused(x, y) * 1e+2
# loss.backward()
# 
model_fused.capture_graph((batch_size, 4, 224, 224), optimizer_fused)

optimizer_fused.zero_grad()
model_fused.train_with_graph(x, y)

torch.cuda.synchronize()

if args.mode == "verify":
    for param1, param2 in zip(list(model.named_parameters()), list(model_fused.named_parameters())):
        grad_origin = param1[1].grad.to(torch.float16)
        if len(grad_origin.size()) == 4:
            K, C, R, S = grad_origin.size()
            grad_fused = param2[1].grad.view(K, R, S, C).permute(0, 3, 1, 2).contiguous()
        else:
            grad_fused = param2[1].grad.contiguous()
        print(torch.sum(torch.isclose(grad_origin, grad_fused, rtol=3e-1)) / grad_origin.numel())
        try:
            assert torch.sum(torch.isclose(grad_origin, grad_fused, rtol=3e-1)) / grad_origin.numel() > 0.7
        except:
            print(param1[0])
            # print(grad_origin.contiguous().view(-1))
            # print(grad_fused.contiguous().view(-1))

# for profiling
if args.mode == "profile":

    for i in range(10):
        model.train_with_graph(x, y)

    with nvtx.annotate("nhwc_40"):
        for i in range(40):
            with nvtx.annotate("nhwc"):
                model.train_with_graph(x, y)

    x_nhwc = x.contiguous()
    for i in range(10):
        model_fused.train_with_graph(x_nhwc, y)

    with nvtx.annotate("ours_40"):
        for i in range(40):
            with nvtx.annotate("ours"):
                model_fused.train_with_graph(x_nhwc, y)