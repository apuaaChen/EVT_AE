import torch
from resnet_modeling import ResNet, BasicBlock
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

batch_size = 128

x = torch.randn(size=(batch_size, 3, 224, 224), dtype=torch.float16, device="cuda")
y = torch.randint(low=0, high=1000, size=(batch_size,), dtype=torch.int64, device="cuda")

learning_rate = 6e-3

def prepare_model_and_optimizer(device, reference=None):
    model = ResNet(block=BasicBlock, layers=[2, 2, 2, 2])
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
    opt_level="O2", keep_batchnorm_fp32=False, 
    loss_scale="dynamic"
)
model.train()
optimizer.zero_grad()
# for i in range(10):
#     with nvtx.annotate("nchw"):
#         loss_1 = model(x, y)
#         loss_1.backward()

# for i in range(10):
#     with nvtx.annotate("nchw"):
loss_1 = model(x, y) * 1e+4
loss_1.backward()


# model = model.to(memory_format=torch.channels_last)
# x = x.to(memory_format=torch.channels_last)

# for i in range(10):
#     with nvtx.annotate("nhwc"):
#         loss_1 = model(x, y)
#         loss_1.backward()


##############################################################################
model_fused, optimizer_fused = prepare_model_and_optimizer("cuda", reference=model)

model_fused, optimizer_fused = amp.initialize(
    model_fused, optimizer_fused, 
    cast_model_outputs=torch.float16, 
    opt_level="O2", keep_batchnorm_fp32=False, 
    loss_scale="dynamic"
)

model_fused.train()
# model_fused = model_fused.to(memory_format=torch.channels_last)
model_fused.aot_optimize(compiler_fn, compiler_fn, partition_func)
# model_fused = model_fused.to(memory_format=torch.channels_last)

optimizer_fused.zero_grad()
# loss = model_fused(x.to(memory_format=torch.channels_last), y)
loss = model_fused(x, y) * 1e+4
loss.backward()

torch.cuda.synchronize()

# if args.mode == "verify":
for param1, param2 in zip(list(model.named_parameters()), list(model_fused.named_parameters())):
    print(torch.sum(torch.isclose(param1[1].grad, param2[1].grad, rtol=3e-1)) / param2[1].grad.numel())
    try:
        assert torch.sum(torch.isclose(param1[1].grad, param2[1].grad, rtol=3e-1)) / param2[1].grad.numel() > 0.7
    except:
        print(param1[0])
        print(param1[1].grad.view(-1))
        print(param2[1].grad.view(-1))