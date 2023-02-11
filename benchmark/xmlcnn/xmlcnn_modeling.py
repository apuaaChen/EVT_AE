import torch
import torch.nn as nn
import nvtx
from functorch.compile import aot_module

# torch.ops.load_library("/workspace/sparseTraining/Codegen/torchscript/build/libspmm_trace.so")
class Params:
    def __init__(self, embedding_dim, filter_sizes, sequence_length, batch_size, num_filters, y_dim, hidden_dims, pooling_chunks) -> None:
        self.embedding_dim = embedding_dim
        self.filter_sizes = filter_sizes
        self.sequence_length = sequence_length
        self.num_filters = num_filters
        self.y_dim = y_dim
        self.batch_size = batch_size
        self.hidden_dims = hidden_dims
        self.pooling_chunks = pooling_chunks


def out_size(l_in, kernel_size, padding=0, dilation=1, stride=1):
    a = l_in + 2*padding - dilation*(kernel_size - 1) - 1
    b = int(a/stride)
    return b + 1

class cnn_encoder(torch.nn.Module):
    
    def __init__(self, params):
        super(cnn_encoder, self).__init__()
        self.params = params
        self.conv_layers = nn.ModuleList()
        self.pool_layers = nn.ModuleList()
        
        self.drp = nn.Dropout(p=1e-19)
        self.drp5 = nn.Dropout(p=1e-19)

        for fsz in params.filter_sizes:
            l_conv = nn.Conv2d(params.embedding_dim, params.num_filters, (1, fsz), stride=(1, 2), padding=(0, fsz//2 - 1))

            # l_pool = nn.MaxPool1d(2, stride=1, padding=1)
            l_pool = nn.AdaptiveAvgPool1d(output_size=params.pooling_chunks)

            self.conv_layers.append(l_conv)
            self.pool_layers.append(l_pool)
        fin_l_out_size = params.num_filters * params.pooling_chunks * len(params.filter_sizes)
        self.fin_layer = nn.Linear(fin_l_out_size, params.hidden_dims)
        self.out_layer = nn.Linear(params.hidden_dims, params.y_dim)

    def forward(self, inputs):
        #o0 = self.drp(self.bn_1(inputs)).permute(0,2,1)
        o0 = inputs.permute(0,2,1)# self.bn_1(inputs.permute(0,2,1))
        # o0 = o0.contiguous()
        o0 = self.drp(o0) 
        conv_out = []

        for i in range(len(self.params.filter_sizes)):
            with nvtx.annotate("conv"):
                o = self.conv_layers[i](o0.unsqueeze(2)).squeeze()
            with nvtx.annotate("pool"):
                o = self.pool_layers[i](o).contiguous()
            # o = nn.functional.relu(o)
            o = o.view([o.size(0), -1])
            conv_out.append(o)
            del o
        if len(self.params.filter_sizes)>1:
            with nvtx.annotate("cat"):
                o = torch.cat(conv_out,1)
        else:
            o = conv_out[0]
        with nvtx.annotate("relu"):
            o = nn.functional.relu(o)
        o = self.fin_layer(o)
        o = nn.functional.relu(o)
        o = self.drp5(o) 
        o = self.out_layer(o)
        # o = torch.sigmoid(o)
        return o

class xmlCNN_(nn.Module):
    def __init__(self, params):
        super(xmlCNN_, self).__init__()
        self.classifier = cnn_encoder(params)
        self.loss_fn = torch.nn.functional.binary_cross_entropy_with_logits
        
    def forward(self, e_emb, batch_y):
        with nvtx.annotate("cnn encoder"):
            Y = self.classifier.forward(e_emb)
        with nvtx.annotate("loss"):
            loss = self.loss_fn(Y, batch_y, reduction="sum") / e_emb.size(0)

        return loss
    

class xmlCNN(nn.Module):
    def __init__(self, params):
        super(xmlCNN, self).__init__()
        self.model = xmlCNN_(params)
        
    def forward(self, e_emb, batch_y):
        return self.model(e_emb, batch_y)
    
    def aot_optimize(self, fw_compiler, bw_compiler, partition_fn):
        self.model = aot_module(
            self.model, fw_compiler=fw_compiler, 
            bw_compiler=bw_compiler, partition_fn=partition_fn)
    
    def capture_graph(self, batch_size, sequence_length, embedding_dim, y_dim, optimizer, warmup_iteration=3):
        device = next(self.parameters()).device
        # initialize the static tensors
        self.static_e_emb = torch.randn(size=(batch_size, sequence_length, embedding_dim), dtype=torch.float16, device=device)
        self.static_y = torch.ones(size=(batch_size, y_dim), dtype=torch.float16, device=device)

        # warmup iterations
        s = torch.cuda.Stream(priority=-1)
        s.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(s):
            for _ in range(warmup_iteration):
                optimizer.zero_grad()
                loss = self.model(self.static_e_emb, self.static_y)
                loss.backward()

        torch.cuda.current_stream().wait_stream(s)

        self.static_e_emb = torch.randn(size=(batch_size, sequence_length, embedding_dim), dtype=torch.float16, device=device)
        self.static_y = torch.ones(size=(batch_size, y_dim), dtype=torch.float16, device=device)

        # tracing iteration
        s = torch.cuda.Stream(priority=-1)
        s.wait_stream(torch.cuda.current_stream())
        optimizer.zero_grad()
        # self.model_graph = torch.classes.my_ops.SimpleClass(4)# torch.cuda.CUDAGraph()
        with torch.cuda.stream(s):
            self.model_graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(self.model_graph):
                loss = self.model(self.static_e_emb, self.static_y)
                loss.backward()
        torch.cuda.current_stream().wait_stream(s)
    
    def training_with_graph(self, e_emb, y):
        self.static_e_emb.copy_(e_emb)
        self.static_y.copy_(y)
        self.model_graph.replay()






    
