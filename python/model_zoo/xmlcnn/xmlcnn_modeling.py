import torch
import torch.nn as nn
from functorch.compile import aot_module
from torch._dynamo.backends.common import aot_autograd

# torch.ops.load_library("/workspace/sparseTraining/Codegen/torchscript/build/libspmm_trace.so")
class Params:
    def __init__(self, embedding_dim, filter_sizes, sequence_length, batch_size, num_filters, y_dim, hidden_dims, pooling_units) -> None:
        self.embedding_dim = embedding_dim
        self.filter_sizes = filter_sizes
        self.sequence_length = sequence_length
        self.num_filters = num_filters
        self.y_dim = y_dim
        self.batch_size = batch_size
        self.hidden_dims = hidden_dims
        self.pooling_units = pooling_units


def out_size(l_in, kernel_size, padding, dilation=1, stride=1):
    a = l_in + 2*padding - dilation*(kernel_size - 1) - 1
    b = int(a/stride)
    return b + 1

class cnn_encoder(torch.nn.Module):
    
    def __init__(self, params):
        super(cnn_encoder, self).__init__()
        self.params = params
        self.conv_layers = nn.ModuleList()
        self.pool_layers = nn.ModuleList()
        fin_l_out_size = 0
        
        self.drp = nn.Dropout(p=1e-19)
        self.drp5 = nn.Dropout(p=1e-19)

        for fsz in params.filter_sizes:
            l_out_size = out_size(params.sequence_length, fsz, int(fsz/2-1), stride=2)
            pool_size = l_out_size // params.pooling_units
            l_conv = nn.Conv1d(params.embedding_dim, params.num_filters, fsz, stride=2, padding=int(fsz/2-1))
            torch.nn.init.xavier_uniform_(l_conv.weight)
            # if params.pooling_type == 'average':
            # l_pool = nn.AvgPool1d(pool_size, stride=None, count_include_pad=True)
            # pool_out_size = (int((l_out_size - pool_size)/pool_size) + 1)*params.num_filters
            # elif params.pooling_type == 'max':
            l_pool = nn.MaxPool1d(3, stride=1, padding=1)
            pool_out_size = l_out_size*params.num_filters
            fin_l_out_size += pool_out_size

            self.conv_layers.append(l_conv)
            self.pool_layers.append(l_pool)

        self.fin_layer = nn.Linear(fin_l_out_size, params.hidden_dims)
        self.out_layer = nn.Linear(params.hidden_dims, params.y_dim)
        torch.nn.init.xavier_uniform_(self.fin_layer.weight)
        torch.nn.init.xavier_uniform_(self.out_layer.weight)

    def forward(self, inputs):
        #o0 = self.drp(self.bn_1(inputs)).permute(0,2,1)
        o0 = inputs.permute(0,2,1)# self.bn_1(inputs.permute(0,2,1))
        # o0 = o0.contiguous()
        o0 = self.drp(o0) 
        conv_out = []

        for i in range(len(self.params.filter_sizes)):
            o = self.conv_layers[i](o0)
            o = o.view(o.shape[0], 1, o.shape[1]*o.shape[2])
            o = self.pool_layers[i](o)
            o = nn.functional.relu(o)
            o = o.view(o.shape[0],-1)
            conv_out.append(o)
            del o
        if len(self.params.filter_sizes)>1:
            o = torch.cat(conv_out,1)
        else:
            o = conv_out[0]

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
        Y = self.classifier.forward(e_emb)
        loss = self.loss_fn(Y, batch_y, reduction="sum") / e_emb.size(0)

        return loss
    

class xmlCNN(nn.Module):
    def __init__(self, params):
        super(xmlCNN, self).__init__()
        self.model = xmlCNN_(params)
        
    def forward(self, e_emb, batch_y):
        return self.model(e_emb, batch_y)
    
    def aot_optimize(self, fw_compiler, bw_compiler, partition_fn=None):
        if partition_fn is None:
            self.model = aot_module(
                self.model, fw_compiler=fw_compiler, 
                bw_compiler=bw_compiler)
        else:
            my_backend = aot_autograd(fw_compiler=fw_compiler, bw_compiler=bw_compiler, partition_fn=partition_fn)
            self.model = torch.compile(self.model, fullgraph=True, dynamic=False, backend=my_backend)
    
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
    
    def train_with_graph(self, e_emb, y):
        self.static_e_emb.copy_(e_emb)
        self.static_y.copy_(y)
        self.model_graph.replay()






    
