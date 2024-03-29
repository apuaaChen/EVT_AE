from dgl.nn.pytorch import GATConv, edge_softmax
from dgl import function as fn
import dgl
import dgl.ops as F
import torch
from torch.fx._symbolic_trace import _create_wrapped_func
import torch.nn as nn
from functorch.compile import aot_module
from apex.contrib.xentropy import SoftmaxCrossEntropyLoss
from torch._dynamo.backends.common import aot_autograd

################################################################################
# DGL GCN
class DGLGraphConv(nn.Module):
    def __init__(self, in_dim, out_dim, activation=None) -> None:
        super(DGLGraphConv, self).__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.activation = activation
        self.weight = nn.Parameter(torch.Tensor(in_dim, out_dim))
        self.bias = nn.Parameter(torch.Tensor(out_dim))
        nn.init.xavier_normal_(self.weight)
        nn.init.zeros_(self.bias)
    
    def forward(self, graph, feat):
        with graph.local_scope():
            graph.ndata["h"] = feat
            graph.update_all(fn.u_mul_e('h', 'w', 'm'), fn.sum('m', 'h'))
            h = graph.ndata["h"]
            h = torch.matmul(h, self.weight) + self.bias
            if self.activation is not None:
                h = self.activation(h)
            return h

    def mfunc(self, edges):
        return {"m": edges.src["h"], "ci": edges.src["ci"]}

    def rfunc(self, nodes):
        ci = nodes.mailbox["ci"].unsqueeze(2)
        newh = (nodes.mailbox["m"] * ci).sum(1) * nodes.data["cj"].unsqueeze(1)
        return {"h": newh}

class DGLGCN_(nn.Module):
    def __init__(
        self, in_feats, n_hidden, n_classes, n_layers, activation, dropout
    ):
        super(DGLGCN_, self).__init__()
        self.layers = nn.ModuleList()
        # input layer
        self.layers.append(DGLGraphConv(in_feats, n_hidden, activation=activation))
        for i in range(n_layers - 1):
            self.layers.append(
                DGLGraphConv(n_hidden, n_hidden, activation=activation)
            )
        self.layers.append(DGLGraphConv(n_hidden, n_classes))
        self.dropout = nn.Dropout(p=dropout)
        self.loss = torch.nn.CrossEntropyLoss()

    def forward(self, g, features, labels):
        h = features
        for i, layer in enumerate(self.layers):
            if i != 0:
                h = self.dropout(h)
            h = layer(g, h)
        return self.loss(h, labels)

################################################################################


class myGATConv(GATConv):
    def set_graph(self, graph):
        self.graph = graph
    def forward(self, feat, get_attention=False):
        src_prefix_shape = dst_prefix_shape = feat.shape[:-1]
        h_src = h_dst = self.feat_drop(feat)
        feat_src = feat_dst = self.fc(h_src).view(
            *src_prefix_shape, self._num_heads, self._out_feats)
        
        # a \dot hi
        el = (feat_src * self.attn_l).sum(dim=-1).unsqueeze(-1)
        # a \dot hj
        er = (feat_dst * self.attn_r).sum(dim=-1).unsqueeze(-1)

        e = self.leaky_relu(F.u_add_v(self.graph, el, er))
        e = self.attn_drop(edge_softmax(self.graph, e))

        # aggregate
        rst = F.u_mul_e_sum(self.graph, feat_src, e)

        if self.res_fc is not None:
            # Use -1 rather than self._num_heads to handle broadcasting
            resval = self.res_fc(h_dst).view(*dst_prefix_shape, -1, self._out_feats)
            # rst = rst + resval
            rst = rst + resval
        # bias
        if self.bias is not None:
            rst = rst + self.bias.view(
                *((1,) * len(dst_prefix_shape)), self._num_heads, self._out_feats)
            
        # activation
        if self.activation:
            rst = self.activation(rst)

        return rst

# define autograph function
class Aggregate(torch.autograd.Function):
    @staticmethod
    def forward(ctx, csr, csc, x):
        ctx.save_for_backward(csc)
        # dcsr = SparseTensor.from_torch_sparse_csr_tensor(
        #     csr.clone().detach(), True, requires_grad=True
        # )
        # return spmm_sum(dcsr, x, 0)
        return torch.matmul(csr, x)

    @staticmethod
    def backward(ctx, grad_output):
        csc, = ctx.saved_tensors
        return None, None, torch.matmul(csc, grad_output)


class GraphConv(nn.Module):
    def __init__(self, in_dim, out_dim, activation=None):
        super(GraphConv, self).__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.activation = activation
        self.linear = nn.Linear(in_dim, out_dim, bias=True)
        # self.weight = nn.Parameter(torch.Tensor(in_dim, out_dim))
        # self.bias = nn.Parameter(torch.Tensor(out_dim))

    def forward(self, h):
        if self.out_dim > self.in_dim:
            # h = torch.matmul(self.graph, h) + h
            h = Aggregate.apply(self.csr, self.csc, h) + h
            h = self.linear(h)
        else:
            h = self.linear(h)
            h = Aggregate.apply(self.csr, self.csc, h) + h
        # h = F.u_mul_e_sum(self.graph, h, e) + h
        if self.activation is not None:
            h = self.activation(h)
        return h
    
    def set_graph(self, graph):
        self.csr = graph[0]
        self.csc = graph[1]


class GCN_(nn.Module):
    def __init__(
        self, in_feats, n_hidden, n_classes, n_layers, activation, dropout, f32_loss, apex_loss
    ):
        super(GCN_, self).__init__()
        self.layers = nn.ModuleList()
        # input layer
        self.layers.append(GraphConv(in_feats, n_hidden, activation=activation))
        # hidden layers
        for i in range(n_layers - 1):
            self.layers.append(
                GraphConv(n_hidden, n_hidden, activation=activation)
            )
        # output layer
        self.layers.append(GraphConv(n_hidden, n_classes))
        self.dropout = nn.Dropout(p=dropout)
        self.loss_fcn = torch.nn.CrossEntropyLoss(reduction='sum')
        self.apex_loss = apex_loss
        self.f32_loss = f32_loss
    
    def set_graph(self, graph):
        for layer in self.layers:
            layer.set_graph(graph)

    def forward(self, features, labels, mask=None):
        h = features
        for i, layer in enumerate(self.layers):
            if i != 0:
                h = self.dropout(h)
            h = layer(h)
        # return self.loss_fcn(h[mask], labels[mask])
        # issue: inf loss!!
        if not self.apex_loss:
            if self.f32_loss:
                h = h.to(torch.float32)
            return self.loss_fcn(h, labels)
        else:
            loss = SoftmaxCrossEntropyLoss.apply(h, labels, 0.0, 0, True).mean()
            return loss

class GCN(nn.Module):
    def __init__(
        self, in_feats, n_hidden, n_classes, n_layers, activation, dropout, f32_loss=True, apex_loss=False
    ):
        super(GCN, self).__init__()
        self.model = GCN_(in_feats, n_hidden, n_classes, n_layers, activation, dropout, f32_loss, apex_loss)

    def forward(self, features, labels):
        return self.model(features, labels)

    def set_graph(self, graph):
        for layer in self.model.layers:
            layer.set_graph(graph)
    
    def aot_optimize(self, fw_compiler, bw_compiler, partition_fn=None):
        if partition_fn is None:
            self.model = aot_module(
                self.model, fw_compiler=fw_compiler, 
                bw_compiler=bw_compiler)
        else:
            my_backend = aot_autograd(fw_compiler=fw_compiler, bw_compiler=bw_compiler, partition_fn=partition_fn)
            self.model = torch.compile(self.model, fullgraph=True, dynamic=False, backend=my_backend)
            # self.model = aot_module(
            #     self.model, fw_compiler=fw_compiler, 
            #     bw_compiler=bw_compiler, partition_fn=partition_fn)
    
    def torch_compile(self, backend="inductor", mode="default"):
        self.model = torch.compile(self.model, backend=backend, mode=mode)
    
    def capture_graph(self, features, labels, optimizer, warmup_iteration=3):
        self.static_features = torch.randn_like(features)
        self.static_labels = torch.ones_like(labels)
        # self.static_mask = torch.ones_like(mask)

        # warmup iterations
        s = torch.cuda.Stream(priority=-1)
        s.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(s):
            for _ in range(warmup_iteration):
                optimizer.zero_grad()
                loss = self.model(self.static_features, self.static_labels) * 1e-3
                loss.backward()
        torch.cuda.current_stream().wait_stream(s)

        self.static_features = torch.randn_like(features)
        self.static_labels = torch.ones_like(labels)
        # self.static_mask = torch.ones_like(mask)

        # tracing iterations
        self.encoder_graph = torch.cuda.CUDAGraph()
        optimizer.zero_grad()
        s = torch.cuda.Stream(priority=-1)
        s.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(s):
            with torch.cuda.graph(self.encoder_graph):
                loss = self.model(self.static_features, self.static_labels) * 1e-3
                loss.backward()
        
        torch.cuda.current_stream().wait_stream(s)

    def set_features(self, features, labels):
        self.static_features.copy_(features)
        self.static_labels.copy_(labels)
        
    def train_with_graph(self):
        self.encoder_graph.replay()
