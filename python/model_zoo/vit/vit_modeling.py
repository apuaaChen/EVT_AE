import torch
from torch import nn

from functorch.compile import aot_module
from torch._dynamo.backends.common import aot_autograd

# helpers

def pair(t):
    return t if isinstance(t, tuple) else (t, t)

# classes

class PreNorm(nn.Module):
    def __init__(self, dim, fn):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.fn = fn
    def forward(self, x, **kwargs):
        return self.fn(self.norm(x), **kwargs)

class FeedForward(nn.Module):
    def __init__(self, dim, hidden_dim, dropout = 0.):
        super().__init__()
        self.dim = dim
        self.hidden_dim = hidden_dim
        self.net = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(dropout)
        )
    def forward(self, x):
        return self.net(x)

class Attention(nn.Module):
    def __init__(self, dim, heads = 8, dim_head = 64, dropout = 0.):
        super().__init__()
        inner_dim = dim_head *  heads
        project_out = not (heads == 1 and dim_head == dim)

        self.heads = heads
        self.dim_head = dim_head
        self.scale = dim_head ** -0.5
        self.dim = dim

        self.attend = nn.Softmax(dim = -1)
        self.dropout = nn.Dropout(dropout)

        self.query = nn.Linear(dim, inner_dim)
        self.key = nn.Linear(dim, inner_dim)
        self.value = nn.Linear(dim, inner_dim)

        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, dim),
            nn.Dropout(dropout)
        ) if project_out else nn.Identity()
    
    def transpose_for_scores(self, x):
        x = x.view(x.size(0), x.size(1) * self.heads, self.dim_head).transpose(0, 1)
        return x

    def transpose_key_for_scores(self, x):
        x = x.view(x.size(0), x.size(1) * self.heads, self.dim_head).permute(1, 2, 0)
        return x

    def forward(self, x):
        batch_size = x.size(1)
        seq_length = x.size(0)
        mixed_query_layer = self.query(x)
        mixed_key_layer = self.key(x)
        mixed_value_layer = self.value(x)

        q = self.transpose_for_scores(mixed_query_layer)
        k = self.transpose_key_for_scores(mixed_key_layer)
        v = self.transpose_for_scores(mixed_value_layer)

        # qkv = self.to_qkv(x).chunk(3, dim = -1)
        # q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h = self.heads), qkv)

        dots = torch.bmm(q, k) * self.scale

        attn = self.attend(dots)
        attn = self.dropout(attn)

        out = torch.bmm(attn, v)
        out = out.transpose(0, 1).contiguous()
        out = out.view(seq_length, batch_size, self.dim)
        # out = rearrange(out, 'b h n d -> b n (h d)')
        return self.to_out(out)
    
class TransformerLayer(nn.Module):
    def __init__(self, dim, heads, dim_head, mlp_dim, dropout = 0.):
        super().__init__()
        self.attention = PreNorm(dim, Attention(dim, heads = heads, dim_head = dim_head, dropout = dropout))
        self.ff = PreNorm(dim, FeedForward(dim, mlp_dim, dropout = dropout))
    
    def forward(self, x):
        x = self.attention(x) + x
        x = self.ff(x) + x
        return x

class Transformer(nn.Module):
    def __init__(self, dim, depth, heads, dim_head, mlp_dim, dropout = 0.):
        super().__init__()
        self.layers = nn.ModuleList([])
        for _ in range(depth):
            self.layers.append(TransformerLayer(dim, heads, dim_head, mlp_dim, dropout))
    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x

class ViT_(nn.Module):
    def __init__(self, image_size, patch_size, num_classes, dim, depth, heads, mlp_dim, pool = 'mean', channels = 3, dim_head = 64, dropout = 0., emb_dropout = 0., batch_size = 128):
        super().__init__()
        image_height, image_width = pair(image_size)
        patch_height, patch_width = pair(patch_size)

        assert image_height % patch_height == 0 and image_width % patch_width == 0, 'Image dimensions must be divisible by the patch size.'

        num_patches = (image_height // patch_height) * (image_width // patch_width)
        self.patch_height = patch_height
        self.patch_width = patch_width
        patch_dim = channels * patch_height * patch_width
        self.patch_dim = patch_dim
        assert pool in {'cls', 'mean'}, 'pool type must be either cls (cls token) or mean (mean pooling)'

        self.to_patch_embedding_linear = nn.Linear(patch_dim, dim)

        self.pos_embedding = nn.Parameter(torch.randn(1, num_patches + 8, dim))
        # self.pos_embedding = nn.Parameter(torch.randn(1, num_patches, dim))
        self.cls_token = nn.Parameter(torch.randn(128, 8, dim))
        self.dropout = nn.Dropout(emb_dropout)

        self.transformer = Transformer(dim, depth, heads, dim_head, mlp_dim, dropout)

        self.pool = pool
        self.to_latent = nn.Identity()

        self.mlp_head = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, num_classes)
        )
        self.loss = nn.CrossEntropyLoss(label_smoothing=0.0, reduction="sum")
    
    def to_patch_embedding(self, x):
        n, c, h, w = x.size()
        x = x.view(n, c, h // self.patch_height, self.patch_height, w // self.patch_width, self.patch_width).permute(0, 2, 4, 3, 5, 1).contiguous()
        x = x.view(n, -1, self.patch_dim)
        return self.to_patch_embedding_linear(x)

    def forward(self, img, y):
        x = self.to_patch_embedding(img)
        b, n, _ = x.shape

        x = torch.cat((self.cls_token, x), dim=1)
        x += self.pos_embedding

        x = self.dropout(x)
        # permute to l, b, d
        x = x.permute(1, 0, 2).contiguous()

        x = self.transformer(x)
        # permute to b, l, d
        x = x.permute(1, 0, 2).contiguous()
        x = x.mean(dim = 1) if self.pool == 'mean' else x[:, 0]

        x = self.to_latent(x)
        x = self.mlp_head(x)
        return self.loss(x, y)

class ViT(nn.Module):
    def __init__(
        self, image_size, patch_size, num_classes, dim, depth, heads, mlp_dim, 
        pool = 'mean', channels=3, dim_head=64, dropout=0., emb_dropout=0.):
        #
        super().__init__()

        self.model = ViT_(
            image_size, patch_size, num_classes, dim, depth, heads, 
            mlp_dim, pool, channels, dim_head, dropout, emb_dropout)
    
    def forward(self, img, y):
        return self.model(img, y)

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
        self.model = torch.compile(self.model, backend=backend, mode=mode, fullgraph=True, dynamic=False)

    def capture_graph(self, input_size, optimizer, warmup_iteration=3):
        self.static_x = torch.randn(size=input_size, dtype=torch.float16, device="cuda")
        self.static_y = torch.randint(low=0, high=1000, size=(input_size[0],), dtype=torch.int64, device="cuda")

        # warmup iterations
        s = torch.cuda.Stream(priority=-1)
        s.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(s):
            for _ in range(warmup_iteration):
                optimizer.zero_grad()
                loss = self.model(self.static_x, self.static_y) * 1e+2
                loss.backward()
        
        torch.cuda.current_stream().wait_stream(s)

        self.static_x = torch.randn(size=input_size, dtype=torch.float16, device="cuda")
        self.static_y = torch.randint(low=0, high=1000, size=(input_size[0],), dtype=torch.int64, device="cuda")

        # tracing iterations
        self.encoder_graph = torch.cuda.CUDAGraph()
        optimizer.zero_grad()
        s = torch.cuda.Stream(priority=-1)
        s.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(s):
            with torch.cuda.graph(self.encoder_graph):
                loss = self.model(self.static_x, self.static_y) * 1e+2
                loss.backward()
        
        torch.cuda.current_stream().wait_stream(s)
    
    def train_with_graph(self, x, y):
        self.static_x.copy_(x)
        self.static_y.copy_(y)

        self.encoder_graph.replay()