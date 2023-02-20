import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import copy
import math


def _no_grad_trunc_normal_(tensor, mean, std, a, b):
    def norm_cdf(x):
        return (1. + math.erf(x / math.sqrt(2.))) / 2.

    with torch.no_grad():
        l = norm_cdf((a - mean) / std)
        u = norm_cdf((b - mean) / std)
        tensor.uniform_(2 * l - 1, 2 * u - 1)
        tensor.erfinv_()
        tensor.mul_(std * math.sqrt(2.))
        tensor.add_(mean)
        tensor.clamp_(min=a, max=b)
        return tensor


def drop_path(x, drop_prob: float = 0., training: bool = False):
    if drop_prob == 0. or not training:
        return x
    keep_prob = 1 - drop_prob
    shape = (x.shape[0],) + (1,) * (x.ndim - 1)  # work with diff dim tensors, not just 2D ConvNets
    random_tensor = keep_prob + torch.rand(shape, dtype=x.dtype, device=x.device)
    random_tensor.floor_()  # binarize
    output = x.div(keep_prob) * random_tensor
    return output


def gelu(x):
    return x * 0.5 * (1.0 + torch.erf(x / math.sqrt(2.0)))


def trunc_normal_(tensor, mean=0., std=1., a=-2., b=2.):
    return _no_grad_trunc_normal_(tensor, mean, std, a, b)


class Matmul(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x1, x2):
        x = x1 @ x2
        return x


class DropPath(nn.Module):
    def __init__(self, drop_prob=None):
        super(DropPath, self).__init__()
        self.drop_prob = drop_prob

    def forward(self, x):
        return drop_path(x, self.drop_prob, self.training)


class PixelNorm(nn.Module):
    def __init__(self, dim):
        super().__init__()

    def forward(self, input):
        return input * torch.rsqrt(torch.mean(input ** 2, dim=2, keepdim=True) + 1e-8)


class CustomAct(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        return x * 0.5 * (1.0 + torch.erf(x / math.sqrt(2.0)))


class Mlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = CustomAct()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


class Attention(nn.Module):
    def __init__(self, dim, num_heads=8, qkv_bias=False, qk_scale=None, attn_drop=0., proj_drop=0., window_size=16):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim ** -0.5
        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)
        self.mat = Matmul()
        self.window_size = window_size

        self.relative_position_bias_table = nn.Parameter(
            torch.zeros((2 * window_size - 1) * (2 * window_size - 1), num_heads))  # 2*Wh-1 * 2*Ww-1, nH

        # get pair-wise relative position index for each token inside the window
        coords_h = torch.arange(window_size)
        coords_w = torch.arange(window_size)
        coords = torch.stack(torch.meshgrid([coords_h, coords_w]))  # 2, Wh, Ww
        coords_flatten = torch.flatten(coords, 1)  # 2, Wh*Ww
        relative_coords = coords_flatten[:, :, None] - coords_flatten[:, None, :]  # 2, Wh*Ww, Wh*Ww
        relative_coords = relative_coords.permute(1, 2, 0).contiguous()  # Wh*Ww, Wh*Ww, 2
        relative_coords[:, :, 0] += window_size - 1  # shift to start from 0
        relative_coords[:, :, 1] += window_size - 1
        relative_coords[:, :, 0] *= 2 * window_size - 1
        relative_position_index = relative_coords.sum(-1)  # Wh*Ww, Wh*Ww
        self.register_buffer("relative_position_index", relative_position_index)

        trunc_normal_(self.relative_position_bias_table, std=.02)

    def forward(self, x):
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]
        attn = (self.mat(q, k.transpose(-2, -1))) * self.scale
        relative_position_bias = self.relative_position_bias_table[self.relative_position_index.view(-1)].view(
            self.window_size * self.window_size, self.window_size * self.window_size, -1)  # Wh*Ww,Wh*Ww,nH
        relative_position_bias = relative_position_bias.permute(2, 0, 1).contiguous()  # nH, Wh*Ww, Wh*Ww
        attn = attn + relative_position_bias.unsqueeze(0)

        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)
        x = self.mat(attn, v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


class CustomNorm(nn.Module):
    def __init__(self, norm_layer, dim):
        super().__init__()
        self.norm_type = norm_layer
        if norm_layer == "ln":
            self.norm = nn.LayerNorm(dim)
        elif norm_layer == "bn":
            self.norm = nn.BatchNorm1d(dim)
        elif norm_layer == "in":
            self.norm = nn.InstanceNorm1d(dim)
        elif norm_layer == "pn":
            self.norm = PixelNorm(dim)

    def forward(self, x):
        if self.norm_type == "bn" or self.norm_type == "in":
            x = self.norm(x.permute(0, 2, 1)).permute(0, 2, 1)
            return x
        elif self.norm_type == "none":
            return x
        else:
            return self.norm(x)


class Block(nn.Module):
    def __init__(self, dim, num_heads, mlp_ratio=4., qkv_bias=False, qk_scale=None, drop=0., attn_drop=0.,
                 drop_path=0., norm_layer='ln', window_size=16):
        super().__init__()
        self.norm1 = CustomNorm(norm_layer, dim)
        self.attn = Attention(
            dim, num_heads=num_heads, qkv_bias=qkv_bias, qk_scale=qk_scale, attn_drop=attn_drop, proj_drop=drop,
            window_size=window_size)
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = CustomNorm(norm_layer, dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, drop=drop)

    def forward(self, x):
        x = x + self.drop_path(self.attn(self.norm1(x)))
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        return x


class StageBlock(nn.Module):
    def __init__(self, depth, dim, num_heads, mlp_ratio=4., qkv_bias=False, qk_scale=None, drop=0., attn_drop=0.,
                 drop_path=0., norm_layer='ln', window_size=16):
        super().__init__()
        self.depth = depth
        models = [Block(
            dim=dim,
            num_heads=num_heads,
            mlp_ratio=mlp_ratio,
            qkv_bias=qkv_bias,
            qk_scale=qk_scale,
            drop=drop,
            attn_drop=attn_drop,
            drop_path=drop_path,
            norm_layer=norm_layer,
            window_size=window_size
        ) for i in range(depth)]
        self.block = nn.Sequential(*models)

    def forward(self, x):
        x = self.block(x)
        return x


class TGenerator(nn.Module):
    def __init__(self):
        super().__init__()
        self.mlp = nn.Linear(in_features=256, out_features=65536)
        self.pos_embed_1 = nn.Parameter(torch.zeros(1, 64, 1024))
        self.pos_embed_2 = nn.Parameter(torch.zeros(1, 256, 1024 // 4))
        self.pos_embed_3 = nn.Parameter(torch.zeros(1, 1024, 1024 // 16))

        self.blocks_1 = StageBlock(
            depth=5,
            dim=1024,
            num_heads=4,
            mlp_ratio=4,
            qkv_bias=False,
            qk_scale=None,
            drop=.1,
            attn_drop=.1,
            drop_path=0,
            norm_layer='ln',
            window_size=8
        )

        self.blocks_2 = StageBlock(
            depth=4,
            dim=1024 // 4,
            num_heads=4,
            mlp_ratio=4,
            qkv_bias=False,
            qk_scale=None,
            drop=.1,
            attn_drop=.1,
            drop_path=0,
            norm_layer='ln',
            window_size=16
        )

        self.blocks_3 = StageBlock(
            depth=2,
            dim=1024 // 16,
            num_heads=4,
            mlp_ratio=4,
            qkv_bias=False,
            qk_scale=None,
            drop=.1,
            attn_drop=.1,
            drop_path=0,
            norm_layer='ln',
            window_size=32
        )

        self.deconv = nn.Sequential(
            nn.Conv2d(1024 // 16, 3, 1, 1, 0)
        )

    def forward(self, x):
        x = self.mlp(x).view(-1, 64, 1024)
        x = x + self.pos_embed_1
        B = x.size()
        H, W = 8, 8
        x = self.blocks_1(x)

        x, H, W = self.bicubic_upsample(x, H, W)
        x = x + self.pos_embed_2
        B, _, C = x.size()
        x = self.blocks_2(x)

        x, H, W = self.bicubic_upsample(x, H, W)
        x = x + self.pos_embed_3
        B, _, C = x.size()
        x = self.blocks_3(x)
        output = self.deconv(x.permute(0, 2, 1).view(-1, 1024 // 16, H, W))
        return output

    def bicubic_upsample(self, x, H, W):
        B, N, C = x.size()
        assert N == H * W
        x = x.permute(0, 2, 1)
        x = x.view(-1, C, H, W)
        x = nn.PixelShuffle(2)(x)
        B, C, H, W = x.size()
        x = x.view(-1, C, H * W)
        x = x.permute(0, 2, 1)
        return x, H, W


class PositionEmbedding(nn.Module):
    def __init__(self, input_seq, d_model):
        super().__init__()
        self.position_embedding = nn.Parameter(torch.zeros(1, input_seq, d_model))

    def forward(self, x):
        x = x + self.position_embedding
        return x


class MHA(nn.Module):
    def __init__(self, d_model, nhead, dropout):
        super(MHA, self).__init__()
        self.q = nn.Linear(d_model, d_model)
        self.k = nn.Linear(d_model, d_model)
        self.v = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(dropout)
        self.nhead = nhead
        self.scores = None

    def forward(self, x):
        q = self.q(x)
        k = self.k(x)
        v = self.v(x)
        q, k, v = (self.split(x, (self.nhead, -1)).transpose(1, 2) for x in [q, k, v])
        scores = q @ k.transpose(-2, -1) / np.sqrt(k.size(-1))
        scores = self.dropout(F.softmax(scores, dim=-1))
        h = (scores @ v).transpose(1, 2).contiguous()
        h = self.merge(h, 2)
        self.scores = scores
        return h, scores

    def split(self, x, shape):
        shape = list(shape)
        assert shape.count(-1) <= 1
        if -1 in shape:
            shape[shape.index(-1)] = int(x.size(-1) / -np.prod(shape))
        return x.view(*x.size()[:-1], *shape)

    def merge(self, x, n_dims):
        s = x.size()
        assert 1 < n_dims < len(s)
        return x.view(*s[:-n_dims], -1)


class FeedForward(nn.Module):
    def __init__(self, d_model, d_ff):
        super(FeedForward, self).__init__()
        self.ff1 = nn.Linear(in_features=d_model, out_features=d_ff)
        self.ff2 = nn.Linear(in_features=d_ff, out_features=d_model)

    def forward(self, x):
        x = self.ff2(F.gelu(self.ff1(x)))
        return x


class TransformerEncoderLayer(nn.Module):
    def __init__(self, d_model, nhead, d_ff, dropout):
        super().__init__()
        self.attn = MHA(d_model=d_model, nhead=nhead, dropout=dropout)
        self.linproj = nn.Linear(in_features=d_model, out_features=d_model)
        self.norm1 = nn.LayerNorm(normalized_shape=d_model, eps=1e-6)

        self.ff = FeedForward(d_model=d_model, d_ff=d_ff)
        self.norm2 = nn.LayerNorm(normalized_shape=d_model, eps=1e-6)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        h, scores = self.attn(self.norm1(x))
        h = self.dropout(self.linproj(h))
        x = x + h
        h = self.dropout(self.ff(self.norm2(x)))
        x = x + h
        return x, scores


class TransformerEncoder(nn.Module):
    def __init__(self, num_layers, d_model, nhead, d_ff, dropout):
        super(TransformerEncoder, self).__init__()
        self.blocks = nn.ModuleList([
            TransformerEncoderLayer(d_model, nhead, d_ff, dropout) for _ in range(num_layers)])

    def forward(self, x):
        scores = []
        for block in self.blocks:
            x, score = block(x)
            scores.append(score)
        return x, scores


class TDiscriminator(nn.Module):
    def __init__(self,
                 patches=(2,2),  # Patch size: height width
                 d_model=384,  # Token Dim
                 d_ff=384,  # Feed Forward Dim
                 num_heads=4,  # Num MHA
                 num_layers=1,  # Num Transformer Layers
                 dropout=.1,  # Dropout rate
                 image_size=(3, 32, 32),  # channels, height, width
                 num_classes=2,  # Dataset Categories
                 ):
        super(TDiscriminator, self).__init__()

        self.image_size = image_size

        # ---- 1 Patch Embedding ---
        c, h, w = image_size  # image sizes

        ph, pw= patches  # patch sizes

        n, m = h // ph, w // pw
        seq_len = n * m  # number of patches

        # Patch embedding
        self.patch_embedding = nn.Conv2d(in_channels=c, out_channels=d_model, kernel_size=(ph, pw), stride=(ph, pw))

        # Class token
        self.class_embedding = nn.Parameter(torch.zeros(1, 1, d_model))

        # Position embedding
        self.position_embedding = PositionEmbedding(input_seq=(seq_len + 1), d_model=d_model)

        # Transformer
        self.transformer = TransformerEncoder(num_layers=num_layers, d_model=d_model, nhead=num_heads,
                                              d_ff=d_ff, dropout=dropout)

        # Classifier head
        self.norm = nn.LayerNorm(normalized_shape=d_model, eps=1e-6)
        self.mlp = nn.Linear(in_features=d_model, out_features=num_classes)

    def forward(self, x):
        b, c, ph, pw = x.shape

        x = self.patch_embedding(x)
        x = x.flatten(2).transpose(1, 2)

        x = torch.cat((self.class_embedding.expand(b, -1, -1), x), dim=1)

        x = self.position_embedding(x)

        x, scores = self.transformer(x)
        x = self.norm(x)[:, 0]
        x = self.mlp(x)
        return x


if __name__ == '__main__':
    g = TGenerator().to('cuda:3')
    noise = torch.zeros((1, 256)).to('cuda:3')
    fake = g(noise)

    img = torch.zeros((1, 3, 32, 32)).to('cuda:3')
    d = TDiscriminator().to('cuda:3')
    realc = d(img)
    fakec = d(fake)



    pass
