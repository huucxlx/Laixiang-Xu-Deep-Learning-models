import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional

class DiffeomorphicAttention(nn.Module):
    def __init__(self, dim, window_size, num_heads, qkv_bias=True, attn_drop=0., proj_drop=0.):
        super().__init__()
        self.dim = dim
        self.window_size = window_size
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = head_dim ** -0.5

        # 定义MLP生成相对位置偏置
        self.relative_position_mlp = nn.Sequential(
            nn.Linear(2, 64),
            nn.GELU(),
            nn.Linear(64, num_heads)
        )

        # 生成相对位置坐标索引
        coords_h = torch.arange(self.window_size[0], dtype=torch.float32)
        coords_w = torch.arange(self.window_size[1], dtype=torch.float32)
        coords = torch.stack(torch.meshgrid([coords_h, coords_w], indexing="ij"))
        coords_flatten = torch.flatten(coords, 1)
        relative_coords = coords_flatten[:, :, None] - coords_flatten[:, None, :]
        relative_coords = relative_coords.permute(1, 2, 0).contiguous()
        relative_coords[:, :, 0] /= max(self.window_size[0] - 1, 1)
        relative_coords[:, :, 1] /= max(self.window_size[1] - 1, 1)
        self.register_buffer("relative_coords", relative_coords)

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x, mask: Optional[torch.Tensor] = None):
        B_, N, C = x.shape
        qkv = self.qkv(x).reshape(B_, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv.unbind(0)

        q = q * self.scale
        attn = (q @ k.transpose(-2, -1))

        # 动态生成相对位置偏置
        relative_bias = self.relative_position_mlp(self.relative_coords)
        relative_bias = relative_bias.permute(2, 0, 1).contiguous()
        attn = attn + relative_bias.unsqueeze(0)

        if mask is not None:
            nW = mask.shape[0]
            attn = attn.view(B_ // nW, nW, self.num_heads, N, N) + mask.unsqueeze(1).unsqueeze(0)
            attn = attn.view(-1, self.num_heads, N, N)
            attn = self.softmax(attn)
        else:
            attn = self.softmax(attn)

        attn = self.attn_drop(attn)
        x = (attn @ v).transpose(1, 2).reshape(B_, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x