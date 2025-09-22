import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional

class ImplicitNeuralAttention(nn.Module):
    def __init__(self, dim, window_size, num_heads, qkv_bias=True, attn_drop=0., proj_drop=0.):
        super().__init__()
        self.dim = dim
        self.window_size = window_size
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = head_dim ** -0.5

        # 隐式神经网络的参数化位置编码
        self.position_net = nn.Sequential(
            nn.Linear(2, 64),
            nn.GELU(),
            nn.Linear(64, num_heads * 3)  # 生成QKV的隐式参数
        )

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)
        self.softmax = nn.Softmax(dim=-1)

        # 生成相对坐标网格
        coords = torch.stack(torch.meshgrid(
            torch.arange(window_size),
            torch.arange(window_size)), dim=-1).float()
        self.register_buffer("coords", coords)

    def forward(self, x, mask=None):
        B_, N, C = x.shape
        qkv = self.qkv(x).reshape(B_, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv.unbind(0)

        # 隐式位置编码生成注意力参数
        rel_coords = self.coords.view(-1, 2)
        position_params = self.position_net(rel_coords).view(
            N, self.num_heads, 3, -1).permute(2, 0, 1, 3)
        q = q + position_params[0] * self.scale
        k = k + position_params[1] * self.scale

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = self.softmax(attn)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B_, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x