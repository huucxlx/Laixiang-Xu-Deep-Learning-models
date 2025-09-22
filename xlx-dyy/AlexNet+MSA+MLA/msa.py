import torch
import torch.nn as nn


class MultiHeadSelfAttention(nn.Module):
    def __init__(self, embed_dim, num_heads=4, dropout=0.1):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        assert self.head_dim * num_heads == embed_dim, "embed_dim必须能被num_heads整除"

        # 投影层
        self.norm = nn.LayerNorm(embed_dim)  # 前置归一化
        self.qkv = nn.Linear(embed_dim, 3 * embed_dim)
        self.proj = nn.Linear(embed_dim, embed_dim)

        self.scale = 1.0 / (self.head_dim ** 0.5)
        self.attn_dropout = nn.Dropout(dropout)
        self.proj_dropout = nn.Dropout(dropout)

    def forward(self, x):
        B, C, H, W = x.shape
        N = H * W

        # 通道维度归一化 (避免inplace问题)
        x = x.permute(0, 2, 3, 1)  # [B, H, W, C]
        x = self.norm(x)
        x = x.permute(0, 3, 1, 2)  # [B, C, H, W]

        # 生成Q/K/V
        x_seq = x.reshape(B, C, N).permute(0, 2, 1)  # [B, N, C]
        qkv = self.qkv(x_seq)  # [B, N, 3*C]
        qkv = qkv.reshape(B, N, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        q, k, v = qkv.unbind(0)  # 每个形状为 [B, heads, N, head_dim]

        # 注意力计算
        attn = (q @ k.transpose(-2, -1)) * self.scale  # [B, heads, N, N]
        attn = attn.softmax(dim=-1)
        attn = self.attn_dropout(attn)

        # 输出投影
        x = (attn @ v)  # [B, heads, N, head_dim]
        x = x.transpose(1, 2).reshape(B, N, C)  # [B, N, C]
        x = self.proj(x)
        x = self.proj_dropout(x)
        x = x.permute(0, 2, 1).view(B, C, H, W)  # 恢复空间维度
        return x