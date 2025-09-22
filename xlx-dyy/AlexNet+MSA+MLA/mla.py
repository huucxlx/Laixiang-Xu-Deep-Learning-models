import torch
import torch.nn as nn
import torch.nn.functional as F


class MultiHeadLatentAttention(nn.Module):
    def __init__(self, in_channels, num_heads=4, latent_dim=32):
        super().__init__()
        assert in_channels % num_heads == 0, "in_channels must be divisible by num_heads"

        self.num_heads = num_heads
        self.latent_dim = latent_dim
        self.head_dim = in_channels // num_heads

        # 低秩投影矩阵（无bias）
        self.q_proj = nn.Conv2d(in_channels, in_channels, kernel_size=1, bias=False)
        self.kv_proj = nn.Conv2d(in_channels, 2 * in_channels, kernel_size=1, bias=False)
        self.latent_proj = nn.Linear(self.head_dim, latent_dim, bias=False)

        # 输出投影
        self.out_proj = nn.Conv2d(in_channels, in_channels, kernel_size=1, bias=False)

        # 缩放因子
        self.scale = (latent_dim) ** -0.5

    def forward(self, x):
        B, C, H, W = x.shape

        # 1. 生成Q/K/V (通过1x1卷积)
        q = self.q_proj(x)  # [B, C, H, W]
        kv = self.kv_proj(x)  # [B, 2*C, H, W]
        k, v = torch.split(kv, C, dim=1)

        # 2. 多头分割
        q = q.view(B, self.num_heads, self.head_dim, H * W).permute(0, 1, 3, 2)  # [B, h, HW, d]
        k = k.view(B, self.num_heads, self.head_dim, H * W)  # [B, h, d, HW]
        v = v.view(B, self.num_heads, self.head_dim, H * W)  # [B, h, d, HW]

        # 3. 低秩注意力计算
        latent_q = self.latent_proj(q)  # [B, h, HW, latent_dim]
        latent_k = self.latent_proj(k.permute(0, 1, 3, 2))  # [B, h, HW, latent_dim]

        attn = torch.matmul(latent_q, latent_k.transpose(-2, -1)) * self.scale
        attn = F.softmax(attn, dim=-1)

        # 4. 特征聚合
        out = torch.matmul(attn, v.permute(0, 1, 3, 2))  # [B, h, HW, d]
        out = out.permute(0, 1, 3, 2).reshape(B, C, H, W)

        # 5. 残差连接
        return self.out_proj(out) + x
