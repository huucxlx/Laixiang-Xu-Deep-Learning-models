import torch
import torch.nn as nn
from einops import rearrange


class DHformer(nn.Module):
    def __init__(self, in_channels, out_channels, heads=4, dim_head=64):
        super().__init__()
        # 深度可分离卷积
        self.depthwise_conv = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=1, groups=in_channels),
            nn.BatchNorm2d(in_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels, out_channels, kernel_size=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

        # Transformer 自注意力
        self.attention = nn.MultiheadAttention(embed_dim=out_channels, num_heads=heads)
        self.norm = nn.LayerNorm(out_channels)
        self.feed_forward = nn.Sequential(
            nn.Linear(out_channels, out_channels * 4),
            nn.ReLU(),
            nn.Linear(out_channels * 4, out_channels)
        )

    def forward(self, x):
        # 深度可分离卷积分支
        conv_out = self.depthwise_conv(x)

        # Transformer 分支
        b, c, h, w = conv_out.shape
        x_flat = rearrange(conv_out, 'b c h w -> (h w) b c')  # 转换为序列
        attn_out, _ = self.attention(x_flat, x_flat, x_flat)  # 自注意力
        attn_out = self.norm(attn_out + x_flat)  # 残差连接 + 层归一化
        ff_out = self.feed_forward(attn_out)
        ff_out = rearrange(ff_out, '(h w) b c -> b c h w', h=h, w=w)  # 恢复形状

        return ff_out
