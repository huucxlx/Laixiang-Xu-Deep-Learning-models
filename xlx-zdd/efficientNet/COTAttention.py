import torch
from torch import nn

class CoTAttention(nn.Module):
    def __init__(self, dim=512, kernel_size=3, reduction_ratio=4):
        super().__init__()
        self.dim = dim
        self.kernel_size = kernel_size
        reduction_dim = max(4, dim // reduction_ratio)

        self.static_conv = nn.Sequential(
            nn.Conv2d(dim, dim, kernel_size,
                      padding=kernel_size // 2, groups=dim, bias=False),
            nn.BatchNorm2d(dim),
            nn.ReLU()
        )

        self.key_embed = nn.Sequential(
            nn.Conv2d(dim, reduction_dim, 1),
            nn.BatchNorm2d(reduction_dim),
            nn.ReLU()
        )
        self.value_embed = nn.Conv2d(dim, dim, 1)

        self.attention = nn.Sequential(
            nn.Conv2d(dim + reduction_dim, dim, 1),
            nn.BatchNorm2d(dim),
            nn.Sigmoid()
        )

        self.proj = nn.Conv2d(dim, dim, 1)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out')

    def forward(self, x):
        B, C, H, W = x.shape
        static_context = self.static_conv(x)
        keys = self.key_embed(x)
        values = self.value_embed(x)

        attn = torch.cat([static_context, keys], dim=1)
        attn = self.attention(attn)

        dynamic_context = attn * values
        out = static_context + dynamic_context
        return self.proj(out)