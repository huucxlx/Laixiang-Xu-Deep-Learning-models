import torch
import torch.nn as nn

class CDFA(nn.Module):
    def __init__(self, in_channels, reduction=16, kernel_size=3):
        super().__init__()
        self.in_channels = in_channels

        # 通道对比度分支
        self.reduction = max(1, in_channels // reduction)  # 新增保护逻辑
        self.channel_att = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(in_channels, in_channels // reduction, 1, bias=False),
            nn.ReLU(),
            nn.Conv2d(in_channels // reduction, in_channels, 1, bias=False),
            nn.Sigmoid()
        )

        # 空间对比度分支
        self.spatial_att = nn.Sequential(
            nn.Conv2d(in_channels, 1, kernel_size, padding=kernel_size // 2, bias=False),
            nn.Sigmoid()
        )
        self.alpha = nn.Parameter(torch.tensor(0.1))
        self.beta = nn.Parameter(torch.tensor(0.1))

    def forward(self, x):
        # 通道注意力加权
        channel_att = self.channel_att(x)
        x_channel = x * channel_att

        # 空间注意力加权
        spatial_att = self.spatial_att(x)
        x_spatial = x * spatial_att

        # 特征聚合（残差连接）
        # return x + x_channel + x_spatial  # 增强特征对比度
        return x + self.alpha * x_channel + self.beta * x_spatial  # 增强特征对比度