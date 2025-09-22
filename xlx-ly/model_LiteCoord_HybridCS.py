import torch
import torch.nn as nn


class HybridCS(nn.Module):
    """混合通道-空间注意力"""

    def __init__(self, channels, reduction=16):
        super().__init__()
        # 通道注意力
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        self.channel_fc = nn.Sequential(
            nn.Linear(channels, channels // reduction),
            nn.ReLU(inplace=True),
            nn.Linear(channels // reduction, channels),
            nn.Sigmoid()
        )

        # 空间注意力
        self.spatial_conv = nn.Conv2d(2, 1, kernel_size=7, padding=3)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        # 通道注意力
        b, c, _, _ = x.size()
        channel_avg = self.channel_fc(self.avg_pool(x).view(b, c)).view(b, c, 1, 1)
        channel_max = self.channel_fc(self.max_pool(x).view(b, c)).view(b, c, 1, 1)
        channel_out = channel_avg + channel_max

        # 空间注意力
        avg_spatial = torch.mean(x, dim=1, keepdim=True)
        max_spatial, _ = torch.max(x, dim=1, keepdim=True)
        spatial_out = self.sigmoid(self.spatial_conv(torch.cat([avg_spatial, max_spatial], dim=1)))

        return x * channel_out * spatial_out


class LiteCoordAttention(nn.Module):
    """轻量坐标注意力"""

    def __init__(self, channels, reduction=32):
        super().__init__()
        self.pool_h = nn.AdaptiveAvgPool2d((None, 1))
        self.pool_w = nn.AdaptiveAvgPool2d((1, None))
        mid_c = max(8, channels // reduction)

        self.conv1 = nn.Conv2d(channels, mid_c, kernel_size=1)
        self.conv_h = nn.Conv2d(mid_c, channels, kernel_size=1)
        self.conv_w = nn.Conv2d(mid_c, channels, kernel_size=1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        identity = x
        b, c, h, w = x.size()

        x_h = self.pool_h(x)
        x_w = self.pool_w(x).permute(0, 1, 3, 2)

        y = torch.cat([x_h, x_w], dim=2)
        y = self.conv1(y)
        y = torch.relu(y)

        x_h, x_w = torch.split(y, [h, w], dim=2)
        a_h = self.sigmoid(self.conv_h(x_h))
        a_w = self.sigmoid(self.conv_w(x_w.permute(0, 1, 3, 2)))

        return identity * a_h * a_w


class HybridAttentionAlexNet(nn.Module):
    def __init__(self, num_classes=4):
        super().__init__()
        # 特征提取层
        self.features = nn.Sequential(
            # Stage 1: 低层特征提取
            nn.Conv2d(3, 64, kernel_size=11, stride=4, padding=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
            HybridCS(64),  # 优先使用通道-空间注意力

            # Stage 2: 中层特征
            nn.Conv2d(64, 128, kernel_size=5, padding=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
            LiteCoordAttention(128),  # 引入坐标注意力

            # Stage 3-4: 高层特征
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            SmartFusion(256),  # 混合两种注意力
            nn.MaxPool2d(kernel_size=3, stride=2)
        )

        # 分类器
        self.classifier = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(256 * 6 * 6, 1024),
            nn.ReLU(inplace=True),
            nn.Linear(1024, num_classes)
        )

    def forward(self, x):
        x = self.features(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x


class SmartFusion(nn.Module):
    """自适应融合两种注意力"""

    def __init__(self, channels):
        super().__init__()
        self.hybrid = HybridCS(channels)
        self.coord = LiteCoordAttention(channels)
        self.weight = nn.Parameter(torch.tensor([0.6, 0.4]))  # 可学习权重

    def forward(self, x):
        h_out = self.hybrid(x)
        c_out = self.coord(x)
        return self.weight[0] * h_out + self.weight[1] * c_out