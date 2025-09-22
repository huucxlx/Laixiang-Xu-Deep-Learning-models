import torch
import torch.nn as nn


class CoodAttention(nn.Module):
    """Coordinate Attention Module"""

    def __init__(self, in_channels, reduction=32):
        super(CoodAttention, self).__init__()
        self.pool_h = nn.AdaptiveAvgPool2d((None, 1))
        self.pool_w = nn.AdaptiveAvgPool2d((1, None))

        mid_channels = max(in_channels // reduction, 8)

        self.conv1 = nn.Conv2d(in_channels, mid_channels, kernel_size=1, stride=1, padding=0)
        self.bn1 = nn.BatchNorm2d(mid_channels)
        self.conv_h = nn.Conv2d(mid_channels, in_channels, kernel_size=1, stride=1, padding=0)
        self.conv_w = nn.Conv2d(mid_channels, in_channels, kernel_size=1, stride=1, padding=0)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        identity = x

        n, c, h, w = x.size()
        x_h = self.pool_h(x)
        x_w = self.pool_w(x).permute(0, 1, 3, 2)

        y = torch.cat([x_h, x_w], dim=2)
        y = self.conv1(y)
        y = self.bn1(y)
        y = torch.relu(y)

        x_h, x_w = torch.split(y, [h, w], dim=2)
        x_w = x_w.permute(0, 1, 3, 2)

        a_h = self.sigmoid(self.conv_h(x_h))
        a_w = self.sigmoid(self.conv_w(x_w))

        out = identity * a_h * a_w
        return out


class HybridCS(nn.Module):
    """Hybrid Channel-Spatial Attention Module"""

    def __init__(self, channels, reduction=16):
        super(HybridCS, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)

        self.channel_fc = nn.Sequential(
            nn.Linear(channels, channels // reduction),
            nn.ReLU(inplace=True),
            nn.Linear(channels // reduction, channels),
            nn.Sigmoid()
        )

        self.spatial_conv = nn.Conv2d(2, 1, kernel_size=7, padding=3)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        b, c, _, _ = x.size()
        avg_out = self.avg_pool(x).view(b, c)
        max_out = self.max_pool(x).view(b, c)

        channel_avg = self.channel_fc(avg_out).view(b, c, 1, 1)
        channel_max = self.channel_fc(max_out).view(b, c, 1, 1)
        channel_out = channel_avg + channel_max

        avg_pool = torch.mean(x, dim=1, keepdim=True)
        max_pool, _ = torch.max(x, dim=1, keepdim=True)
        spatial_concat = torch.cat([avg_pool, max_pool], dim=1)
        spatial_out = self.sigmoid(self.spatial_conv(spatial_concat))

        out = x * channel_out * spatial_out
        return out


class DualAttentionAlexNet(nn.Module):
    """AlexNet with Dual Attention Mechanisms (CoodAttention + HybridCS)"""

    def __init__(self, num_classes=4, init_weights=True):
        super(DualAttentionAlexNet, self).__init__()

        self.features = nn.Sequential(
            # Block 1
            nn.Conv2d(3, 64, kernel_size=11, stride=4, padding=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
            CoodAttention(64),

            # Block 2
            nn.Conv2d(64, 128, kernel_size=5, padding=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
            HybridCS(128),

            # Block 3
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            CoodAttention(256),

            # Block 4
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
            HybridCS(256),
        )

        self.classifier = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(256 * 6 * 6, 1024),
            nn.ReLU(inplace=True),
            nn.Linear(1024, num_classes),
        )

        if init_weights:
            self._initialize_weights()

    def forward(self, x):
        x = self.features(x)
        x = torch.flatten(x, start_dim=1)
        x = self.classifier(x)
        return x

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)


# 测试代码
if __name__ == "__main__":
    model = DualAttentionAlexNet(num_classes=4)
    dummy_input = torch.randn(1, 3, 224, 224)
    output = model(dummy_input)
    print(f"模型输出形状: {output.shape}")  # 预期输出: torch.Size([1, 4])