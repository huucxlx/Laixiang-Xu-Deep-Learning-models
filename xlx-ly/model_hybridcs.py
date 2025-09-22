import torch.nn as nn
import torch


class HybridCS(nn.Module):
    """HybridCS Attention Mechanism"""

    def __init__(self, channels, reduction=16):
        super(HybridCS, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)

        # Channel attention
        self.channel_fc = nn.Sequential(
            nn.Linear(channels, channels // reduction),
            nn.ReLU(inplace=True),
            nn.Linear(channels // reduction, channels),
            nn.Sigmoid()
        )

        # Spatial attention
        self.spatial_conv = nn.Conv2d(2, 1, kernel_size=7, padding=3)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        # Channel attention
        b, c, _, _ = x.size()
        avg_out = self.avg_pool(x).view(b, c)
        max_out = self.max_pool(x).view(b, c)

        channel_avg = self.channel_fc(avg_out).view(b, c, 1, 1)
        channel_max = self.channel_fc(max_out).view(b, c, 1, 1)
        channel_out = channel_avg + channel_max

        # Spatial attention
        avg_pool = torch.mean(x, dim=1, keepdim=True)
        max_pool, _ = torch.max(x, dim=1, keepdim=True)
        spatial_concat = torch.cat([avg_pool, max_pool], dim=1)
        spatial_out = self.sigmoid(self.spatial_conv(spatial_concat))

        # Hybrid attention
        out = x * channel_out * spatial_out
        return out


class LicensePlateAlexNetHybridCS(nn.Module):
    def __init__(self, num_classes=4, init_weights=True):
        super(LicensePlateAlexNetHybridCS, self).__init__()
        self.features = nn.Sequential(
            # 第一层卷积
            nn.Conv2d(3, 64, kernel_size=11, stride=4, padding=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),

            # 在第一层后添加HybridCS
            HybridCS(64),

            # 第二层卷积
            nn.Conv2d(64, 128, kernel_size=5, padding=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),

            # 在第二层后添加HybridCS
            HybridCS(128),

            # 第三层卷积
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),

            # 第四层卷积
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),

            # 在最后一层卷积后添加HybridCS
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


# 使用示例
if __name__ == "__main__":
    model = LicensePlateAlexNetHybridCS(num_classes=4)
    dummy_input = torch.randn(1, 3, 224, 224)
    output = model(dummy_input)
    print(f"模型输出形状: {output.shape}")