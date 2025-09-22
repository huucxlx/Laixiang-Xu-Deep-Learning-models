import torch
import torch.nn as nn


class LiteCoordAttention(nn.Module):
    """轻量版坐标注意力（修正了AdaptiveAvgPool2d拼写错误）"""

    def __init__(self, channels, reduction=32):
        super().__init__()
        self.pool_h = nn.AdaptiveAvgPool2d((None, 1))  # 修正拼写
        self.pool_w = nn.AdaptiveAvgPool2d((1, None))

        mid_c = max(8, channels // reduction)

        self.conv1 = nn.Conv2d(channels, mid_c, kernel_size=1)
        self.conv_h = nn.Conv2d(mid_c, channels, kernel_size=1)
        self.conv_w = nn.Conv2d(mid_c, channels, kernel_size=1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        identity = x
        b, c, h, w = x.size()

        x_h = self.pool_h(x)  # [b,c,h,1]
        x_w = self.pool_w(x).permute(0, 1, 3, 2)  # [b,c,w,1]

        y = torch.cat([x_h, x_w], dim=2)
        y = self.conv1(y)
        y = torch.relu(y)

        x_h, x_w = torch.split(y, [h, w], dim=2)
        x_w = x_w.permute(0, 1, 3, 2)

        a_h = self.sigmoid(self.conv_h(x_h))
        a_w = self.sigmoid(self.conv_w(x_w))

        return identity * a_h * a_w


class  LicensePlateAlexNet(nn.Module):
    def __init__(self, num_classes=4, init_weights=True):
        super().__init__()
        self.features = nn.Sequential(
            # Block 1
            nn.Conv2d(3, 64, kernel_size=11, stride=4, padding=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
            LiteCoordAttention(64, reduction=16),  # 浅层注意力

            # Block 2
            nn.Conv2d(64, 128, kernel_size=5, padding=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
            LiteCoordAttention(128, reduction=32),

            # Block 3-4
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            LiteCoordAttention(256, reduction=32),
            nn.MaxPool2d(kernel_size=3, stride=2))

        self.classifier = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(256 * 6 * 6, 1024),
            nn.ReLU(inplace=True),
            nn.Linear(1024, num_classes))

        if init_weights:
            self._initialize_weights()

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        x = self.features(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x


# 测试
if __name__ == "__main__":
    model = LicensePlateAlexNet(num_classes=4)
    dummy_input = torch.randn(1, 3, 224, 224)
    output = model(dummy_input)
    print(f"输出形状: {output.shape}")  # 应为 [1, 4]