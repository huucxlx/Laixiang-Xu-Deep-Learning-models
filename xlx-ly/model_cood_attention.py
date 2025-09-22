import torch
import torch.nn as nn
import torch.nn.functional as F

class CoordAtt(nn.Module):
    """Coordinate Attention 机制 (来自论文: Coordinate Attention for Efficient Mobile Network Design)"""
    def __init__(self, in_channels, reduction=32):
        super(CoordAtt, self).__init__()
        self.pool_h = nn.AdaptiveAvgPool2d((None, 1))  # 高度方向的全局池化
        self.pool_w = nn.AdaptiveAvgPool2d((1, None))  # 宽度方向的全局池化

        mid_channels = max(8, in_channels // reduction)  # 确保mid_channels不小于8

        self.conv1 = nn.Conv2d(in_channels, mid_channels, kernel_size=1, stride=1, padding=0)
        self.bn1 = nn.BatchNorm2d(mid_channels)
        self.conv_h = nn.Conv2d(mid_channels, in_channels, kernel_size=1, stride=1, padding=0)
        self.conv_w = nn.Conv2d(mid_channels, in_channels, kernel_size=1, stride=1, padding=0)

    def forward(self, x):
        identity = x

        # 高度和宽度方向的注意力
        n, c, h, w = x.size()
        x_h = self.pool_h(x)  # [n, c, h, 1]
        x_w = self.pool_w(x).permute(0, 1, 3, 2)  # [n, c, w, 1]

        # 合并特征并卷积
        y = torch.cat([x_h, x_w], dim=2)  # [n, c, h + w, 1]
        y = self.conv1(y)
        y = self.bn1(y)
        y = F.relu(y)

        # 分离高度和宽度注意力
        x_h, x_w = torch.split(y, [h, w], dim=2)
        x_w = x_w.permute(0, 1, 3, 2)  # [n, c, 1, w]

        # 生成注意力权重
        att_h = torch.sigmoid(self.conv_h(x_h))  # [n, c, h, 1]
        att_w = torch.sigmoid(self.conv_w(x_w))  # [n, c, 1, w]

        # 应用注意力
        out = identity * att_h * att_w
        return out

class LicensePlateAlexNet(nn.Module):
    def __init__(self, num_classes=4, init_weights=True):
        super(LicensePlateAlexNet, self).__init__()
        # 特征提取网络
        self.features = nn.Sequential(
            # 第一层卷积
            nn.Conv2d(3, 64, kernel_size=11, stride=4, padding=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
            CoordAtt(64),  # 在第一层后加入CoordAtt

            # 第二层卷积
            nn.Conv2d(64, 128, kernel_size=5, padding=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
            CoordAtt(128),  # 在第二层后加入CoordAtt

            # 第三层卷积
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            CoordAtt(256),  # 在第三层后加入CoordAtt

            # 第四层卷积
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
        )

        # 分类器
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
    model = LicensePlateAlexNet(num_classes=4)
    dummy_input = torch.randn(1, 3, 224, 224)
    output = model(dummy_input)
    print(f"模型输出形状: {output.shape}")  # 预期输出: [1, 4]