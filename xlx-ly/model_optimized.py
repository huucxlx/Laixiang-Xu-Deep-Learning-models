import torch.nn as nn
import torch


class LicensePlateAlexNet(nn.Module):
    def __init__(self, num_classes=4, init_weights=True):  # 假设车牌有4个类别
        super(LicensePlateAlexNet, self).__init__()
        # 简化后的特征提取网络
        self.features = nn.Sequential(
            # 第一层卷积 (输入3通道，输出64通道)
            nn.Conv2d(3, 64, kernel_size=11, stride=4, padding=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),

            # 第二层卷积 (64 -> 128)
            nn.Conv2d(64, 128, kernel_size=5, padding=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),

            # 第三层卷积 (128 -> 256)
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),

            # 第四层卷积 (256 -> 256)
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
        )

        # 简化后的分类器
        self.classifier = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(256 * 6 * 6, 1024),  # 减少全连接层维度
            nn.ReLU(inplace=True),
            nn.Linear(1024, num_classes),  # 直接输出到类别数
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
    model = LicensePlateAlexNet(num_classes=4)  # 替换为你的实际类别数
    dummy_input = torch.randn(1, 3, 224, 224)  # 模拟输入图像
    output = model(dummy_input)
    print(f"模型输出形状: {output.shape}")  # 应为 [1, num_classes]