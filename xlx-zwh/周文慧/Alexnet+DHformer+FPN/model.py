import torch
import torch.nn as nn
from dhformer import DHformer


class AlexNetFPN(nn.Module):
    def __init__(self, num_classes=1000, init_weights=False):
        super(AlexNetFPN, self).__init__()
        # Feature extraction layers (same as original AlexNet)
        self.conv1 = nn.Sequential(
            nn.Conv2d(3, 48, kernel_size=11, stride=4, padding=2),
            nn.ReLU(inplace=True),
            nn.GroupNorm(num_channels=48, num_groups=6),
            nn.MaxPool2d(kernel_size=3, stride=2)
        )

        self.conv2 = nn.Sequential(
            nn.Conv2d(48, 128, kernel_size=5, padding=2),
            nn.ReLU(inplace=True),
            nn.GroupNorm(num_channels=128, num_groups=16),
            nn.MaxPool2d(kernel_size=3, stride=2)
        )

        self.conv3 = nn.Sequential(
            nn.Conv2d(128, 192, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.GroupNorm(num_channels=192, num_groups=16),
            DHformer(192, out_channels=192)
        )

        self.conv4 = nn.Sequential(
            nn.Conv2d(192, 128, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.GroupNorm(num_channels=128, num_groups=16),
            DHformer(128, out_channels=128),
            nn.MaxPool2d(kernel_size=3, stride=2)
        )

        # FPN lateral connections and top-down pathway
        self.lateral4 = nn.Conv2d(128, 256, kernel_size=1)  # C4
        self.lateral3 = nn.Conv2d(192, 256, kernel_size=1)  # C3
        self.lateral2 = nn.Conv2d(128, 256, kernel_size=1)  # C2

        self.smooth3 = nn.Conv2d(256, 256, kernel_size=3, padding=1)
        self.smooth2 = nn.Conv2d(256, 256, kernel_size=3, padding=1)

        # Original classifier
        self.classifier = nn.Sequential(
            nn.Dropout(p=0.5),
            nn.Linear(128 * 6 * 6, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.5),
            nn.Linear(4096, 4096),
            nn.ReLU(inplace=True),
            nn.Linear(4096, num_classes),
        )

        if init_weights:
            self._initialize_weights()

    def forward(self, x):
        # Bottom-up pathway
        c1 = self.conv1(x)  # [48, 55, 55]
        c2 = self.conv2(c1)  # [128, 27, 27]
        c3 = self.conv3(c2)  # [192, 13, 13]
        c4 = self.conv4(c3)  # [128, 6, 6]

        # Top-down pathway
        p4 = self.lateral4(c4)  # [256, 6, 6]

        # 修改1：在上采样时指定目标尺寸
        p3_up = nn.functional.interpolate(p4, size=c3.shape[2:], mode='nearest')  # 上采样到c3的尺寸
        p3 = self.lateral3(c3) + p3_up
        p3 = self.smooth3(p3)  # [256, 13, 13]

        # 修改2：同样处理p2的上采样
        p2_up = nn.functional.interpolate(p3, size=c2.shape[2:], mode='nearest')  # 上采样到c2的尺寸
        p2 = self.lateral2(c2) + p2_up
        p2 = self.smooth2(p2)  # [256, 27, 27]

        # 分类分支保持不变
        x = torch.flatten(c4, start_dim=1)
        x = self.classifier(x)

        if self.training:  # 训练时只返回分类输出
            return x
        else:  # 验证/测试时返回完整信息
            return x, {'p2': p2, 'p3': p3, 'p4': p4}

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)


"""net = AlexNetFPN()
print(net)"""
