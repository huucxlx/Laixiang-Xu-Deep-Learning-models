import torch
from torch import nn
import os

def _make_divisible(ch, divisor=8, min_ch=None):#给通道数调小一点，减小计算量（减小后大概增快了1.5倍）
    if min_ch is None:
        min_ch = divisor
    new_ch = max(min_ch, int(ch + divisor / 2) // divisor * divisor)
    if new_ch < 0.9 * ch:
        new_ch += divisor
    return new_ch

class ConvBNReLU(nn.Sequential):#MobileNet的激活函数（后续调整参数进行优化）
    def __init__(self, in_channel, out_channel, kernel_size=3, stride=1, groups=1):
        padding = (kernel_size - 1) // 2
        super().__init__(
            nn.Conv2d(in_channel, out_channel, kernel_size, stride, padding, groups=groups, bias=False),
            nn.BatchNorm2d(out_channel),
            nn.ReLU6(inplace=True)
        )

class PSA(nn.Module):
    def __init__(self, c1, c2):
        super().__init__()
        self.conv1 = ConvBNReLU(c1, c2, 3, 1)
        self.conv2 = ConvBNReLU(c2, c2, 3, 1)

    def forward(self, x):
        return self.conv2(self.conv1(x))

class C2PSA(nn.Module):
    def __init__(self, c1, c2):
        super().__init__()
        assert c2 % 2 == 0, "c2 must be even"
        self.conv1 = ConvBNReLU(c1, c2 // 2, 1)
        self.conv2 = ConvBNReLU(c1, c2 // 2, 1)
        self.psa = PSA(c2 // 2, c2 // 2)
        self.conv3 = ConvBNReLU(c2, c2, 1)

    def forward(self, x):
        x1 = self.psa(self.conv1(x))
        x2 = self.conv2(x)
        return self.conv3(torch.cat((x1, x2), dim=1))
#MobileNet的到残差模块（3.17——跳跃连接，3.20——扩展卷积）
class InvertedResidual(nn.Module):
    def __init__(self, in_channel, out_channel, stride, expand_ratio):
        super().__init__()
        hidden_channel = in_channel * expand_ratio
        self.use_shortcut = stride == 1 and in_channel == out_channel

        layers = []
        if expand_ratio != 1:
            layers.append(ConvBNReLU(in_channel, hidden_channel, kernel_size=1))
        layers.extend([
            ConvBNReLU(hidden_channel, hidden_channel, stride=stride, groups=hidden_channel),
            nn.Conv2d(hidden_channel, out_channel, 1, bias=False),
            nn.BatchNorm2d(out_channel),
        ])
        self.conv = nn.Sequential(*layers)

    def forward(self, x):
        return x + self.conv(x) if self.use_shortcut else self.conv(x)

class MobileNetV2(nn.Module):
    def __init__(self, num_classes=1000, alpha=1.0, round_nearest=8):
        super().__init__()
        input_channel = _make_divisible(32 * alpha, round_nearest)
        last_channel = _make_divisible(1280 * alpha, round_nearest)
        inverted_residual_setting = [
            [1, 16, 1, 1], [6, 24, 2, 2], [6, 32, 3, 2],
            [6, 64, 4, 2], [6, 96, 3, 1], [6, 160, 3, 2], [6, 320, 1, 1]
        ]

        features = [ConvBNReLU(3, input_channel, stride=2)]
        for t, c, n, s in inverted_residual_setting:
            output_channel = _make_divisible(c * alpha, round_nearest)
            for i in range(n):
                stride = s if i == 0 else 1
                features.append(InvertedResidual(input_channel, output_channel, stride, t))
                input_channel = output_channel
        features.extend([      #插到到残差结构后面
            C2PSA(input_channel, last_channel),
            ConvBNReLU(last_channel, last_channel, 1),
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten()
        ])
        self.features = nn.Sequential(*features)
        self.classifier = nn.Sequential(
            nn.Dropout(0.2),
            nn.Linear(last_channel, num_classes)
        )

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out')
                if m.bias is not None: nn.init.zeros_(m.bias)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.zeros_(m.bias)

    def forward(self, x):
        x = self.features(x)
        return self.classifier(x)