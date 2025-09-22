import math
import copy
from functools import partial
from collections import OrderedDict
from typing import Optional, Callable

import torch
import torch.nn as nn
from torch import Tensor
from torch.nn import functional as F


def _make_divisible(ch, divisor=8, min_ch=None):
    if min_ch is None:
        min_ch = divisor
    new_ch = max(min_ch, int(ch + divisor / 2) // divisor * divisor)
    if new_ch < 0.9 * ch:
        new_ch += divisor
    return new_ch


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


class SKConv(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1, M=2, G=32, r=16, L=32):
        super().__init__()
        self.M = M
        self.out_channels = out_channels

        self.convs = nn.ModuleList()
        for i in range(M):
            self.convs.append(
                nn.Sequential(
                    nn.Conv2d(in_channels, out_channels, 3 + i * 2,
                              stride=stride, padding=1 + i, groups=G, bias=False),
                    nn.BatchNorm2d(out_channels),
                    nn.ReLU(inplace=True)
                )
            )

        self.gap = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Conv2d(out_channels, max(out_channels // r, L), 1),
            nn.BatchNorm2d(max(out_channels // r, L)),
            nn.ReLU(inplace=True)
        )

        self.fcs = nn.ModuleList()
        for i in range(M):
            self.fcs.append(nn.Conv2d(max(out_channels // r, L), out_channels, 1))

        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        batch_size = x.shape[0]
        feats = [conv(x) for conv in self.convs]
        feats = torch.stack(feats, dim=1)

        feats_U = torch.sum(feats, dim=1)
        feats_S = self.gap(feats_U)
        feats_Z = self.fc(feats_S)

        attention_vectors = [fc(feats_Z) for fc in self.fcs]
        attention_vectors = torch.stack(attention_vectors, dim=1)
        attention_vectors = self.softmax(attention_vectors)

        feats_V = torch.sum(feats * attention_vectors.unsqueeze(-1).unsqueeze(-1), dim=1)
        return feats_V


class DropPath(nn.Module):
    def __init__(self, drop_prob=None):
        super().__init__()
        self.drop_prob = drop_prob

    def forward(self, x):
        if self.drop_prob == 0. or not self.training:
            return x
        keep_prob = 1 - self.drop_prob
        shape = (x.shape[0],) + (1,) * (x.ndim - 1)
        random_tensor = keep_prob + torch.rand(shape, dtype=x.dtype, device=x.device)
        random_tensor.floor_()
        return x.div(keep_prob) * random_tensor


class ConvBNActivation(nn.Sequential):
    def __init__(self,
                 in_planes: int,
                 out_planes: int,
                 kernel_size: int = 3,
                 stride: int = 1,
                 groups: int = 1,
                 norm_layer: Optional[Callable[..., nn.Module]] = None,
                 activation_layer: Optional[Callable[..., nn.Module]] = None):
        padding = (kernel_size - 1) // 2
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        if activation_layer is None:
            activation_layer = nn.SiLU
        super().__init__(
            nn.Conv2d(in_planes, out_planes, kernel_size, stride,
                      padding, groups=groups, bias=False),
            norm_layer(out_planes),
            activation_layer()
        )


class SqueezeExcitation(nn.Module):
    def __init__(self, input_c: int, expand_c: int, squeeze_factor=4):
        super().__init__()
        squeeze_c = _make_divisible(expand_c // squeeze_factor, 8)
        self.fc1 = nn.Conv2d(expand_c, squeeze_c, 1)
        self.act1 = nn.SiLU()
        self.fc2 = nn.Conv2d(squeeze_c, expand_c, 1)
        self.act2 = nn.Sigmoid()

    def forward(self, x: Tensor) -> Tensor:
        scale = F.adaptive_avg_pool2d(x, (1, 1))
        scale = self.fc1(scale)
        scale = self.act1(scale)
        scale = self.fc2(scale)
        return x * self.act2(scale)


class InvertedResidualConfig:
    def __init__(self,
                 kernel: int,
                 input_c: int,
                 out_c: int,
                 expanded_ratio: int,
                 stride: int,
                 use_se: bool,
                 drop_rate: float,
                 index: str,
                 width_coefficient: float,
                 use_cot: bool = False,
                 use_sk: bool = False):
        self.input_c = self.adjust_channels(input_c, width_coefficient)
        self.kernel = kernel
        self.expanded_c = self.input_c * expanded_ratio
        self.out_c = self.adjust_channels(out_c, width_coefficient)
        self.use_se = use_se
        self.stride = stride
        self.drop_rate = drop_rate
        self.index = index
        self.use_cot = use_cot and (stride == 1)
        self.use_sk = use_sk and (stride == 1)

    @staticmethod
    def adjust_channels(channels: int, width_coefficient: float):
        return _make_divisible(channels * width_coefficient, 8)


class InvertedResidual(nn.Module):
    def __init__(self,
                 cnf: InvertedResidualConfig,
                 norm_layer: Callable[..., nn.Module]):
        super().__init__()
        if cnf.stride not in [1, 2]:
            raise ValueError("illegal stride value.")

        self.use_res_connect = (cnf.stride == 1 and cnf.input_c == cnf.out_c)
        layers = OrderedDict()
        activation = nn.SiLU

        if cnf.expanded_c != cnf.input_c:
            layers["expand"] = ConvBNActivation(cnf.input_c, cnf.expanded_c,
                                                kernel_size=1, norm_layer=norm_layer,
                                                activation_layer=activation)

        if cnf.use_sk:
            layers["dwconv"] = SKConv(cnf.expanded_c, cnf.expanded_c,
                                      stride=cnf.stride, G=cnf.expanded_c)
        else:
            layers["dwconv"] = ConvBNActivation(cnf.expanded_c, cnf.expanded_c,
                                                kernel_size=cnf.kernel, stride=cnf.stride,
                                                groups=cnf.expanded_c, norm_layer=norm_layer,
                                                activation_layer=activation)

        if cnf.use_se:
            layers["se"] = SqueezeExcitation(cnf.input_c, cnf.expanded_c)

        layers["project"] = ConvBNActivation(cnf.expanded_c, cnf.out_c,
                                             kernel_size=1, norm_layer=norm_layer,
                                             activation_layer=nn.Identity)

        self.block = nn.Sequential(layers)
        self.out_channels = cnf.out_c
        self.cot = CoTAttention(cnf.out_c) if cnf.use_cot else nn.Identity()
        self.drop_path = DropPath(cnf.drop_rate) if cnf.drop_rate > 0 else nn.Identity()

    def forward(self, x: Tensor) -> Tensor:
        result = self.block(x)
        result = self.cot(result)
        result = self.drop_path(result)
        if self.use_res_connect:
            result += x
        return result


class SKBlock(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None, groups=32,
                 base_width=64, norm_layer=None, M=2, r=16, L=32):
        super().__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        width = int(planes * (base_width / 64.)) * groups

        self.conv1 = nn.Conv2d(inplanes, width, 1, bias=False)
        self.bn1 = norm_layer(width)
        self.conv2 = SKConv(width, width, stride, M=M, G=groups, r=r, L=L)
        self.conv3 = nn.Conv2d(width, planes * self.expansion, 1, bias=False)
        self.bn3 = norm_layer(planes * self.expansion)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)
        return out


class SKNet(nn.Module):
    def __init__(self, block, layers, num_classes=1000, M=2, r=16, L=32, groups=32,
                 width_per_group=64, norm_layer=None):
        super().__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        self._norm_layer = norm_layer

        self.inplanes = 64
        self.groups = groups
        self.base_width = width_per_group

        self.conv1 = nn.Conv2d(3, self.inplanes, 7, 2, 3, bias=False)
        self.bn1 = norm_layer(self.inplanes)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(3, 2, 1)

        self.layer1 = self._make_layer(block, 64, layers[0], M=M, r=r)
        self.layer2 = self._make_layer(block, 128, layers[1], 2, M, r)
        self.layer3 = self._make_layer(block, 256, layers[2], 2, M, r)
        self.layer4 = self._make_layer(block, 512, layers[3], 2, M, r)

        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(512 * block.expansion, num_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def _make_layer(self, block, planes, blocks, stride=1, M=2, r=16):
        norm_layer = self._norm_layer
        downsample = None

        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion, 1, stride, bias=False),
                norm_layer(planes * block.expansion)
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample, self.groups,
                            self.base_width, norm_layer, M, r))
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes, groups=self.groups,
                                base_width=self.base_width, norm_layer=norm_layer,
                                M=M, r=r))

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)
        return x

    @staticmethod
    def sknet50(num_classes=1000):
        return SKNet(SKBlock, [3, 4, 6, 3], num_classes=num_classes)

    @staticmethod
    def sknet101(num_classes=1000):
        return SKNet(SKBlock, [3, 4, 23, 3], num_classes=num_classes)


class EfficientNet(nn.Module):
    def __init__(self,
                 width_coefficient: float,
                 depth_coefficient: float,
                 num_classes: int = 1000,
                 dropout_rate: float = 0.2,
                 drop_connect_rate: float = 0.2,
                 block: Optional[Callable[..., nn.Module]] = None,
                 norm_layer: Optional[Callable[..., nn.Module]] = None,
                 use_sknet: bool = False):
        super().__init__()

        # SKNet相关初始化
        if use_sknet and block is None:
            block = partial(SKBlock, M=2, r=16, L=32)

        # 默认配置
        default_cnf = [
            [3, 32, 16, 1, 1, True, drop_connect_rate, 1, False, False],
            [3, 16, 24, 6, 2, True, drop_connect_rate, 2, False, False],
            [5, 24, 40, 6, 2, True, drop_connect_rate, 2, True, True],
            [3, 40, 80, 6, 2, True, drop_connect_rate, 3, True, True],
            [5, 80, 112, 6, 1, True, drop_connect_rate, 3, True, True],
            [5, 112, 192, 6, 2, True, drop_connect_rate, 4, True, True],
            [3, 192, 320, 6, 1, True, drop_connect_rate, 1, True, True]
        ]

        def round_repeats(repeats):
            return int(math.ceil(depth_coefficient * repeats))

        # 设置默认块和归一化层
        if block is None:
            block = InvertedResidual
        if norm_layer is None:
            norm_layer = partial(nn.BatchNorm2d, eps=1e-3, momentum=0.1)

        adjust_channels = partial(InvertedResidualConfig.adjust_channels,
                                  width_coefficient=width_coefficient)

        # 构建倒残差设置
        b = 0
        num_blocks = float(sum(round_repeats(i[-3]) for i in default_cnf))
        inverted_residual_setting = []

        for stage, args in enumerate(default_cnf):
            cnf = copy.copy(args)
            for i in range(round_repeats(cnf.pop(-3))):
                if i > 0:
                    cnf[-5] = 1  # 调整stride
                    cnf[1] = cnf[2]  # 调整input channels
                cnf[-4] = args[-4] * b / num_blocks  # 调整drop_connect_rate
                index = f"stage{stage + 1}_block{i + 1}"  # 生成有意义的索引名称
                inverted_residual_setting.append(
                    InvertedResidualConfig(*cnf[:8], width_coefficient, cnf[-2], cnf[-1])
                )
                b += 1

        # 构建网络层
        layers = OrderedDict()

        # 添加stem层
        layers["stem"] = ConvBNActivation(
            3, adjust_channels(32),
            kernel_size=3, stride=2,
            norm_layer=norm_layer
        )

        # 添加倒残差块
        for cnf in inverted_residual_setting:
            layers[cnf.index] = block(cnf, norm_layer)

        # 添加top层
        last_conv_input_c = inverted_residual_setting[-1].out_c
        last_conv_output_c = adjust_channels(1280)
        layers["top"] = ConvBNActivation(
            last_conv_input_c, last_conv_output_c,
            kernel_size=1, norm_layer=norm_layer
        )

        self.features = nn.Sequential(layers)
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.classifier = nn.Sequential(
            nn.Dropout(p=dropout_rate, inplace=True),
            nn.Linear(last_conv_output_c, num_classes)
        )

        # 初始化权重
        self._initialize_weights()

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out")
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.zeros_(m.bias)

    def forward(self, x: Tensor) -> Tensor:
        x = self.features(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x

    # 各种预定义配置
    @staticmethod
    def efficient0(num_classes=1000):
        return EfficientNet(1.0, 1.0, num_classes=num_classes)

    @staticmethod
    def efficient1(num_classes=1000):
        return EfficientNet(1.0, 1.1, num_classes=num_classes)

    @staticmethod
    def efficient2(num_classes=1000):
        return EfficientNet(1.1, 1.2, num_classes=num_classes)

    @staticmethod
    def efficient3(num_classes=1000):
        return EfficientNet(1.2, 1.4, num_classes=num_classes)

    @staticmethod
    def efficient4(num_classes=1000):
        return EfficientNet(1.4, 1.8, num_classes=num_classes)

    @staticmethod
    def efficient5(num_classes=1000):
        return EfficientNet(1.6, 2.2, num_classes=num_classes)

    @staticmethod
    def efficient6(num_classes=1000):
        return EfficientNet(1.8, 2.6, num_classes=num_classes)

    @staticmethod
    def efficient7(num_classes=1000):
        return EfficientNet(2.0, 3.1, num_classes=num_classes)


# 导出模型创建函数
efficient0 = EfficientNet.efficient0
sknet50 = SKNet.sknet50
sknet101 = SKNet.sknet101
__all__ = ['efficient0', 'sknet50', 'sknet101']
