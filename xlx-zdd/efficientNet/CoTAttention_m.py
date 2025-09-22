import math
import copy
from functools import partial
from collections import OrderedDict
from typing import Optional, Callable

import torch
import torch.nn as nn
from torch import Tensor
from torch.nn import functional as F


class CoTAttention(nn.Module):
    """Contextual Transformer Attention Module"""

    def __init__(self, dim=512, kernel_size=3, reduction_ratio=4):
        super().__init__()
        self.dim = dim
        self.kernel_size = kernel_size
        reduction_dim = max(4, dim // reduction_ratio)  # Ensure minimum channels

        # Static context branch (depth-wise conv)
        self.static_conv = nn.Sequential(
            nn.Conv2d(dim, dim, kernel_size,
                      padding=kernel_size // 2, groups=dim, bias=False),
            nn.BatchNorm2d(dim),
            nn.ReLU()
        )

        # Dynamic context branch
        self.key_embed = nn.Sequential(
            nn.Conv2d(dim, reduction_dim, 1),
            nn.BatchNorm2d(reduction_dim),
            nn.ReLU()
        )
        self.value_embed = nn.Conv2d(dim, dim, 1)

        # Attention mechanism
        self.attention = nn.Sequential(
            nn.Conv2d(dim + reduction_dim, dim, 1),
            nn.BatchNorm2d(dim),
            nn.Sigmoid()
        )

        # Output projection
        self.proj = nn.Conv2d(dim, dim, 1)

        # Initialize weights
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out')

    def forward(self, x):
        B, C, H, W = x.shape

        # Static context
        static_context = self.static_conv(x)

        # Dynamic context
        keys = self.key_embed(x)
        values = self.value_embed(x)

        # Attention map
        attn = torch.cat([static_context, keys], dim=1)
        attn = self.attention(attn)

        # Context fusion
        dynamic_context = attn * values

        # Combine contexts
        out = static_context + dynamic_context
        return self.proj(out)


def _make_divisible(ch, divisor=8, min_ch=None):
    if min_ch is None:
        min_ch = divisor
    new_ch = max(min_ch, int(ch + divisor / 2) // divisor * divisor)
    if new_ch < 0.9 * ch:
        new_ch += divisor
    return new_ch


class DropPath(nn.Module):
    """Stochastic depth drop path"""

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
                 use_cot: bool = False):
        self.input_c = self.adjust_channels(input_c, width_coefficient)
        self.kernel = kernel
        self.expanded_c = self.input_c * expanded_ratio
        self.out_c = self.adjust_channels(out_c, width_coefficient)
        self.use_se = use_se
        self.stride = stride
        self.drop_rate = drop_rate
        self.index = index
        self.use_cot = use_cot and (stride == 1)  # Only use CoT when stride=1

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

        # Expand
        if cnf.expanded_c != cnf.input_c:
            layers["expand"] = ConvBNActivation(cnf.input_c, cnf.expanded_c,
                                                kernel_size=1, norm_layer=norm_layer,
                                                activation_layer=activation)

        # Depth-wise
        layers["dwconv"] = ConvBNActivation(cnf.expanded_c, cnf.expanded_c,
                                            kernel_size=cnf.kernel, stride=cnf.stride,
                                            groups=cnf.expanded_c, norm_layer=norm_layer,
                                            activation_layer=activation)

        # Squeeze-and-Excitation
        if cnf.use_se:
            layers["se"] = SqueezeExcitation(cnf.input_c, cnf.expanded_c)

        # Project
        layers["project"] = ConvBNActivation(cnf.expanded_c, cnf.out_c,
                                             kernel_size=1, norm_layer=norm_layer,
                                             activation_layer=nn.Identity)

        self.block = nn.Sequential(layers)
        self.out_channels = cnf.out_c

        # Contextual Transformer Attention
        self.cot = CoTAttention(cnf.out_c) if cnf.use_cot else nn.Identity()

        # Drop path
        self.drop_path = DropPath(cnf.drop_rate) if cnf.drop_rate > 0 else nn.Identity()

    def forward(self, x: Tensor) -> Tensor:
        result = self.block(x)
        result = self.cot(result)  # Apply CoTAttention
        result = self.drop_path(result)
        if self.use_res_connect:
            result += x
        return result


class EfficientNet(nn.Module):
    def __init__(self,
                 width_coefficient: float,
                 depth_coefficient: float,
                 num_classes: int = 1000,
                 dropout_rate: float = 0.2,
                 drop_connect_rate: float = 0.2,
                 block: Optional[Callable[..., nn.Module]] = None,
                 norm_layer: Optional[Callable[..., nn.Module]] = None):
        super().__init__()

        # Default configuration with CoT flags
        default_cnf = [
            # kernel, in_c, out_c, exp_ratio, stride, use_se, drop_rate, repeats, use_cot
            [3, 32, 16, 1, 1, True, drop_connect_rate, 1, False],
            [3, 16, 24, 6, 2, True, drop_connect_rate, 2, False],
            [5, 24, 40, 6, 2, True, drop_connect_rate, 2, True],
            [3, 40, 80, 6, 2, True, drop_connect_rate, 3, True],
            [5, 80, 112, 6, 1, True, drop_connect_rate, 3, True],
            [5, 112, 192, 6, 2, True, drop_connect_rate, 4, True],
            [3, 192, 320, 6, 1, True, drop_connect_rate, 1, True]
        ]

        def round_repeats(repeats):
            return int(math.ceil(depth_coefficient * repeats))

        if block is None:
            block = InvertedResidual
        if norm_layer is None:
            norm_layer = partial(nn.BatchNorm2d, eps=1e-3, momentum=0.1)

        adjust_channels = partial(InvertedResidualConfig.adjust_channels,
                                  width_coefficient=width_coefficient)

        # Build inverted residual blocks
        b = 0
        num_blocks = float(sum(round_repeats(i[-2]) for i in default_cnf))
        inverted_residual_setting = []
        for stage, args in enumerate(default_cnf):
            cnf = copy.copy(args)
            for i in range(round_repeats(cnf.pop(-2))):  # repeats is now at -2
                if i > 0:
                    cnf[-4] = 1  # stride
                    cnf[1] = cnf[2]  # input_c = output_c
                cnf[-3] = args[-3] * b / num_blocks  # update drop rate
                index = f"{stage + 1}{chr(i + 97)}"
                inverted_residual_setting.append(
                    InvertedResidualConfig(*cnf[:7], index, width_coefficient, cnf[-1])
                )
                b += 1

        # Build layers
        layers = OrderedDict()

        # Stem
        layers["stem"] = ConvBNActivation(3, adjust_channels(32),
                                          kernel_size=3, stride=2,
                                          norm_layer=norm_layer)

        # Blocks
        for cnf in inverted_residual_setting:
            layers[cnf.index] = block(cnf, norm_layer)

        # Head
        last_conv_input_c = inverted_residual_setting[-1].out_c
        last_conv_output_c = adjust_channels(1280)
        layers["top"] = ConvBNActivation(last_conv_input_c, last_conv_output_c,
                                         kernel_size=1, norm_layer=norm_layer)

        self.features = nn.Sequential(layers)
        self.avgpool = nn.AdaptiveAvgPool2d(1)

        # Classifier
        self.classifier = nn.Sequential(
            nn.Dropout(p=dropout_rate, inplace=True),
            nn.Linear(last_conv_output_c, num_classes)
        )

        # Initialize weights
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


def efficient0(num_classes=1000):
    return EfficientNet(1.0, 1.0, num_classes=num_classes)


def efficient1(num_classes=1000):
    return EfficientNet(1.0, 1.1, num_classes=num_classes)


def efficient2(num_classes=1000):
    return EfficientNet(1.1, 1.2, num_classes=num_classes)


def efficient3(num_classes=1000):
    return EfficientNet(1.2, 1.4, num_classes=num_classes)


def efficient4(num_classes=1000):
    return EfficientNet(1.4, 1.8, num_classes=num_classes)


def efficient5(num_classes=1000):
    return EfficientNet(1.6, 2.2, num_classes=num_classes)


def efficient6(num_classes=1000):
    return EfficientNet(1.8, 2.6, num_classes=num_classes)


def efficient7(num_classes=1000):
    return EfficientNet(2.0, 3.1, num_classes=num_classes)