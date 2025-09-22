import math
from functools import partial
from collections import OrderedDict
from typing import Optional, Callable

import torch
import torch.nn as nn
from torch import Tensor
import torch.nn.functional as F
from thop import profile
import logging

# 禁用thop的全部日志输出
logging.getLogger('thop').disabled = True


class FLOPsCounter:
    _calculated = False  # 类变量实现单例模式
    _flops = None
    _params = None

    @classmethod
    def calculate(cls, model, input_tensor):
        if not cls._calculated:
            cls._flops, cls._params = profile(model, inputs=(input_tensor,))
            cls._calculated = True
        return cls._flops, cls._params


def _make_divisible(ch, divisor=8, min_ch=None):
    if min_ch is None:
        min_ch = divisor
    new_ch = max(min_ch, int(ch + divisor / 2) // divisor * divisor)
    if new_ch < 0.9 * ch:
        new_ch += divisor
    return new_ch


class GeM(nn.Module):
    def __init__(self, p=3, eps=1e-6):
        super(GeM, self).__init__()
        self.p = nn.Parameter(torch.ones(1) * p)
        self.eps = eps

    def forward(self, x):
        return self.gem(x, p=self.p, eps=self.eps)

    def gem(self, x, p=3, eps=1e-6):
        return F.avg_pool2d(x.clamp(min=eps).pow(p), (x.size(-2), x.size(-1))).pow(1. / p)


class SKBlock(nn.Module):
    def __init__(self, channels, stride=1, kernels=[3, 5], reduction=4):
        super().__init__()
        self.M = len(kernels)
        self.stride = stride

        self.convs = nn.ModuleList()
        for k in kernels:
            padding = (k - 1) // 2
            self.convs.append(
                nn.Sequential(
                    nn.Conv2d(channels, channels, kernel_size=k,
                              stride=stride, padding=padding,
                              groups=channels, bias=False),
                    nn.BatchNorm2d(channels),
                    nn.SiLU(inplace=False)
                )
            )

        self.gap = nn.AdaptiveAvgPool2d(1)
        # 动态调整压缩率
        reduction_dim = max(channels // reduction, 4)
        self.fc = nn.Sequential(
            nn.Linear(channels, reduction_dim),
            nn.SiLU(inplace=False),
            nn.Linear(reduction_dim, channels * self.M)
        )
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        batch_size = x.size(0)

        feats = [conv(x) for conv in self.convs]
        feats = torch.stack(feats, dim=1)

        attn = self.gap(x).view(batch_size, -1)
        attn = self.fc(attn).view(batch_size, self.M, -1)
        attn = self.softmax(attn)

        attn = attn.unsqueeze(-1).unsqueeze(-1)
        feats = (feats * attn).sum(dim=1)
        return feats


class InvertedResidualConfig:
    def __init__(self, kernel, input_c, out_c, expanded_ratio, stride, index, width_coefficient):
        self.input_c = self.adjust_channels(input_c, width_coefficient)
        self.kernel = kernel
        self.expanded_c = self.input_c * expanded_ratio
        self.out_c = self.adjust_channels(out_c, width_coefficient)
        self.stride = stride
        self.index = index

    @staticmethod
    def adjust_channels(channels, width_coefficient):
        return _make_divisible(channels * width_coefficient, 8)


class InvertedResidual(nn.Module):
    def __init__(self, cnf, norm_layer, use_skblock=True, reduction=4):
        super().__init__()
        self.use_res_connect = (cnf.stride == 1 and cnf.input_c == cnf.out_c)
        layers = OrderedDict()
        activation = nn.SiLU(inplace=False)

        if cnf.expanded_c != cnf.input_c:
            layers["expand"] = nn.Sequential(
                nn.Conv2d(cnf.input_c, cnf.expanded_c, 1, bias=False),
                norm_layer(cnf.expanded_c),
                activation
            )

        if use_skblock:
            layers["dwconv"] = SKBlock(
                channels=cnf.expanded_c,
                stride=cnf.stride,
                kernels=[3, 5],
                reduction=reduction
            )
        else:
            padding = (cnf.kernel - 1) // 2
            layers["dwconv"] = nn.Sequential(
                nn.Conv2d(cnf.expanded_c, cnf.expanded_c, kernel_size=cnf.kernel,
                          stride=cnf.stride, padding=padding,
                          groups=cnf.expanded_c, bias=False),
                norm_layer(cnf.expanded_c),
                activation
            )

        layers["project"] = nn.Sequential(
            nn.Conv2d(cnf.expanded_c, cnf.out_c, 1, bias=False),
            norm_layer(cnf.out_c)
        )

        self.block = nn.Sequential(layers)

    def forward(self, x):
        res = self.block(x)
        if self.use_res_connect:
            res += x
        return res


class EfficientNet(nn.Module):
    def __init__(self, width_coefficient, depth_coefficient, num_classes=1000,
                 dropout_rate=0.2, block=None, norm_layer=None):
        super().__init__()

        default_cnf = [[3, 32, 16, 1, 1, 1],
                       [3, 16, 24, 6, 2, 2],
                       [5, 24, 40, 6, 2, 2],
                       [3, 40, 80, 6, 2, 3],
                       [5, 80, 112, 6, 1, 3],
                       [5, 112, 192, 6, 2, 4],
                       [3, 192, 320, 6, 1, 1]]

        def round_repeats(repeats):
            return int(math.ceil(depth_coefficient * repeats))

        if block is None:
            block = InvertedResidual

        if norm_layer is None:
            norm_layer = partial(nn.BatchNorm2d, eps=1e-3, momentum=0.1)

        adjust_channels = partial(InvertedResidualConfig.adjust_channels,
                                  width_coefficient=width_coefficient)

        bneck_conf = partial(InvertedResidualConfig,
                             width_coefficient=width_coefficient)

        inverted_residual_setting = []
        for stage, args in enumerate(default_cnf):
            kernel, input_c, out_c, exp_ratio, stride, repeats = args
            for i in range(round_repeats(repeats)):
                if i > 0:
                    stride = 1
                    input_c = out_c
                index = f"{stage + 1}{chr(97 + i)}"
                cnf = [kernel, input_c, out_c, exp_ratio, stride, index]
                inverted_residual_setting.append(bneck_conf(*cnf))

        layers = OrderedDict()

        layers["stem_conv"] = nn.Sequential(
            nn.Conv2d(3, adjust_channels(32), kernel_size=3, stride=2, padding=1, bias=False),
            norm_layer(adjust_channels(32)),
            nn.SiLU()
        )

        for i, cnf in enumerate(inverted_residual_setting):
            # 前3阶段禁用 SKBlock
            use_skblock = i >= sum([round_repeats(args[-1]) for args in default_cnf[:3]])
            # 动态调整压缩率
            reduction = max(2, cnf.expanded_c // 64)
            layers[cnf.index] = block(cnf, norm_layer, use_skblock=use_skblock, reduction=reduction)

        self.features = nn.Sequential(layers)

        last_conv_input_c = inverted_residual_setting[-1].out_c
        last_conv_output_c = adjust_channels(1280)
        self.top_conv = nn.Sequential(
            nn.Conv2d(last_conv_input_c, last_conv_output_c, kernel_size=1, bias=False),
            norm_layer(last_conv_output_c),
            nn.SiLU()
        )

        self.avgpool = GeM()
        self.classifier = nn.Sequential(
            nn.Dropout(p=dropout_rate, inplace=False),
            nn.Linear(last_conv_output_c, num_classes)
        )

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out')
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
        x = self.top_conv(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x


def efficientnet_b0(num_classes=1000):
    return EfficientNet(
        width_coefficient=1.0,
        depth_coefficient=1.0,
        dropout_rate=0.1,
        num_classes=num_classes
    )


if __name__ == "__main__":
    # 计算 FLOPs
    model = efficientnet_b0()
    input_tensor = torch.randn(1, 3, 224, 224)
    flops, params = FLOPsCounter.calculate(model, input_tensor)
    print(f"模型的 FLOPs: {flops}")
    print(f"模型的参数数量: {params}")