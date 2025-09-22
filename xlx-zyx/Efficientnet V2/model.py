from collections import OrderedDict
from functools import partial
from typing import Callable, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor


# 移除了不再使用的 torchvision.models.efficientnet_v2_s
# from torchvision.models import efficientnet_v2_s


class DropPath(nn.Module):
    """
    Drop paths (Stochastic Depth) per sample  (when applied in main path of residual blocks).
    "Deep Networks with Stochastic Depth", https://arxiv.org/pdf/1603.09382.pdf
    """

    def __init__(self, drop_prob=0.3):
        super(DropPath, self).__init__()
        self.drop_prob = drop_prob

    def forward(self, x):
        return drop_path(x, self.drop_prob, self.training)


def drop_path(x, drop_prob: float = 0., training: bool = False):
    """
    Drop paths (Stochastic Depth) per sample (when applied in main path of residual blocks).
    "Deep Networks with Stochastic Depth", https://arxiv.org/pdf/1603.09382.pdf

    This function is taken from the rwightman.
    It can be seen here:
    https://github.com/rwightman/pytorch-image-models/blob/master/timm/models/layers/drop.py#L140
    """
    if drop_prob == 0. or not training:
        return x
    keep_prob = 1 - drop_prob
    # work with diff dim tensors, not just 2D ConvNets
    shape = (x.shape[0],) + (1,) * (x.ndim - 1)
    random_tensor = keep_prob + \
                    torch.rand(shape, dtype=x.dtype, device=x.device)
    random_tensor.floor_()  # binarize
    output = x.div(keep_prob) * random_tensor
    return output


class ConvBNAct(nn.Module):
    def __init__(self,
                 in_planes: int,
                 out_planes: int,
                 kernel_size: int = 3,
                 stride: int = 1,
                 groups: int = 1,
                 norm_layer: Optional[Callable[..., nn.Module]] = None,
                 activation_layer: Optional[Callable[..., nn.Module]] = None):
        super(ConvBNAct, self).__init__()

        padding = (kernel_size - 1) // 2
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        if activation_layer is None:
            activation_layer = nn.SiLU  # alias Swish  (torch>=1.7)

        self.conv = nn.Conv2d(in_channels=in_planes,
                              out_channels=out_planes,
                              kernel_size=kernel_size,
                              stride=stride,
                              padding=padding,
                              groups=groups,
                              bias=False)

        self.bn = norm_layer(out_planes)
        self.act = activation_layer()

    def forward(self, x):
        result = self.conv(x)
        result = self.bn(result)
        result = self.act(result)

        return result


class SqueezeExcite(nn.Module):
    def __init__(self,
                 input_c: int,  # block input channel
                 expand_c: int,  # block expand channel
                 se_ratio: float = 0.25):
        super(SqueezeExcite, self).__init__()
        squeeze_c = int(input_c * se_ratio)
        self.conv_reduce = nn.Conv2d(expand_c, squeeze_c, 1)
        self.act1 = nn.SiLU()  # alias Swish
        self.conv_expand = nn.Conv2d(squeeze_c, expand_c, 1)
        self.act2 = nn.Sigmoid()

    def forward(self, x: Tensor) -> Tensor:
        scale = x.mean((2, 3), keepdim=True)
        scale = self.conv_reduce(scale)
        scale = self.act1(scale)
        scale = self.conv_expand(scale)
        scale = self.act2(scale)
        return scale * x


class MBConv(nn.Module):
    def __init__(self,
                 kernel_size: int,
                 input_c: int,
                 out_c: int,
                 expand_ratio: int,
                 stride: int,
                 se_ratio: float,
                 drop_rate: float,
                 norm_layer: Callable[..., nn.Module]):
        super(MBConv, self).__init__()

        if stride not in [1, 2]:
            raise ValueError("illegal stride value.")

        self.has_shortcut = (stride == 1 and input_c == out_c)

        activation_layer = nn.SiLU  # alias Swish
        expanded_c = input_c * expand_ratio

        # 在EfficientNetV2中，MBConv中不存在expansion=1的情况所以conv_pw肯定存在
        assert expand_ratio != 1
        # Point-wise expansion
        self.expand_conv = ConvBNAct(input_c,
                                     expanded_c,
                                     kernel_size=1,
                                     norm_layer=norm_layer,
                                     activation_layer=activation_layer)

        # Depth-wise convolution
        self.dwconv = ConvBNAct(expanded_c,
                                expanded_c,
                                kernel_size=kernel_size,
                                stride=stride,
                                groups=expanded_c,
                                norm_layer=norm_layer,
                                activation_layer=activation_layer)

        self.se = SqueezeExcite(input_c, expanded_c,
                                se_ratio) if se_ratio > 0 else nn.Identity()

        # Point-wise linear projection
        self.project_conv = ConvBNAct(expanded_c,
                                      out_planes=out_c,
                                      kernel_size=1,
                                      norm_layer=norm_layer,
                                      activation_layer=nn.Identity)  # 注意这里没有激活函数，所有传入Identity

        self.out_channels = out_c

        # 只有在使用shortcut连接时才使用dropout层
        self.drop_rate = drop_rate
        if self.has_shortcut and drop_rate > 0:
            self.dropout = DropPath(drop_rate)

    def forward(self, x: Tensor) -> Tensor:
        result = self.expand_conv(x)
        result = self.dwconv(result)
        result = self.se(result)
        result = self.project_conv(result)

        if self.has_shortcut:
            if self.drop_rate > 0:
                result = self.dropout(result)
            result += x

        return result


class FusedMBConv(nn.Module):
    def __init__(self,
                 kernel_size: int,
                 input_c: int,
                 out_c: int,
                 expand_ratio: int,
                 stride: int,
                 se_ratio: float,
                 drop_rate: float,
                 norm_layer: Callable[..., nn.Module]):
        super(FusedMBConv, self).__init__()

        assert stride in [1, 2]
        assert se_ratio == 0

        self.has_shortcut = stride == 1 and input_c == out_c
        self.drop_rate = drop_rate

        self.has_expansion = expand_ratio != 1

        activation_layer = nn.SiLU  # alias Swish
        expanded_c = input_c * expand_ratio

        # 只有当expand ratio不等于1时才有expand conv
        if self.has_expansion:
            # Expansion convolution
            self.expand_conv = ConvBNAct(input_c,
                                         expanded_c,
                                         kernel_size=kernel_size,
                                         stride=stride,
                                         norm_layer=norm_layer,
                                         activation_layer=activation_layer)

            self.project_conv = ConvBNAct(expanded_c,
                                          out_c,
                                          kernel_size=1,
                                          norm_layer=norm_layer,
                                          activation_layer=nn.Identity)  # 注意没有激活函数
        else:
            # 当只有project_conv时的情况
            self.project_conv = ConvBNAct(input_c,
                                          out_c,
                                          kernel_size=kernel_size,
                                          stride=stride,
                                          norm_layer=norm_layer,
                                          activation_layer=activation_layer)  # 注意有激活函数

        self.out_channels = out_c

        # 只有在使用shortcut连接时才使用dropout层
        self.drop_rate = drop_rate
        if self.has_shortcut and drop_rate > 0:
            self.dropout = DropPath(drop_rate)

    def forward(self, x: Tensor) -> Tensor:
        if self.has_expansion:
            result = self.expand_conv(x)
            result = self.project_conv(result)
        else:
            result = self.project_conv(x)

        if self.has_shortcut:
            if self.drop_rate > 0:
                result = self.dropout(result)

            result += x

        return result


class LSKblock(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.conv0 = nn.Conv2d(dim, dim, 5, padding=2, groups=dim)
        self.conv_spatial = nn.Conv2d(dim, dim, 7, stride=1, padding=9, groups=dim, dilation=3)
        self.conv1 = nn.Conv2d(dim, dim // 2, 1)
        self.conv2 = nn.Conv2d(dim, dim // 2, 1)
        self.conv_squeeze = nn.Conv2d(2, 2, 7, padding=3)
        self.conv = nn.Conv2d(dim // 2, dim, 1)

        # Use a more stable initialization
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, x):
        # Add a residual connection for stability
        residual = x

        # Apply gradient scaling factor to prevent explosion
        scale_factor = 0.1

        attn1 = self.conv0(x)
        attn2 = self.conv_spatial(attn1)

        attn1 = self.conv1(attn1)
        attn2 = self.conv2(attn2)

        attn = torch.cat([attn1, attn2], dim=1)
        avg_attn = torch.mean(attn, dim=1, keepdim=True)
        max_attn, _ = torch.max(attn, dim=1, keepdim=True)
        agg = torch.cat([avg_attn, max_attn], dim=1)

        # Apply sigmoid with numerical stability
        sig = torch.clamp(self.conv_squeeze(agg), -20, 20).sigmoid()

        # Safe multiplication with clamping
        weighted1 = attn1 * torch.clamp(sig[:, 0:1], 0, 1)
        weighted2 = attn2 * torch.clamp(sig[:, 1:2], 0, 1)
        attn = weighted1 + weighted2

        # Apply scaling factor and clamp before final conv
        attn = torch.clamp(attn * scale_factor, -10, 10)
        attn = self.conv(attn)

        # Final output with residual connection for stability
        return residual + torch.tanh(attn) * scale_factor


class BioEnhancedModule(nn.Module):
    def __init__(self, dim):
        super().__init__()
        # 径向卷积实现
        self.radial_conv = nn.Conv2d(dim, dim, 5, padding=2, groups=dim)
        # 初始化成一个单位卷积核（中心为1，其他为0）
        with torch.no_grad():
            # 先将所有权重置为0
            nn.init.zeros_(self.radial_conv.weight)
            # 然后将中心位置设为1
            center_y, center_x = 2, 2  # 5x5卷积核的中心
            for i in range(self.radial_conv.weight.size(0)):
                self.radial_conv.weight[i, 0, center_y, center_x] = 1.0

        self.edge_detector = nn.Sequential(
            nn.Conv2d(dim, dim // 4, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(dim // 4, 1, 1),
            nn.Sigmoid())
        self.modulation = nn.Conv2d(dim * 2, dim, 1)

    def forward(self, x):
        # Radial convolution
        radial = self.radial_conv(x)

        # Edge detection
        edge_mask = self.edge_detector(x)

        # Modulation
        modulated = self.modulation(torch.cat([x, radial], dim=1))

        return modulated * edge_mask + x


class BioEnhancedBiFPN(nn.Module):
    def __init__(self, in_channels_list, out_channels):
        super().__init__()
        self.in_channels_list = in_channels_list
        self.out_channels = out_channels

        # Adapters to transform input features to common dimension
        self.adapters = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(in_channels, out_channels, 1),
                nn.BatchNorm2d(out_channels),
                nn.ReLU()
            ) for in_channels in in_channels_list
        ])

        # Bio-enhanced modules
        self.bio_modules = nn.ModuleList([
            BioEnhancedModule(out_channels) for _ in range(len(in_channels_list))
        ])

        # Fusion weights (ensuring proper broadcasting by reshaping in forward)
        self.fusion_weights = nn.Parameter(torch.ones(len(in_channels_list)), requires_grad=True)

        # Final processing
        self.final_conv = nn.Conv2d(out_channels, out_channels, 3, padding=1)
        self.final_bn = nn.BatchNorm2d(out_channels)
        self.final_act = nn.ReLU()

    def forward(self, features):
        # Ensure features is a list of tensors
        if not isinstance(features, list):
            features = [features]

        # Use only available features (in case fewer than expected)
        num_features = min(len(features), len(self.adapters))
        features = features[:num_features]

        # Process features through adapters
        processed_features = [self.adapters[i](feat) for i, feat in enumerate(features)]

        # Apply bio-enhancement
        enhanced_features = [self.bio_modules[i](feat) for i, feat in enumerate(processed_features)]

        # Normalize fusion weights
        weights = F.softmax(self.fusion_weights[:num_features], dim=0)

        # Find the smallest feature map size
        min_size = (float('inf'), float('inf'))
        for feat in enhanced_features:
            h, w = feat.shape[2:]
            if h * w < min_size[0] * min_size[1]:
                min_size = (h, w)

        # Resize all features to the smallest size to ensure compatibility
        resized_features = [
            F.interpolate(feat, size=min_size, mode='bilinear', align_corners=False)
            for feat in enhanced_features
        ]

        # Weighted fusion with reshaping weights for proper broadcasting
        fused = sum(w.view(1, 1, 1, 1) * feat for w, feat in zip(weights, resized_features))

        # Final processing
        output = self.final_act(self.final_bn(self.final_conv(fused)))

        return [output]  # Return as list for compatibility


class EfficientNetV2_LSK_Bio(nn.Module):
    def __init__(self,
                 model_cnf: list,
                 num_classes: int = 1000,
                 num_features: int = 1280,
                 dropout_rate: float = 0.2,
                 drop_connect_rate: float = 0.2):
        super().__init__()
        self.model_cnf = model_cnf
        norm_layer = partial(nn.BatchNorm2d, eps=1e-3, momentum=0.1)

        # Stem部分保持不变
        stem_filter_num = model_cnf[0][4]
        self.stem = ConvBNAct(3, stem_filter_num, kernel_size=3, stride=2, norm_layer=norm_layer)

        # 构建blocks (不直接使用nn.Sequential来允许我们收集中间特征)
        total_blocks = sum([i[0] for i in model_cnf])
        block_id = 0
        self.blocks = nn.ModuleList()
        for cnf in model_cnf:
            repeats = cnf[0]
            op = FusedMBConv if cnf[-2] == 0 else partial(MBConvWithLSK, lsk_ratio=0.5)
            for i in range(repeats):
                self.blocks.append(op(
                    kernel_size=cnf[1],
                    input_c=cnf[4] if i == 0 else cnf[5],
                    out_c=cnf[5],
                    expand_ratio=cnf[3],
                    stride=cnf[2] if i == 0 else 1,
                    se_ratio=cnf[-1],
                    drop_rate=drop_connect_rate * block_id / total_blocks,
                    norm_layer=norm_layer
                ))
                block_id += 1

        # BiFPN stage
        # 注意：我们确保通道数列表与模型配置匹配
        self.bifpn = BioEnhancedBiFPN(
            in_channels_list=[64, 128, 160, 256],  # 根据model_config的out_c设置
            out_channels=num_features // 2
        )

        # Head部分调整
        head = OrderedDict()
        head.update({"project_conv": ConvBNAct(num_features // 2, num_features, kernel_size=1, norm_layer=norm_layer)})
        head.update({"avgpool": nn.AdaptiveAvgPool2d(1)})
        head.update({"flatten": nn.Flatten()})
        if dropout_rate > 0:
            head.update({"dropout": nn.Dropout(p=dropout_rate, inplace=True)})
        head.update({"classifier": nn.Linear(num_features, num_classes)})
        self.head = nn.Sequential(head)

        # 初始化
        self._init_weights()

        # 设备移动助手 - 自动检测并移动到可用GPU
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.to(self.device)
        print(f"Model initialized on: {self.device}")

    def _init_weights(self):
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
        # 确保输入在正确的设备上
        x = x.to(self.device)

        # Stage 1: Stem
        x = self.stem(x)

        # Stage 2: 收集中间特征
        features = []
        stage_ends = [sum(cnf[0] for cnf in self.model_cnf[:i + 1]) for i in range(len(self.model_cnf))]
        stage_ends = stage_ends[-4:]  # 只保留最后四个阶段的末尾索引

        current_feature = x
        for i, block in enumerate(self.blocks):
            current_feature = block(current_feature)
            if i + 1 in stage_ends:  # 如果当前block是某个stage的最后一个
                features.append(current_feature)

        # Stage 3: BioEnhanced BiFPN
        x = self.bifpn(features)[0]  # 取最高层特征

        # Stage 4: Head
        x = self.head(x)
        return x


class MBConvWithLSK(nn.Module):
    """在MBConv中嵌入LSKblock的混合模块"""

    def __init__(self,
                 kernel_size: int,
                 input_c: int,
                 out_c: int,
                 expand_ratio: int,
                 stride: int,
                 se_ratio: float,
                 drop_rate: float,
                 norm_layer: Callable[..., nn.Module],
                 lsk_ratio: float = 0.5):
        super().__init__()
        self.lsk_ratio = lsk_ratio

        # 原始MBConv组件
        expanded_c = input_c * expand_ratio
        self.expand_conv = ConvBNAct(input_c, expanded_c, kernel_size=1,
                                     norm_layer=norm_layer, activation_layer=nn.SiLU)
        self.dwconv = ConvBNAct(expanded_c, expanded_c, kernel_size=kernel_size,
                                stride=stride, groups=expanded_c,
                                norm_layer=norm_layer, activation_layer=nn.SiLU)
        self.se = SqueezeExcite(input_c, expanded_c, se_ratio) if se_ratio > 0 else nn.Identity()
        self.project_conv = ConvBNAct(expanded_c, out_c, kernel_size=1,
                                      norm_layer=norm_layer, activation_layer=nn.Identity)

        # 新增LSK组件
        self.lsk = LSKblock(expanded_c)
        self.lsk_gate = nn.Parameter(torch.tensor(lsk_ratio))  # 可学习的混合比例

        # DropPath
        self.has_shortcut = (stride == 1 and input_c == out_c)
        self.drop_rate = drop_rate
        if self.has_shortcut and drop_rate > 0:
            self.dropout = DropPath(drop_rate)

    def forward(self, x: Tensor) -> Tensor:
        # Expansion
        result = self.expand_conv(x)

        # Depth-wise conv + LSK融合
        dw_result = self.dwconv(result)
        lsk_result = self.lsk(result)

        # 确保尺寸一致（通过插值调整）
        if dw_result.shape != lsk_result.shape:
            lsk_result = F.interpolate(lsk_result, size=dw_result.shape[2:], mode='bilinear', align_corners=False)

        gate = torch.sigmoid(self.lsk_gate)
        result = dw_result * (1 - gate) + lsk_result * gate

        # SE注意力
        result = self.se(result)

        # Projection
        result = self.project_conv(result)

        # Shortcut
        if self.has_shortcut:
            if self.drop_rate > 0:
                result = self.dropout(result)
            result += x

        return result


def efficientnetv2_s(num_classes: int = 1000):
    """整合LSK和BioEnhancedBiFPN的EfficientNetV2-S变体"""
    model_config = [
        [2, 3, 1, 1, 24, 24, 0, 0],
        [4, 3, 2, 4, 24, 48, 0, 0],
        [4, 3, 2, 4, 48, 64, 0, 0],  # 64 -> BiFPN输入1
        [6, 3, 2, 4, 64, 128, 1, 0.25],  # 128 -> BiFPN输入2
        [9, 3, 1, 6, 128, 160, 1, 0.25],  # 160 -> BiFPN输入3
        [15, 3, 2, 6, 160, 256, 1, 0.25]  # 256 -> BiFPN输入4
    ]

    model = EfficientNetV2_LSK_Bio(
        model_cnf=model_config,
        num_classes=num_classes,
        dropout_rate=0.2
    )

    return model