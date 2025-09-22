import torch
import torch.nn as nn
import math
from torchvision.ops import deform_conv2d

__all__ = ['mobilenetv3_small']


def _make_divisible(v, divisor, min_value=None):
    if min_value is None:
        min_value = divisor
    new_v = max(min_value, int(v + divisor / 2) // divisor * divisor)
    if new_v < 0.9 * v:
        new_v += divisor
    return new_v


class h_sigmoid(nn.Module):
    def __init__(self, inplace=True):
        super(h_sigmoid, self).__init__()
        self.relu = nn.ReLU6(inplace=inplace)

    def forward(self, x):
        return self.relu(x + 3) / 6


class h_swish(nn.Module):
    def __init__(self, inplace=True):
        super(h_swish, self).__init__()
        self.sigmoid = h_sigmoid(inplace=inplace)

    def forward(self, x):
        return x * self.sigmoid(x)


class SELayer(nn.Module):
    def __init__(self, channel, reduction=4):
        super(SELayer, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channel, _make_divisible(channel // reduction, 8)),
            nn.ReLU(inplace=True),
            nn.Linear(_make_divisible(channel // reduction, 8), channel),
            h_sigmoid()
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y


class AdditiveAttention(nn.Module):
    """Additive Attention (Bahdanau-style) implementation"""

    def __init__(self, in_channels):
        super(AdditiveAttention, self).__init__()
        self.query_proj = nn.Conv2d(in_channels, in_channels // 8, kernel_size=1)
        self.key_proj = nn.Conv2d(in_channels, in_channels // 8, kernel_size=1)
        self.value_proj = nn.Conv2d(in_channels, in_channels, kernel_size=1)
        self.gamma = nn.Parameter(torch.zeros(1))

        # Learnable parameters for additive attention
        self.W = nn.Parameter(torch.randn(in_channels // 8, in_channels // 8))
        self.b = nn.Parameter(torch.zeros(1))

    def forward(self, x):
        batch_size, C, H, W = x.size()

        # Projections
        Q = self.query_proj(x).view(batch_size, -1, H * W).permute(0, 2, 1)  # (B, N, C')
        K = self.key_proj(x).view(batch_size, -1, H * W)  # (B, C', N)
        V = self.value_proj(x).view(batch_size, -1, H * W)  # (B, C, N)

        # Additive attention calculation
        energy = torch.tanh(torch.bmm(Q, self.W.unsqueeze(0).expand(batch_size, -1, -1)) + self.b)
        energy = torch.bmm(energy, K)  # (B, N, N)
        attention = torch.softmax(energy, dim=-1)

        # Apply attention
        out = torch.bmm(V, attention.permute(0, 2, 1))
        out = out.view(batch_size, C, H, W)

        # Residual connection
        return self.gamma * out + x


class SelfAttention(nn.Module):
    """Scaled Dot-Product Self-Attention implementation"""

    def __init__(self, in_channels):
        super(SelfAttention, self).__init__()
        self.query_conv = nn.Conv2d(in_channels, in_channels // 8, kernel_size=1)
        self.key_conv = nn.Conv2d(in_channels, in_channels // 8, kernel_size=1)
        self.value_conv = nn.Conv2d(in_channels, in_channels, kernel_size=1)
        self.gamma = nn.Parameter(torch.zeros(1))

    def forward(self, x):
        batch_size, C, height, width = x.size()

        # Calculate queries, keys, values
        proj_query = self.query_conv(x).view(batch_size, -1, height * width).permute(0, 2, 1)
        proj_key = self.key_conv(x).view(batch_size, -1, height * width)
        proj_value = self.value_conv(x).view(batch_size, -1, height * width)

        # Calculate attention
        energy = torch.bmm(proj_query, proj_key)
        attention = torch.softmax(energy, dim=-1)

        # Apply attention to values
        out = torch.bmm(proj_value, attention.permute(0, 2, 1))
        out = out.view(batch_size, C, height, width)

        # Add residual connection
        out = self.gamma * out + x
        return out


class DeformableConv2d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, groups=1, bias=True):
        super(DeformableConv2d, self).__init__()
        if isinstance(kernel_size, int):
            kernel_size = (kernel_size, kernel_size)
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.dilation = dilation
        self.groups = groups

        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, stride=stride,
                              padding=padding, dilation=dilation, groups=groups, bias=bias)

        self.offset_conv = nn.Conv2d(in_channels, 2 * kernel_size[0] * kernel_size[1],
                                     kernel_size=kernel_size, stride=stride,
                                     padding=padding, dilation=dilation, groups=1, bias=True)

        nn.init.constant_(self.offset_conv.weight, 0.)
        nn.init.constant_(self.offset_conv.bias, 0.)

    def forward(self, x):
        offset = self.offset_conv(x)
        out = deform_conv2d(x, offset, self.conv.weight, self.conv.bias,
                            stride=self.stride, padding=self.padding,
                            dilation=self.dilation)
        return out


def conv_3x3_bn(inp, oup, stride):
    return nn.Sequential(
        nn.Conv2d(inp, oup, 3, stride, 1, bias=False),
        nn.BatchNorm2d(oup),
        h_swish()
    )


def conv_1x1_bn(inp, oup):
    return nn.Sequential(
        nn.Conv2d(inp, oup, 1, 1, 0, bias=False),
        nn.BatchNorm2d(oup),
        h_swish()
    )


class InvertedResidual(nn.Module):
    def __init__(self, inp, hidden_dim, oup, kernel_size, stride, use_se, use_hs,
                 use_deformable=False, use_self_attention=False, use_additive_attention=False):
        super(InvertedResidual, self).__init__()
        assert stride in [1, 2]

        if isinstance(kernel_size, int):
            kernel_size = (kernel_size, kernel_size)

        self.identity = stride == 1 and inp == oup
        self.use_deformable = use_deformable
        self.use_self_attention = use_self_attention
        self.use_additive_attention = use_additive_attention

        if inp == hidden_dim:
            self.conv = nn.Sequential(
                DeformableConv2d(hidden_dim, hidden_dim, kernel_size, stride,
                                 (kernel_size[0] - 1) // 2, groups=hidden_dim) if use_deformable else
                nn.Conv2d(hidden_dim, hidden_dim, kernel_size, stride,
                          (kernel_size[0] - 1) // 2, groups=hidden_dim, bias=False),
                nn.BatchNorm2d(hidden_dim),
                h_swish() if use_hs else nn.ReLU(inplace=True),
                SELayer(hidden_dim) if use_se else nn.Identity(),
                nn.Conv2d(hidden_dim, oup, 1, 1, 0, bias=False),
                nn.BatchNorm2d(oup),
            )
        else:
            self.conv = nn.Sequential(
                nn.Conv2d(inp, hidden_dim, 1, 1, 0, bias=False),
                nn.BatchNorm2d(hidden_dim),
                h_swish() if use_hs else nn.ReLU(inplace=True),
                DeformableConv2d(hidden_dim, hidden_dim, kernel_size, stride,
                                 (kernel_size[0] - 1) // 2, groups=hidden_dim) if use_deformable else
                nn.Conv2d(hidden_dim, hidden_dim, kernel_size, stride,
                          (kernel_size[0] - 1) // 2, groups=hidden_dim, bias=False),
                nn.BatchNorm2d(hidden_dim),
                SELayer(hidden_dim) if use_se else nn.Identity(),
                h_swish() if use_hs else nn.ReLU(inplace=True),
                nn.Conv2d(hidden_dim, oup, 1, 1, 0, bias=False),
                nn.BatchNorm2d(oup),
            )

        # Add attention modules if specified
        if self.use_self_attention:
            self.self_attn = SelfAttention(oup)
        if self.use_additive_attention:
            self.additive_attn = AdditiveAttention(oup)

    def forward(self, x):
        out = self.conv(x)

        # Apply attention modules in sequence
        if hasattr(self, 'self_attn'):
            out = self.self_attn(out)
        if hasattr(self, 'additive_attn'):
            out = self.additive_attn(out)

        if self.identity:
            return x + out
        else:
            return out


class MobileNetV3(nn.Module):
    def __init__(self, cfgs, mode, num_classes=4, width_mult=1.):
        super(MobileNetV3, self).__init__()
        self.cfgs = cfgs
        assert mode in ['large', 'small']

        input_channel = _make_divisible(16 * width_mult, 8)
        layers = [conv_3x3_bn(3, input_channel, 2)]

        block = InvertedResidual
        for layer_idx, (k, t, c, use_se, use_hs, s, use_deformable) in enumerate(self.cfgs):
            output_channel = _make_divisible(c * width_mult, 8)
            exp_size = _make_divisible(input_channel * t, 8)

            # Configure attention types for specific layers
            use_self_attention = (layer_idx in [5, 9])  # Layers 6 and 10 (0-based 5 and 9)
            use_additive_attention = (layer_idx in [6, 8])  # Layers 7 and 9 (0-based 6 and 8)

            layers.append(block(input_channel, exp_size, output_channel, k, s,
                                use_se, use_hs, use_deformable=use_deformable,
                                use_self_attention=use_self_attention,
                                use_additive_attention=use_additive_attention))
            input_channel = output_channel

        self.features = nn.Sequential(*layers)
        self.conv = conv_1x1_bn(input_channel, exp_size)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        output_channel = {'large': 1280, 'small': 1024}
        output_channel = _make_divisible(output_channel[mode] * width_mult, 8) if width_mult > 1.0 else output_channel[
            mode]
        self.classifier = nn.Sequential(
            nn.Linear(exp_size, output_channel),
            h_swish(),
            nn.Dropout(0.2),
            nn.Linear(output_channel, num_classes),
        )

        self._initialize_weights()

    def forward(self, x):
        x = self.features(x)
        x = self.conv(x)
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                m.weight.data.normal_(0, 0.01)
                m.bias.data.zero_()


def mobilenetv3_small(**kwargs):
    """
    Constructs a MobileNetV3-Small model with:
    - Deformable Convolution (DCN) in layers 3, 4, and 8
    - Self-Attention in layers 6 and 10
    - Additive Attention in layers 7 and 9
    (Note: Layer indices start from 1)
    """
    cfgs = [
        # k, t, c, SE, HS, s, use_deformable
        [(3, 3), 1, 16, 1, 0, 2, False],  # Layer 1
        [(3, 3), 4.5, 24, 0, 0, 2, False],  # Layer 2
        [(3, 3), 3.67, 24, 0, 0, 1, True],  # Layer 3 (DCN enabled)
        [(5, 5), 4, 40, 1, 1, 2, True],  # Layer 4 (DCN enabled)
        [(5, 5), 6, 40, 1, 1, 1, False],  # Layer 5
        [(5, 5), 3, 48, 1, 1, 1, False],  # Layer 6 (Self-Attention)
        [(5, 5), 3, 48, 1, 1, 1, False],  # Layer 7 (Additive Attention)
        [(5, 5), 6, 96, 1, 1, 2, True],  # Layer 8 (DCN enabled)
        [(5, 5), 6, 96, 1, 1, 1, False],  # Layer 9 (Additive Attention)
        [(5, 5), 6, 96, 1, 1, 1, False],  # Layer 10 (Self-Attention)
    ]

    return MobileNetV3(cfgs, mode='small', **kwargs)


if __name__ == "__main__":
    model = mobilenetv3_small(num_classes=4)
    print(model)

    # Test with dummy input
    inputs = torch.randn(1, 3, 224, 224)
    outputs = model(inputs)
    print(outputs.shape)  # Should be torch.Size([1, 1000])