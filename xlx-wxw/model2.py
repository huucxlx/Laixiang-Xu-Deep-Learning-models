from typing import Optional
import numpy as np
import torch
import torch.nn as nn
from torch import Tensor
import torch.nn.functional as F
import einops
from timm.layers import trunc_normal_

def _make_divisible(ch, divisor=8, min_ch=None):
    """
    This function is taken from the original tf repo.
    It ensures that all layers have a channel number that is divisible by 8
    It can be seen here:
    https://github.com/tensorflow/models/blob/master/research/slim/nets/mobilenet/mobilenet.py
    """
    if min_ch is None:
        min_ch = divisor
    new_ch = max(min_ch, int(ch + divisor / 2) // divisor * divisor)
    # Make sure that round down does not go down by more than 10%.
    if new_ch < 0.9 * ch:
        new_ch += divisor
    return new_ch


def _mcfg(**kwargs):
    """动态生成模型配置字典"""
    # 确保所有必要的键都存在
    default_keys = ['w0', 'wa', 'wm', 'group_w', 'depth']
    for key in default_keys:
        if key not in kwargs:
            raise ValueError(f"Missing key '{key}' in model configuration.")
    return kwargs

model_cfgs = {
    "regnetx_200mf": _mcfg(w0=24, wa=36.44, wm=2.49, group_w=8, depth=13),
    "regnetx_400mf": _mcfg(w0=24, wa=24.48, wm=2.54, group_w=16, depth=22),
    "regnetx_600mf": _mcfg(w0=48, wa=36.97, wm=2.24, group_w=24, depth=16),
    "regnetx_800mf": _mcfg(w0=56, wa=35.73, wm=2.28, group_w=16, depth=16),
    "regnetx_1.6gf": _mcfg(w0=80, wa=34.01, wm=2.25, group_w=24, depth=18),
    "regnetx_3.2gf": _mcfg(w0=88, wa=26.31, wm=2.25, group_w=48, depth=25),
    "regnetx_4.0gf": _mcfg(w0=96, wa=38.65, wm=2.43, group_w=40, depth=23),
    "regnetx_6.4gf": _mcfg(w0=184, wa=60.83, wm=2.07, group_w=56, depth=17),
    "regnetx_8.0gf": _mcfg(w0=80, wa=49.56, wm=2.88, group_w=120, depth=23),
    "regnetx_12gf": _mcfg(w0=168, wa=73.36, wm=2.37, group_w=112, depth=19),
    "regnetx_16gf": _mcfg(w0=216, wa=55.59, wm=2.1, group_w=128, depth=22),
    "regnetx_32gf": _mcfg(w0=320, wa=69.86, wm=2.0, group_w=168, depth=23),
    "regnety_200mf": _mcfg(w0=24, wa=36.44, wm=2.49, group_w=8, depth=13, se_ratio=0.25),
    "regnety_400mf": _mcfg(w0=48, wa=27.89, wm=2.09, group_w=8, depth=16, se_ratio=0.25,stem_width=32),
    "regnety_600mf": _mcfg(w0=48, wa=32.54, wm=2.32, group_w=16, depth=15, se_ratio=0.25),
    "regnety_800mf": _mcfg(w0=56, wa=38.84, wm=2.4, group_w=16, depth=14, se_ratio=0.25),
    "regnety_1.6gf": _mcfg(w0=48, wa=20.71, wm=2.65, group_w=24, depth=27, se_ratio=0.25),
    "regnety_3.2gf": _mcfg(w0=80, wa=42.63, wm=2.66, group_w=24, depth=21, se_ratio=0.25),
    "regnety_4.0gf": _mcfg(w0=96, wa=31.41, wm=2.24, group_w=64, depth=22, se_ratio=0.25),
    "regnety_6.4gf": _mcfg(w0=112, wa=33.22, wm=2.27, group_w=72, depth=25, se_ratio=0.25),
    "regnety_8.0gf": _mcfg(w0=192, wa=76.82, wm=2.19, group_w=56, depth=17, se_ratio=0.25),
    "regnety_12gf": _mcfg(w0=168, wa=73.36, wm=2.37, group_w=112, depth=19, se_ratio=0.25),
    "regnety_16gf": _mcfg(w0=200, wa=106.23, wm=2.48, group_w=112, depth=18, se_ratio=0.25),
    "regnety_32gf": _mcfg(w0=232, wa=115.89, wm=2.53, group_w=232, depth=20, se_ratio=0.25)
}


def generate_width_depth(wa, w0, wm, depth):
    """
    根据给定的参数生成宽度和阶段数量
    :param wa: 宽度增长斜率
    :param w0: 初始宽度
    :param wm: 宽度增长因子
    :param depth: 网络深度
    :return: 宽度列表和阶段数量
    """
    # 生成宽度列表
    widths = [int(round(w0 + wa * i)) for i in range(depth)]
    # 对宽度进行分组
    width_group = []
    current_width = widths[0]
    current_group = [current_width]
    for width in widths[1:]:
        if width != current_width:
            width_group.append(current_group)
            current_group = [width]
            current_width = width
        else:
            current_group.append(width)
    width_group.append(current_group)
    # 计算每个阶段的宽度
    stage_widths = [group[0] for group in width_group]
    # 计算阶段数量
    num_stages = len(stage_widths)

    # 检查阶段数量是否为 4，如果不是则调整
    if num_stages != 4:
        print(f"Warning: Generated {num_stages} stages, expected 4. Adjusting...")
        # 这里可以添加调整逻辑，例如重新计算参数或者截断/扩展宽度列表
        # 简单示例：如果阶段数量小于 4，扩展宽度列表
        while num_stages < 4:
            stage_widths.append(stage_widths[-1])
            num_stages = len(stage_widths)
        # 如果阶段数量大于 4，截断宽度列表
        if num_stages > 4:
            stage_widths = stage_widths[:4]
            num_stages = 4

    return stage_widths, num_stages


def adjust_width_groups_comp(widths: list, groups: list):
    """Adjusts the compatibility of widths and groups."""
    groups = [min(g, w_bot) for g, w_bot in zip(groups, widths)]
    # Adjust w to an integral multiple of g
    widths = [int(round(w / g) * g) for w, g in zip(widths, groups)]
    return widths, groups


class ConvBNAct(nn.Module):
    def __init__(self,
                 in_c: int,
                 out_c: int,
                 kernel_s: int = 1,
                 stride: int = 1,
                 padding: int = 0,
                 groups: int = 1,
                 act: Optional[nn.Module] = nn.ReLU(inplace=True)):
        super(ConvBNAct, self).__init__()
        if in_c % groups != 0:
            # 调整 groups 为 1 以避免错误
            groups = 1
        self.conv = nn.Conv2d(in_channels=in_c, out_channels=out_c, kernel_size=kernel_s, stride=stride,
                              padding=padding, groups=groups)
        self.bn = nn.BatchNorm2d(out_c)
        self.act = nn.ReLU(inplace=True)
        self.act = act if act is not None else nn.Identity()

    def forward(self, x: Tensor) -> Tensor:
        x = self.conv(x)
        x = self.bn(x)
        x = self.act(x)
        return x


class RegHead(nn.Module):
    def __init__(self,
                 in_unit: int = 1768,
                 out_unit: int = 1000,
                 output_size: tuple = (2, 2),
                 drop_ratio: float = 0.25):
        super(RegHead, self).__init__()
        self.pool = nn.AdaptiveAvgPool2d(output_size=output_size)
        self.dropout = nn.Dropout(drop_ratio)
        # 计算全连接层的输入维度
        fc_in_features = in_unit * output_size[0] * output_size[1]
        self.fc = nn.Linear(fc_in_features, out_unit)

    def forward(self, x: Tensor) -> Tensor:
        x = self.pool(x)
        x = torch.flatten(x, 1)
        x = self.dropout(x)
        x = self.fc(x)
        return x


class SqueezeExcitation(nn.Module):
    def __init__(self, input_c: int, expand_c: int, se_ratio: float = 0.25):
        super(SqueezeExcitation, self).__init__()
        squeeze_c = int(input_c * se_ratio)
        self.fc1 = nn.Conv2d(expand_c, squeeze_c, 1)
        self.ac1 = nn.ReLU(inplace=True)
        self.fc2 = nn.Conv2d(squeeze_c, expand_c, 1)
        self.ac2 = nn.Sigmoid()

    def forward(self, x: Tensor) -> Tensor:
        scale = x.mean((2, 3), keepdim=True)
        scale = self.fc1(scale)
        scale = self.ac1(scale)
        scale = self.fc2(scale)
        scale = self.ac2(scale)
        return scale * x


class Bottleneck(nn.Module):
    def __init__(self,
                 in_c: int,
                 out_c: int,
                 stride: int = 1,
                 group_width: int = 1,
                 se_ratio: float = 0.,
                 drop_ratio: float = 0.,
                 use_gat: bool = False,  # 新增参数
                 use_dattention: bool = False,  # 新增参数
                 dattention_kwargs: dict = {}):  # 新增参数
        super(Bottleneck, self).__init__()

        self.conv1 = ConvBNAct(in_c=in_c, out_c=out_c, kernel_s=1)
        self.conv2 = ConvBNAct(in_c=out_c,
                               out_c=out_c,
                               kernel_s=3,
                               stride=stride,
                               padding=1,
                               groups=out_c // group_width)

        if se_ratio > 0:
            self.se = SqueezeExcitation(in_c, out_c, se_ratio)
        else:
            self.se = nn.Identity()

        self.conv3 = ConvBNAct(in_c=out_c, out_c=out_c, kernel_s=1, act=None)
        self.ac3 = nn.ReLU(inplace=True)

        if drop_ratio > 0:
            self.dropout = nn.Dropout(p=drop_ratio)
        else:
            self.dropout = nn.Identity()

        if (in_c != out_c) or (stride != 1):
            self.downsample = ConvBNAct(in_c=in_c, out_c=out_c, kernel_s=1, stride=stride, act=None)
        else:
            self.downsample = nn.Identity()
            # 动态添加注意力模块
        if use_gat:
                self.gat = GAT(nfeat=out_c, nhid=64, n_final_out=out_c, dropout=0.1, alpha=0.2, nheads=4)

        if use_dattention:
                self.dattention = DAttention(channel=out_c,  ** dattention_kwargs)
    def zero_init_last_bn(self):
        nn.init.zeros_(self.conv3.bn.weight)

    def forward(self, x: Tensor) -> Tensor:
        shortcut = x
        x = self.conv1(x)
        x = self.conv2(x)

        x = self.se(x)
        x = self.conv3(x)

        x = self.dropout(x)

        shortcut = self.downsample(shortcut)

        x += shortcut
        x = self.ac3(x)
        return x


class RegStage(nn.Module):
    def __init__(self,
                 in_c: int,
                 out_c: int,
                 depth: int,
                 group_width: int,
                 se_ratio: float,
                 use_gat, use_dattention, dattention_kwargs):
        super(RegStage, self).__init__()
        self.in_c = in_c
        self.out_c = out_c
        self.depth = depth
        self.group_width = group_width
        self.se_ratio = se_ratio
        self.use_gat = use_gat
        self.use_dattention = use_dattention
        self.dattention_kwargs = dattention_kwargs
        for i in range(depth):
            block_stride = 2 if i == 0 else 1
            block_in_c = in_c if i == 0 else out_c

            name = "b{}".format(i + 1)
            self.add_module(name,
                            Bottleneck(in_c=block_in_c,
                                       out_c=out_c,
                                       stride=block_stride,
                                       group_width=group_width,
                                       se_ratio=se_ratio))

    def forward(self, x: Tensor) -> Tensor:
        for block in self.children():
            x = block(x)
        return x


class RegNet(nn.Module):
    """RegNet model.

    Paper: https://arxiv.org/abs/2003.13678
    Original Impl: https://github.com/facebookresearch/pycls/blob/master/pycls/models/regnet.py
    and refer to: https://github.com/rwightman/pytorch-image-models/blob/master/timm/models/regnet.py
    """

    def __init__(self,
                 cfg: dict,
                 in_c: int = 3,
                 num_classes: int = 1000,
                 attn_config: Optional[dict] = None,
                 zero_init_last_bn: bool = True):
        super(RegNet, self).__init__()
        stage_info = self._build_stage_info(cfg, attn_config)  # 传递注意力配置
        # RegStem
        stem_c = cfg["stem_width"]
        self.stem = ConvBNAct(3, cfg["stem_width"], kernel_s=3, stride=2, padding=1)

        # build stages
        input_channels = stem_c
        for i, stage_args in enumerate(stage_info):
            stage_name = "s{}".format(i + 1)
            # 如果 stage_args 中已经有 in_c，移除它
            if 'in_c' in stage_args:
                del stage_args['in_c']
            # 确保 in_c 只传入一次
            stage_args['in_c'] = input_channels
            self.add_module(stage_name, RegStage(**stage_args))
            input_channels = stage_args["out_c"]

        # RegHead
        self.head = RegHead(in_unit=input_channels, out_unit=num_classes)

        # initial weights
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_uniform_(m.weight, mode="fan_out",  nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, mean=0.0, std=0.01)
                nn.init.zeros_(m.bias)

        if zero_init_last_bn:
            for m in self.modules():
                if hasattr(m, "zero_init_last_bn"):
                    m.zero_init_last_bn()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for layer in self.children():
            x = layer(x)
        return x

    @staticmethod
    def _build_stage_info(cfg: dict, attn_config: Optional[dict] = None):
        # 检查参数是否存在
        required_keys = ["wa", "w0", "wm", "depth"]
        for key in required_keys:
            if key not in cfg:
                raise KeyError(f"Missing key '{key}' in cfg: {cfg}")

        # 生成宽度和阶段数量
        stage_widths, num_stages = generate_width_depth(cfg["wa"], cfg["w0"], cfg["wm"], cfg["depth"])

        # 构建阶段信息
        stage_info = []
        input_c = cfg["stem_width"]
        for stage_idx in range(num_stages):
            stage_info.append({
                "out_c": stage_widths[stage_idx],
                "depth": cfg["depth"] if isinstance(cfg["depth"], int) else cfg["depth"][stage_idx],
                "group_width": cfg["group_w"],
                "se_ratio": cfg.get("se_ratio", 0),
                "use_gat": attn_config["use_gat"][stage_idx],
                "use_dattention": attn_config["use_dattention"][stage_idx],
                "dattention_kwargs": attn_config["dattention_kwargs"]
            })
        return stage_info
class GraphAttentionLayer(nn.Module):
    def __init__(self, in_features, out_features, dropout, alpha, concat=True):
        super(GraphAttentionLayer, self).__init__()
        self.dropout = dropout
        self.in_features = in_features
        self.out_features = out_features
        self.alpha = alpha
        self.concat = concat
        self.W = nn.Parameter(torch.zeros(size=(in_features, out_features)))
        nn.init.xavier_uniform_(self.W.data, gain=1.414)
        self.a = nn.Parameter(torch.zeros(size=(2*out_features, 1)))
        nn.init.xavier_uniform_(self.a.data, gain=1.414)
        self.leakyrelu = nn.LeakyReLU(self.alpha)

    def forward(self, x, adj):
        h = torch.bmm(x, self.W)
        N = x.size()[0]
        a_input = torch.cat([h.repeat(1, N).view(N * N, self.out_features), h.repeat(N, 1)], dim=1).view(N, N, 2 * self.out_features)
        attention = self.leakyrelu(torch.matmul(a_input, self.a).squeeze(2))
        zero_vec = -9e15 * torch.ones_like(attention)
        attention = torch.where(adj > 0, attention, zero_vec)
        attention = F.softmax(attention, dim=1)
        attention = F.dropout(attention, self.dropout, training=self.training)
        h_prime = torch.bmm(attention, h)
        if self.concat:
            return F.elu(h_prime)
        else:
            return h_prime

    def __repr__(self):
        return self.__class__.__name__ + ' (' + str(self.in_features) + ' -> ' + str(self.out_features) + ')'


class GAT(nn.Module):
    def __init__(self, nfeat, nhid, n_final_out, dropout, alpha, nheads):
        """Dense version of GAT."""
        super(GAT, self).__init__()
        self.dropout = dropout

        self.attentions = [GraphAttentionLayer(nfeat, nhid, dropout=dropout, alpha=alpha, concat=True) for _ in range(nheads)]
        for i, attention in enumerate(self.attentions):
            self.add_module('attention_{}'.format(i), attention)

        self.out_att = GraphAttentionLayer(nhid * nheads, n_final_out, dropout=dropout, alpha=alpha, concat=False)

    def forward(self, x, adj):
        x = F.dropout(x, self.dropout, training=self.training)
        x = torch.cat([att(x, adj) for att in self.attentions], dim=1)
        x = F.dropout(x, self.dropout, training=self.training)
        x = F.elu(self.out_att(x, adj))
        return x


class LayerNormProxy(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.norm = nn.LayerNorm(dim)

    def forward(self, x):
        x = einops.rearrange(x, 'b c h w -> b h w c')
        x = self.norm(x)
        return einops.rearrange(x, 'b h w c -> b c h w')


class DAttention(nn.Module):
    def __init__(
            self, channel, q_size, n_heads=8,  n_groups=4,
            attn_drop=0.0, proj_drop=0.0, stride=1,
            offset_range_factor=4, use_pe=True, dwc_pe=True,
            no_off=False, fixed_pe=False, ksize=3, log_cpb=False,
    ):
        super().__init__()
        n_head_channels = channel // n_heads
        self.dwc_pe = dwc_pe
        self.n_head_channels = n_head_channels
        self.scale = self.n_head_channels ** -0.5
        self.n_heads = n_heads
        self.q_h, self.q_w = q_size
        self.kv_h, self.kv_w = self.q_h // stride, self.q_w // stride
        self.nc = n_head_channels * n_heads
        self.n_groups = n_groups
        self.n_group_channels = self.nc // self.n_groups
        self.n_group_heads = self.n_heads // self.n_groups
        self.use_pe = use_pe
        self.fixed_pe = fixed_pe
        self.no_off = no_off
        self.offset_range_factor = offset_range_factor
        self.ksize = ksize
        self.log_cpb = log_cpb
        self.stride = stride
        kk = self.ksize
        pad_size = kk // 2 if kk != stride else 0

        self.conv_offset = nn.Sequential(
            nn.Conv2d(self.n_group_channels, self.n_group_channels, kk, stride, pad_size, groups=self.n_group_channels),
            LayerNormProxy(self.n_group_channels),
            nn.GELU(),
            nn.Conv2d(self.n_group_channels, 2, 1, 1, 0, bias=False)
        )
        if self.no_off:
            for m in self.conv_offset.parameters():
                m.requires_grad_(False)

        self.proj_q = nn.Conv2d(
            channel, self.nc,  # 修改输入通道数
            kernel_size=1, stride=1, padding=0
        )

        self.proj_k = nn.Conv2d(
            channel, self.nc,  # 修改输入通道数
            kernel_size=1, stride=1, padding=0
        )

        self.proj_v = nn.Conv2d(
            channel, self.nc,  # 修改输入通道数
            kernel_size=1, stride=1, padding=0
        )

        self.proj_out = nn.Conv2d(
            self.nc, channel,  # 修改输出通道数
            kernel_size=1, stride=1, padding=0
        )

        self.proj_drop = nn.Dropout(proj_drop, inplace=True)
        self.attn_drop = nn.Dropout(attn_drop, inplace=True)

        if self.use_pe and not self.no_off:
            if self.dwc_pe:
                self.rpe_table = nn.Conv2d(
                    self.nc, self.nc, kernel_size=3, stride=1, padding=1, groups=self.nc)
            elif self.fixed_pe:
                self.rpe_table = nn.Parameter(
                    torch.zeros(self.n_heads, self.q_h * self.q_w, self.kv_h * self.kv_w)
                )
                trunc_normal_(self.rpe_table, std=0.01)
            elif self.log_cpb:
                # Borrowed from Swin-V2
                self.rpe_table = nn.Sequential(
                    nn.Linear(2, 32, bias=True),
                    nn.ReLU(inplace=True),
                    nn.Linear(32, self.n_group_heads, bias=False)
                )
            else:
                self.rpe_table = nn.Parameter(
                    torch.zeros(self.n_heads, self.q_h * 2 - 1, self.q_w * 2 - 1)
                )
                trunc_normal_(self.rpe_table, std=0.01)
        else:
            self.rpe_table = None

    @torch.no_grad()
    def _get_ref_points(self, H_key, W_key, B, dtype, device):
        ref_y, ref_x = torch.meshgrid(
            torch.linspace(0.5, H_key - 0.5, H_key, dtype=dtype, device=device),
            torch.linspace(0.5, W_key - 0.5, W_key, dtype=dtype, device=device),
            indexing='ij'
        )
        ref = torch.stack((ref_y, ref_x), -1)
        ref[..., 1].div_(W_key - 1.0).mul_(2.0).sub_(1.0)
        ref[..., 0].div_(H_key - 1.0).mul_(2.0).sub_(1.0)
        ref = ref[None, ...].expand(B * self.n_groups, -1, -1, -1)  # B * g H W 2

        return ref

    @torch.no_grad()
    def _get_q_grid(self, H, W, B, dtype, device):
        ref_y, ref_x = torch.meshgrid(
            torch.arange(0, H, dtype=dtype, device=device),
            torch.arange(0, W, dtype=dtype, device=device),
            indexing='ij'
        )
        ref = torch.stack((ref_y, ref_x), -1)
        ref[..., 1].div_(W - 1.0).mul_(2.0).sub_(1.0)
        ref[..., 0].div_(H - 1.0).mul_(2.0).sub_(1.0)
        ref = ref[None, ...].expand(B * self.n_groups, -1, -1, -1)  # B * g H W 2
        return ref

    def forward(self, x):
        print(f"Input shape to DAttention: {x.shape}")  # 添加调试信息
        B, C, H, W = x.size()
        dtype, device = x.dtype, x.device

        q = self.proj_q(x)
        q_off = einops.rearrange(q, 'b (g c) h w -> (b g) c h w', g=self.n_groups, c=self.n_group_channels)
        offset = self.conv_offset(q_off).contiguous()  # B * g 2 Hg Wg
        Hk, Wk = offset.size(2), offset.size(3)
        n_sample = Hk * Wk

        if self.offset_range_factor >= 0 and not self.no_off:
            offset_range = torch.tensor([1.0 / (Hk - 1.0), 1.0 / (Wk - 1.0)], device=device).reshape(1, 2, 1, 1)
            offset = offset.tanh().mul(offset_range).mul(self.offset_range_factor)

        offset = einops.rearrange(offset, 'b p h w -> b h w p')
        reference = self._get_ref_points(Hk, Wk, B, dtype, device)

        if self.no_off:
            offset = offset.fill_(0.0)

        if self.offset_range_factor >= 0:
            pos = offset + reference
        else:
            pos = (offset + reference).clamp(-1., +1.)

        if self.no_off:
            x_sampled = F.avg_pool2d(x, kernel_size=self.stride, stride=self.stride)
            assert x_sampled.size(2) == Hk and x_sampled.size(3) == Wk, f"Size is {x_sampled.size()}"
        else:
            x_sampled = F.grid_sample(
                input=x.reshape(B * self.n_groups, self.n_group_channels, H, W),
                grid=pos[..., (1, 0)],  # y, x -> x, y
                mode='bilinear', align_corners=True)  # B * g, Cg, Hg, Wg

        x_sampled = x_sampled.reshape(B, C, 1, n_sample)

        q = q.reshape(B * self.n_heads, self.n_head_channels, H * W)
        k = self.proj_k(x_sampled).reshape(B * self.n_heads, self.n_head_channels, n_sample)
        v = self.proj_v(x_sampled).reshape(B * self.n_heads, self.n_head_channels, n_sample)

        attn = torch.einsum('b c m, b c n -> b m n', q, k)  # B * h, HW, Ns
        attn = attn.mul(self.scale)

        if self.use_pe and (not self.no_off):
            if self.dwc_pe:
                residual_lepe = self.rpe_table(q.reshape(B, C, H, W)).reshape(B * self.n_heads, self.n_head_channels,
                                                                              H * W)
            elif self.fixed_pe:
                rpe_table = self.rpe_table
                attn_bias = rpe_table[None, ...].expand(B, -1, -1, -1)
                attn = attn + attn_bias.reshape(B * self.n_heads, H * W, n_sample)
            elif self.log_cpb:
                q_grid = self._get_q_grid(H, W, B, dtype, device)
                displacement = (
                            q_grid.reshape(B * self.n_groups, H * W, 2).unsqueeze(2) - pos.reshape(B * self.n_groups,
                                                                                                   n_sample,
                                                                                                   2).unsqueeze(1)).mul(
                    4.0)  # d_y, d_x [-8, +8]
                displacement = torch.sign(displacement) * torch.log2(torch.abs(displacement) + 1.0) / np.log2(8.0)
                attn_bias = self.rpe_table(displacement)  # B * g, H * W, n_sample, h_g
                attn = attn + einops.rearrange(attn_bias, 'b m n h -> (b h) m n', h=self.n_group_heads)
            else:
                rpe_table = self.rpe_table
                rpe_bias = rpe_table[None, ...].expand(B, -1, -1, -1)
                q_grid = self._get_q_grid(H, W, B, dtype, device)
                displacement = (
                            q_grid.reshape(B * self.n_groups, H * W, 2).unsqueeze(2) - pos.reshape(B * self.n_groups,
                                                                                                   n_sample,
                                                                                                   2).unsqueeze(1)).mul(
                    0.5)
                attn_bias = F.grid_sample(
                    input=einops.rearrange(rpe_bias, 'b (g c) h w -> (b g) c h w', c=self.n_group_heads,
                                           g=self.n_groups),
                    grid=displacement[..., (1, 0)],
                    mode='bilinear', align_corners=True)  # B * g, h_g, HW, Ns

                attn_bias = attn_bias.reshape(B * self.n_heads, H * W, n_sample)
                attn = attn + attn_bias

        attn = F.softmax(attn, dim=2)
        attn = self.attn_drop(attn)

        out = torch.einsum('b m n, b c n -> b c m', attn, v)

        if self.use_pe and self.dwc_pe:
            out = out + residual_lepe
        out = out.reshape(B, C, H, W)

        y = self.proj_drop(self.proj_out(out))

        return y


# 可以添加一个新的函数来创建新模型
def create_new_model(model_name, num_classes):
    if model_name.lower() == 'gat':
        model = GAT(nfeat=128, nhid=64, n_final_out=num_classes, dropout=0.2, alpha=0.2, nheads=4)
    elif model_name.lower() == 'dattention':
        model = DAttention(channel=128, q_size=(16, 16))
    elif model_name.lower() == 'regnety_400mf':
        # 假设 create_regnet 函数可以创建 RegNet 模型
        model = create_regnet(model_name, num_classes)
    else:
        print("support model name: gat, dattention, regnety_400mf")
        raise KeyError("not support model name: {}".format(model_name))
    return model


def create_regnet(
    model_name: str,
    num_classes: int = 1000,
    attn_config: Optional[dict] = None,
    weights: Optional[str] = None,
    freeze_layers: bool = False
):
    # 从 model_cfgs 中获取基础配置
    if model_name not in model_cfgs:
        raise ValueError(f"Unsupported model name: {model_name}")
    base_cfg = model_cfgs[model_name]

    # 合并基础配置和注意力配置
    cfg = base_cfg.copy()
    if attn_config is not None:
        cfg.update(attn_config)

    # 创建模型
    model = RegNet(cfg=cfg, num_classes=num_classes, attn_config=attn_config)

    # 加载预训练权重
    if weights is not None:
        try:
            weights_dict = torch.load(weights, map_location="cpu")
            model.load_state_dict(weights_dict, strict=False)
        except FileNotFoundError:
            print(f"Error: The specified weights file {weights} was not found.")
        except Exception as e:
            print(f"Error loading weights: {e}")

    # 冻结层
    if freeze_layers:
        for name, param in model.named_parameters():
            if "fc" not in name:
                param.requires_grad = False

    return model