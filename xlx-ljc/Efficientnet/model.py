'''
from collections import OrderedDict
from functools import partial
from typing import Callable, Optional

import torch
import torch.nn as nn
from torch import Tensor
import einops
from einops import repeat

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


class DropPath(nn.Module):
    """
    Drop paths (Stochastic Depth) per sample  (when applied in main path of residual blocks).
    "Deep Networks with Stochastic Depth", https://arxiv.org/pdf/1603.09382.pdf
    """

    def __init__(self, drop_prob=None):
        super(DropPath, self).__init__()
        self.drop_prob = drop_prob

    def forward(self, x):
        return drop_path(x, self.drop_prob, self.training)


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
                 input_c: int,   # block input channel
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


class EfficientAdditiveAttention(nn.Module):
    """
    Efficient Additive Attention module for SwiftFormer.
    Input: tensor in shape [B, N, D]
    Output: tensor in shape [B, N, D]
    """

    def __init__(self, in_dims=256, token_dim=256, num_heads=2):
        super().__init__()


        self.in_dims = in_dims
        self.token_dim = token_dim
        self.num_heads = num_heads


        self.to_query = nn.Linear(in_dims, token_dim * num_heads)
        self.to_key = nn.Linear(in_dims, token_dim * num_heads)
        self.w_g = nn.Parameter(torch.randn(token_dim * num_heads, 1))
        self.scale_factor = token_dim ** -0.5
        self.Proj = nn.Linear(token_dim * num_heads, token_dim * num_heads)
        self.final = nn.Linear(token_dim * num_heads, token_dim)

    def forward(self, x):


        # 特征图转token序列：[B, C, H, W] -> [B, N, C]（N=H*W）
        b, c, h, w = x.shape
        x = x.permute(0, 2, 3, 1).reshape(b, -1, c)  # [B, N, C]


        query = self.to_query(x)
        key = self.to_key(x)
        query = torch.nn.functional.normalize(query, dim=-1)  # BxNxD
        key = torch.nn.functional.normalize(key, dim=-1)  # BxNxD
        query_weight = query @ self.w_g  # BxNx1 (BxNxD @ Dx1)
        A = query_weight * self.scale_factor  # BxNx1
        A = torch.nn.functional.normalize(A, dim=1)  # BxNx1
        G = torch.sum(A * query, dim=1)  # BxD
        G = einops.repeat(
            G, "b d -> b repeat d", repeat=key.shape[1]
        )  # BxNxD
        out = self.Proj(G * key) + query  # BxNxD
        out = self.final(out)  # BxNxD


        # 恢复特征图形状：[B, N, token_dim] -> [B, token_dim, H, W]
        out = out.reshape(b, h, w, -1).permute(0, 3, 1, 2)  # [B, token_dim, H, W]


        return out


# 定义多查询注意力模块，继承自torch.nn.Module
class MutiQueryAttention(torch.nn.Module):
    # 初始化函数，hidden_size为隐藏层大小，num_heads为注意力头的数量
    def __init__(self, hidden_size, num_heads):
        # 调用父类的初始化方法
        super(MutiQueryAttention, self).__init__()
        # 保存注意力头的数量
        self.num_heads = num_heads
        # 计算每个注意力头的维度（假设hidden_size可以被num_heads整除）
        self.head_dim = hidden_size // num_heads

        ## 初始化Q、K、V的线性投影层
        # 定义用于生成查询向量的全连接层，输入和输出的维度均为hidden_size
        self.q_linear = nn.Linear(hidden_size, hidden_size)
        # 定义用于生成键向量的全连接层，输出维度为head_dim
        self.k_linear = nn.Linear(hidden_size, self.head_dim)  ###
        # 定义用于生成值向量的全连接层，输出维度为head_dim
        self.v_linear = nn.Linear(hidden_size, self.head_dim)  ###

        ## 初始化输出全连接层，用于整合各注意力头的输出
        self.o_linear = nn.Linear(hidden_size, hidden_size)

    # 定义前向传播函数，hidden_state为输入的隐藏状态，attention_mask为可选的注意力掩码
    def forward(self, hidden_state, attention_mask=None):
        # 获取批次大小，从hidden_state的第一个维度获得
        batch_size = hidden_state.size()[0]

        # 通过q_linear全连接层生成查询向量
        query = self.q_linear(hidden_state)
        # 通过k_linear全连接层生成键向量
        key = self.k_linear(hidden_state)
        # 通过v_linear全连接层生成值向量
        value = self.v_linear(hidden_state)

        # 将查询向量拆分为多个注意力头
        query = self.split_head(query)
        # 将键向量拆分为多个注意力头，传入head_num参数为1
        key = self.split_head(key, 1)
        # 将值向量拆分为多个注意力头，传入head_num参数为1
        value = self.split_head(value, 1)

        ## 计算注意力分数
        # 计算查询和键向量的点积，并对最后一个维度进行转置，再除以head_dim的平方根进行缩放
        attention_scores = torch.matmul(query, key.transpose(-1, -2)) / torch.sqrt(torch.tensor(self.head_dim))

        # 如果提供了注意力掩码，则将其应用于注意力分数（乘以一个很小的负数因子）
        if attention_mask != None:
            attention_scores += attention_mask * -1e-9

        ## 对注意力分数进行归一化
        # 对注意力分数沿着最后一个维度使用softmax函数归一化，得到注意力概率
        attention_probs = torch.softmax(attention_scores, dim=-1)

        # 用归一化的注意力概率对值向量进行加权求和，得到注意力输出
        output = torch.matmul(attention_probs, value)

        # 将输出张量的最后两个维度进行转置，调用contiguous保证内存连续性，
        # 再reshape为(batch_size, 序列长度, head_dim * num_heads)
        output = output.transpose(-1, -2).contiguous().view(batch_size, -1, self.head_dim * self.num_heads)

        # 将整合后的输出通过输出全连接层进行最后的线性变换
        output = self.o_linear(output)

        # 返回最终的注意力输出
        return output








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
        #jizhi++++
        self.eaa = EfficientAdditiveAttention(expanded_c) if se_ratio > 0 else nn.Identity()


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

            # 新增：MultiQueryAttention
            if self.use_mqa:
                b, c, h, w = x.shape
                # 调整形状：(B, C, H, W) -> (B, H*W, C)
                x_flat = x.permute(0, 2, 3, 1).view(b, h * w, c)
                # 应用注意力
                x_attn = self.mqa(x_flat)
                # 恢复形状：(B, H*W, C) -> (B, C, H, W)
                x_attn = x_attn.view(b, h, w, c).permute(0, 3, 1, 2)
                # 残差连接
                x = x + x_attn
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


class EfficientNetV2(nn.Module):
    def __init__(self,
                 model_cnf: list,
                 num_classes: int = 1000,
                 num_features: int = 1280,
                  #dropout_rate: float = 0.2,
                 #drop_connect_rate: float = 0.2
                 dropout_rate = 0.5,  # 原0.2 → 增大分类头部的Dropout
                 drop_connect_rate = 0.5  # 原0.2 → 增大残差连接中的DropPath概率
                 ):
        super(EfficientNetV2, self).__init__()




        for cnf in model_cnf:
            assert len(cnf) == 8

        norm_layer = partial(nn.BatchNorm2d, eps=1e-3, momentum=0.1)

        stem_filter_num = model_cnf[0][4]

        self.stem = ConvBNAct(3,
                              stem_filter_num,
                              kernel_size=3,
                              stride=2,
                              norm_layer=norm_layer)  # 激活函数默认是SiLU

        total_blocks = sum([i[0] for i in model_cnf])
        block_id = 0
        blocks = []
        for cnf in model_cnf:
            repeats = cnf[0]
            op = FusedMBConv if cnf[-2] == 0 else MBConv
            for i in range(repeats):
                blocks.append(op(kernel_size=cnf[1],
                                 input_c=cnf[4] if i == 0 else cnf[5],
                                 out_c=cnf[5],
                                 expand_ratio=cnf[3],
                                 stride=cnf[2] if i == 0 else 1,
                                 se_ratio=cnf[-1],
                                 drop_rate=drop_connect_rate * block_id / total_blocks,
                                 norm_layer=norm_layer))

                block_id += 1










                # 在每个stage的最后一个block后添加注意力
                if i == repeats - 1:
                    out_c = cnf[5]
                    blocks.append(EfficientAdditiveAttention(
                        in_dims=out_c,
                        token_dim=out_c,
                        num_heads=2 if out_c >= 128 else 1  # 动态调整头数
                    ))

            self.blocks = nn.Sequential(*blocks)

            # ... 后续代码保持不变 ...







        self.blocks = nn.Sequential(*blocks)

        head_input_c = model_cnf[-1][-3]
        head = OrderedDict()

        head.update({"project_conv": ConvBNAct(head_input_c,
                                               num_features,
                                               kernel_size=1,
                                               norm_layer=norm_layer)})  # 激活函数默认是SiLU

        head.update({"avgpool": nn.AdaptiveAvgPool2d(1)})
        head.update({"flatten": nn.Flatten()})

        if dropout_rate > 0:
            head.update({"dropout": nn.Dropout(p=dropout_rate, inplace=True)})
        head.update({"classifier": nn.Linear(num_features, num_classes)})

        self.head = nn.Sequential(head)

        # initial weights
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
        x = self.stem(x)
        x = self.blocks(x)
        x = self.head(x)

        return x


def efficientnetv2_s(num_classes: int = 1000):
    """
    EfficientNetV2
    https://arxiv.org/abs/2104.00298
    """
    # train_size: 300, eval_size: 384

    # repeat, kernel, stride, expansion, in_c, out_c, operator, se_ratio
    model_config = [[2, 3, 1, 1, 24, 24, 0, 0],
                    [4, 3, 2, 4, 24, 48, 0, 0],
                    [4, 3, 2, 4, 48, 64, 0, 0],
                    [6, 3, 2, 4, 64, 128, 1, 0.25],
                    [9, 3, 1, 6, 128, 160, 1, 0.25],
                    [15, 3, 2, 6, 160, 256, 1, 0.25]]

    model = EfficientNetV2(model_cnf=model_config,
                           num_classes=num_classes,
                           dropout_rate=0.2)
    return model


def efficientnetv2_m(num_classes: int = 1000):
    """
    EfficientNetV2
    https://arxiv.org/abs/2104.00298
    """
    # train_size: 384, eval_size: 480

    # repeat, kernel, stride, expansion, in_c, out_c, operator, se_ratio
    model_config = [[3, 3, 1, 1, 24, 24, 0, 0],
                    [5, 3, 2, 4, 24, 48, 0, 0],
                    [5, 3, 2, 4, 48, 80, 0, 0],
                    [7, 3, 2, 4, 80, 160, 1, 0.25],
                    [14, 3, 1, 6, 160, 176, 1, 0.25],
                    [18, 3, 2, 6, 176, 304, 1, 0.25],
                    [5, 3, 1, 6, 304, 512, 1, 0.25]]

    model = EfficientNetV2(model_cnf=model_config,
                           num_classes=num_classes,
                           dropout_rate=0.3)
    return model


def efficientnetv2_l(num_classes: int = 1000):
    """
    EfficientNetV2
    https://arxiv.org/abs/2104.00298
    """
    # train_size: 384, eval_size: 480

    # repeat, kernel, stride, expansion, in_c, out_c, operator, se_ratio
    model_config = [[4, 3, 1, 1, 32, 32, 0, 0],
                    [7, 3, 2, 4, 32, 64, 0, 0],
                    [7, 3, 2, 4, 64, 96, 0, 0],
                    [10, 3, 2, 4, 96, 192, 1, 0.25],
                    [19, 3, 1, 6, 192, 224, 1, 0.25],
                    [25, 3, 2, 6, 224, 384, 1, 0.25],
                    [7, 3, 1, 6, 384, 640, 1, 0.25]]

    model = EfficientNetV2(model_cnf=model_config,
                           num_classes=num_classes,
                           dropout_rate=0.4)
    return model
'''

from collections import OrderedDict
from functools import partial
from typing import Callable, Optional

import torch
import torch.nn as nn
from torch import Tensor
import einops
from einops import repeat


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
    random_tensor = keep_prob + torch.rand(shape, dtype=x.dtype, device=x.device)
    random_tensor.floor_()  # binarize
    output = x.div(keep_prob) * random_tensor
    return output


class DropPath(nn.Module):
    """
    Drop paths (Stochastic Depth) per sample  (when applied in main path of residual blocks).
    "Deep Networks with Stochastic Depth", https://arxiv.org/pdf/1603.09382.pdf
    """

    def __init__(self, drop_prob=None):
        super(DropPath, self).__init__()
        self.drop_prob = drop_prob

    def forward(self, x):
        return drop_path(x, self.drop_prob, self.training)


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

'''
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
        return scale * 
'''



# 定义多查询注意力模块，继承自torch.nn.Module
class MutiQueryAttention(torch.nn.Module):
    # 初始化函数，hidden_size为隐藏层大小，num_heads为注意力头的数量
    def __init__(self, hidden_size, num_heads):
        # 调用父类的初始化方法
        super(MutiQueryAttention, self).__init__()
        # 保存注意力头的数量
        self.num_heads = num_heads
        # 计算每个注意力头的维度（假设hidden_size可以被num_heads整除）
        self.head_dim = hidden_size // num_heads

        ## 初始化Q、K、V的线性投影层
        # 定义用于生成查询向量的全连接层，输入和输出的维度均为hidden_size
        self.q_linear = nn.Linear(hidden_size, hidden_size)
        # 定义用于生成键向量的全连接层，输出维度为head_dim
        self.k_linear = nn.Linear(hidden_size, self.head_dim)
        # 定义用于生成值向量的全连接层，输出维度为head_dim
        self.v_linear = nn.Linear(hidden_size, self.head_dim)

        ## 初始化输出全连接层，用于整合各注意力头的输出
        self.o_linear = nn.Linear(hidden_size, hidden_size)

    # 定义前向传播函数，hidden_state为输入的隐藏状态，attention_mask为可选的注意力掩码
    def forward(self, hidden_state, attention_mask=None):
        # 获取批次大小，从hidden_state的第一个维度获得
        batch_size = hidden_state.size()[0]

        # 通过q_linear全连接层生成查询向量
        query = self.q_linear(hidden_state)
        # 通过k_linear全连接层生成键向量
        key = self.k_linear(hidden_state)
        # 通过v_linear全连接层生成值向量
        value = self.v_linear(hidden_state)

        # 将查询向量拆分为多个注意力头
        query = self.split_head(query)
        # 将键向量拆分为多个注意力头，传入head_num参数为1
        key = self.split_head(key, 1)
        # 将值向量拆分为多个注意力头，传入head_num参数为1
        value = self.split_head(value, 1)

        ## 计算注意力分数
        # 计算查询和键向量的点积，并对最后一个维度进行转置，再除以head_dim的平方根进行缩放
        attention_scores = torch.matmul(query, key.transpose(-1, -2)) / torch.sqrt(torch.tensor(self.head_dim))

        # 如果提供了注意力掩码，则将其应用于注意力分数（乘以一个很小的负数因子）
        if attention_mask is not None:
            attention_scores += attention_mask * -1e-9

        ## 对注意力分数进行归一化
        # 对注意力分数沿着最后一个维度使用softmax函数归一化，得到注意力概率
        attention_probs = torch.softmax(attention_scores, dim=-1)

        # 用归一化的注意力概率对值向量进行加权求和，得到注意力输出
        output = torch.matmul(attention_probs, value)

        # 将输出张量的最后两个维度进行转置，调用contiguous保证内存连续性，
        # 再reshape为(batch_size, 序列长度, head_dim * num_heads)
        output = output.transpose(-1, -2).contiguous().view(batch_size, -1, self.head_dim * self.num_heads)

        # 将整合后的输出通过输出全连接层进行最后的线性变换
        output = self.o_linear(output)

        # 返回最终的注意力输出
        return output

    def split_head(self, x, head_num=None):
        if head_num is None:
            head_num = self.num_heads
        batch_size, seq_length, hidden_size = x.size()
        return x.view(batch_size, seq_length, head_num, hidden_size // head_num).transpose(1, 2)











class EfficientAdditiveAttention(nn.Module):
    """
    Efficient Additive Attention module for SwiftFormer.
    Input: tensor in shape [B, N, D]
    Output: tensor in shape [B, N, D]
    """

    def __init__(self, in_dims=256, token_dim=256, num_heads=2):
        super().__init__()

        self.in_dims = in_dims
        self.token_dim = token_dim
        self.num_heads = num_heads

        self.to_query = nn.Linear(in_dims, token_dim * num_heads)
        self.to_key = nn.Linear(in_dims, token_dim * num_heads)
        self.w_g = nn.Parameter(torch.randn(token_dim * num_heads, 1))
        self.scale_factor = token_dim ** -0.5
        self.Proj = nn.Linear(token_dim * num_heads, token_dim * num_heads)
        self.final = nn.Linear(token_dim * num_heads, token_dim)

    def forward(self, x):
        # 特征图转token序列：[B, C, H, W] -> [B, N, C]（N=H*W）
        b, c, h, w = x.shape
        x = x.permute(0, 2, 3, 1).reshape(b, -1, c)  # [B, N, C]

        query = self.to_query(x)
        key = self.to_key(x)
        query = torch.nn.functional.normalize(query, dim=-1)  # BxNxD
        key = torch.nn.functional.normalize(key, dim=-1)  # BxNxD
        query_weight = query @ self.w_g  # BxNx1 (BxNxD @ Dx1)
        A = query_weight * self.scale_factor  # BxNx1
        A = torch.nn.functional.normalize(A, dim=1)  # BxNx1
        G = torch.sum(A * query, dim=1)  # BxD
        G = einops.repeat(
            G, "b d -> b repeat d", repeat=key.shape[1]
        )  # BxNxD
        out = self.Proj(G * key) + query  # BxNxD
        out = self.final(out)  # BxNxD

        # 恢复特征图形状：[B, N, token_dim] -> [B, token_dim, H, W]
        out = out.reshape(b, h, w, -1).permute(0, 3, 1, 2)  # [B, token_dim, H, W]

        return out



class MBConv(nn.Module):
    def __init__(self,
                 kernel_size: int,
                 input_c: int,
                 out_c: int,
                 expand_ratio: int,
                 stride: int,
                 se_ratio: float,
                 drop_rate: float,
                 norm_layer: Callable[..., nn.Module],
                 use_mqa: bool = False,#外加
                 num_heads: int = 8):#外加
        super(MBConv, self).__init__()

        if stride not in [1, 2]:
            raise ValueError("illegal stride value.")

        self.has_shortcut = (stride == 1 and input_c == out_c)
        self.use_mqa = use_mqa#外加

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

        #self.se = SqueezeExcite(input_c, expanded_c,se_ratio) if se_ratio > 0 else nn.Identity()
        self.se = nn.Identity()  # 替代原SqueezeExcite
        # jizhi++++
        self.eaa = EfficientAdditiveAttention(expanded_c) if se_ratio > 0 else nn.Identity()

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

        if self.use_mqa:
            self.mqa = MutiQueryAttention(out_c, num_heads)

    def forward(self, x: Tensor) -> Tensor:
        result = self.expand_conv(x)
        result = self.dwconv(result)
        result = self.se(result)
        result = self.project_conv(result)

        if self.has_shortcut:
            if self.drop_rate > 0:
                result = self.dropout(result)
            result += x




            # 新增：MultiQueryAttention
            if self.use_mqa:
                b, c, h, w = x.shape
                # 调整形状：(B, C, H, W) -> (B, H*W, C)
                x_flat = x.permute(0, 2, 3, 1).view(b, h * w, c)
                # 应用注意力
                x_attn = self.mqa(x_flat)
                # 恢复形状：(B, H*W, C) -> (B, C, H, W)
                x_attn = x_attn.view(b, h, w, c).permute(0, 3, 1, 2)
                # 残差连接
                result = result + x_attn

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
                 norm_layer: Callable[..., nn.Module],
                 use_mqa: bool = False,#外加
                 num_heads: int = 8):#外加
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


class EfficientNetV2(nn.Module):
    def __init__(self,
                 model_cnf: list,
                 num_classes: int = 1000,
                 num_features: int = 1280,
                 dropout_rate=0.3,  # 原0.2 → 增大分类头部的Dropout
                 drop_connect_rate=0.3,  # 原0.2 → 增大残差连接中的DropPath概率
                 use_mqa: bool = False,#外加
                 num_heads: int = 8):#外加
        super(EfficientNetV2, self).__init__()

        for cnf in model_cnf:
            assert len(cnf) == 8

        norm_layer = partial(nn.BatchNorm2d, eps=1e-3, momentum=0.1)

        stem_filter_num = model_cnf[0][4]

        self.stem = ConvBNAct(3,
                              stem_filter_num,
                              kernel_size=3,
                              stride=2,
                              norm_layer=norm_layer)  # 激活函数默认是SiLU

        total_blocks = sum([i[0] for i in model_cnf])
        block_id = 0
        blocks = []
        for cnf in model_cnf:
            repeats = cnf[0]
            op = FusedMBConv if cnf[-2] == 0 else MBConv
            for i in range(repeats):
                blocks.append(op(kernel_size=cnf[1],
                                 input_c=cnf[4] if i == 0 else cnf[5],
                                 out_c=cnf[5],
                                 expand_ratio=cnf[3],
                                 stride=cnf[2] if i == 0 else 1,
                                 se_ratio=cnf[-1],
                                 drop_rate=drop_connect_rate * block_id / total_blocks,
                                 norm_layer=norm_layer,
                                 use_mqa=use_mqa,#外加
                                 num_heads=num_heads))#外加

                block_id += 1

        self.blocks = nn.Sequential(*blocks)

        head_input_c = model_cnf[-1][-3]
        head = OrderedDict()

        head.update({"project_conv": ConvBNAct(head_input_c,
                                               num_features,
                                               kernel_size=1,
                                               norm_layer=norm_layer)})  # 激活函数默认是SiLU

        head.update({"avgpool": nn.AdaptiveAvgPool2d(1)})
        head.update({"flatten": nn.Flatten()})

        if dropout_rate > 0:
            head.update({"dropout": nn.Dropout(p=dropout_rate, inplace=True)})
        head.update({"classifier": nn.Linear(num_features, num_classes)})

        self.head = nn.Sequential(head)

        # initial weights
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
        x = self.stem(x)
        x = self.blocks(x)
        x = self.head(x)

        return x


def efficientnetv2_s(num_classes: int = 1000, use_mqa: bool = False, num_heads: int = 8):
    """
    EfficientNetV2
    https://arxiv.org/abs/2104.00298
    """
    # train_size: 300, eval_size: 384

    # repeat, kernel, stride, expansion, in_c, out_c, operator, se_ratio
    model_config = [[2, 3, 1, 1, 24, 24, 0, 0],
                    [4, 3, 2, 4, 24, 48, 0, 0],
                    [4, 3, 2, 4, 48, 64, 0, 0],
                    [6, 3, 2, 4, 64, 128, 1, 0.25],
                    [9, 3, 1, 6, 128, 160, 1, 0.25],
                    [15, 3, 2, 6, 160, 256, 1, 0.25]]

    model = EfficientNetV2(model_cnf=model_config,
                           num_classes=num_classes,
                           dropout_rate=0.2,
                           use_mqa=use_mqa,#外加
                           num_heads=num_heads)#外加
    return model


def efficientnetv2_m(num_classes: int = 1000, use_mqa: bool = False, num_heads: int = 8):
    """
    EfficientNetV2
    https://arxiv.org/abs/2104.00298
    """
    # train_size: 384, eval_size: 480

    # repeat, kernel, stride, expansion, in_c, out_c, operator, se_ratio
    model_config = [[3, 3, 1, 1, 24, 24, 0, 0],
                    [5, 3, 2, 4, 24, 48, 0, 0],
                    [5, 3, 2, 4, 48, 80, 0, 0],
                    [7, 3, 2, 4, 80, 160, 1, 0.25],
                    [14, 3, 1, 6, 160, 176, 1, 0.25],
                    [18, 3, 2, 6, 176, 304, 1, 0.25],
                    [5, 3, 1, 6, 304, 512, 1, 0.25]]

    model = EfficientNetV2(model_cnf=model_config,
                           num_classes=num_classes,
                           dropout_rate=0.3,
                           use_mqa=use_mqa,
                           num_heads=num_heads)
    return model


def efficientnetv2_l(num_classes: int = 1000, use_mqa: bool = False, num_heads: int = 8):
    """
    EfficientNetV2
    https://arxiv.org/abs/2104.00298
    """
    # train_size: 384, eval_size: 480

    # repeat, kernel, stride, expansion, in_c, out_c, operator, se_ratio
    model_config = [[4, 3, 1, 1, 32, 32, 0, 0],
                    [7, 3, 2, 4, 32, 64, 0, 0],
                    [7, 3, 2, 4, 64, 96, 0, 0],
                    [10, 3, 2, 4, 96, 192, 1, 0.25],
                    [19, 3, 1, 6, 192, 224, 1, 0.25],
                    [25, 3, 2, 6, 224, 384, 1, 0.25],
                    [7, 3, 1, 6, 384, 640, 1, 0.25]]

    model = EfficientNetV2(model_cnf=model_config,
                           num_classes=num_classes,
                           dropout_rate=0.4,
                           use_mqa=use_mqa,
                           num_heads=num_heads)
    return model

