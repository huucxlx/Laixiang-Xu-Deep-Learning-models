import torch
import torch.nn as nn


class HybridCS(nn.Module):
    def __init__(self, channels, reduction=16):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)

        self.channel_fc = nn.Sequential(
            nn.Linear(channels, channels // reduction),
            nn.ReLU(inplace=True),
            nn.Linear(channels // reduction, channels),
            nn.Sigmoid()
        )

        self.spatial_conv = nn.Conv2d(2, 1, kernel_size=7, padding=3)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        # 通道注意力
        b, c, _, _ = x.size()
        avg_out = self.channel_fc(self.avg_pool(x).view(b, c))
        max_out = self.channel_fc(self.max_pool(x).view(b, c))
        channel_out = (avg_out + max_out).view(b, c, 1, 1)

        # 空间注意力
        avg_pool = torch.mean(x, dim=1, keepdim=True)
        max_pool, _ = torch.max(x, dim=1, keepdim=True)
        spatial_out = self.sigmoid(self.spatial_conv(torch.cat([avg_pool, max_pool], dim=1)))

        return x * channel_out * spatial_out

    def get_attention(self, x):
        """返回通道注意力和空间注意力图"""
        b, c, h, w = x.size()

        # 通道注意力
        avg_out = self.channel_fc(self.avg_pool(x).view(b, c))
        max_out = self.channel_fc(self.max_pool(x).view(b, c))
        channel_att = (avg_out + max_out).view(b, c, 1, 1)

        # 空间注意力
        avg_pool = torch.mean(x, dim=1, keepdim=True)
        max_pool, _ = torch.max(x, dim=1, keepdim=True)
        spatial_att = self.sigmoid(self.spatial_conv(torch.cat([avg_pool, max_pool], dim=1)))

        return {'channel': channel_att, 'spatial': spatial_att}


class LiteCoordAttention(nn.Module):
    """简化版CoordAttention，减少计算量"""

    def __init__(self, channels, reduction=32):
        super().__init__()
        self.pool_h = nn.AdaptiveAvgPool2d((None, 1))
        self.pool_w = nn.AdaptiveAvgPool2d((1, None))

        mid_c = max(8, channels // reduction)  # 保证最小通道数

        self.conv1 = nn.Conv2d(channels, mid_c, kernel_size=1)
        self.conv_h = nn.Conv2d(mid_c, channels, kernel_size=1)
        self.conv_w = nn.Conv2d(mid_c, channels, kernel_size=1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        identity = x
        b, c, h, w = x.size()

        # 水平池化
        x_h = self.pool_h(x)  # [b,c,h,1]
        # 垂直池化
        x_w = self.pool_w(x).permute(0, 1, 3, 2)  # [b,c,w,1]

        # 联合处理
        y = torch.cat([x_h, x_w], dim=2)
        y = self.conv1(y)
        y = torch.relu(y)

        x_h, x_w = torch.split(y, [h, w], dim=2)
        x_w = x_w.permute(0, 1, 3, 2)

        a_h = self.sigmoid(self.conv_h(x_h))
        a_w = self.sigmoid(self.conv_w(x_w))

        return identity * a_h * a_w

    def get_attention(self, x):
        """返回水平和垂直注意力图"""
        b, c, h, w = x.size()

        x_h = self.pool_h(x)
        x_w = self.pool_w(x).permute(0, 1, 3, 2)

        y = torch.cat([x_h, x_w], dim=2)
        y = self.conv1(y)
        y = torch.relu(y)

        x_h, x_w = torch.split(y, [h, w], dim=2)
        x_w = x_w.permute(0, 1, 3, 2)

        a_h = self.sigmoid(self.conv_h(x_h))
        a_w = self.sigmoid(self.conv_w(x_w))

        return {'horizontal': a_h, 'vertical': a_w}


class SmartFusion(nn.Module):
    """自动平衡两种注意力的融合模块"""

    def __init__(self, channels):
        super().__init__()
        self.hybrid = HybridCS(channels)
        self.coord = LiteCoordAttention(channels)
        # 可学习的权重参数，初始偏向HybridCS
        self.weight = nn.Parameter(torch.tensor([0.7, 0.3]))

    def forward(self, x):
        hybrid_out = self.hybrid(x)
        coord_out = self.coord(x)
        # 动态加权融合
        return self.weight[0] * hybrid_out + self.weight[1] * coord_out

    def get_attention(self, x):
        """返回两种注意力的注意力图"""
        hybrid_att = self.hybrid.get_attention(x)
        coord_att = self.coord.get_attention(x)
        return {
            'hybrid': hybrid_att,
            'coord': coord_att,
            'weights': self.weight
        }


class DualAttentionAlexNet(nn.Module):
    def __init__(self, num_classes=4):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=11, stride=4, padding=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
            HybridCS(64),  # 第一注意力层

            nn.Conv2d(64, 128, kernel_size=5, padding=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
            SmartFusion(128),  # 第二注意力层

            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            LiteCoordAttention(256),  # 第三注意力层

            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
        )

        self.classifier = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(256 * 6 * 6, 1024),
            nn.ReLU(inplace=True),
            nn.Linear(1024, num_classes),
        )

    def forward(self, x):
        x = self.features(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x

    def get_attention_maps(self, x):
        """获取所有注意力层的注意力图"""
        attention_maps = {}
        current_x = x

        # 遍历所有层
        for name, module in self.features.named_children():
            current_x = module(current_x)  # 先通过模块

            # 如果是注意力模块，获取注意力图
            if isinstance(module, (HybridCS, LiteCoordAttention, SmartFusion)):
                if hasattr(module, 'get_attention'):
                    # 使用原始输入x来获取注意力图
                    attention_maps[name] = module.get_attention(current_x)

        return attention_maps


if __name__ == "__main__":
    model = DualAttentionAlexNet(num_classes=4)
    dummy_input = torch.randn(1, 3, 224, 224)

    # 测试前向传播
    output = model(dummy_input)
    print("模型输出形状:", output.shape)  # 应为 [1, 4]

    # 测试注意力图
    attention_maps = model.get_attention_maps(dummy_input)
    print("\n注意力图结构:")
    for name, att in attention_maps.items():
        print(f"{name}:")
        if isinstance(att, dict):
            for k, v in att.items():
                if isinstance(v, dict):
                    print(f"  {k}:")
                    for sub_k, sub_v in v.items():
                        print(f"    {sub_k}: {sub_v.shape if hasattr(sub_v, 'shape') else sub_v}")
                else:
                    print(f"  {k}: {v.shape if hasattr(v, 'shape') else v}")
        else:
            print(f"  {att.shape}")