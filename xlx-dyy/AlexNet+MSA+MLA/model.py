import torch.nn as nn
import torch
from msa import MultiHeadSelfAttention
from mla import MultiHeadLatentAttention


class AlexNet(nn.Module):
    def __init__(
            self,
            num_classes: int = 4,
            init_weights: bool = False,
            use_msa: bool = True,
            # msa_positions: list = [13],
            msa_embed_ratio: float = 0.25,
            use_mla: bool = True,
            mla_positions: list = [4, 5]
    ):
        super().__init__()
        # 基础卷积层配置（所有ReLU禁用inplace）
        base_layers = [
            # 输入尺寸: 224x224x3
            nn.Conv2d(3, 48, kernel_size=11, stride=4, padding=2),  # 输出: 55x55x48
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),  # 输出: 27x27x48
            # *self._build_msa_block(48),
            nn.Conv2d(48, 128, kernel_size=5, padding=2),  # 输出: 27x27x128
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),  # 输出: 13x13x128
            # *self._build_msa_block(128),
            nn.Conv2d(128, 192, kernel_size=3, padding=1),  # 输出: 13x13x192
            nn.ReLU(inplace=True),

            nn.Conv2d(192, 256, kernel_size=3, padding=1),  # 输出: 13x13x256
            nn.ReLU(inplace=True),
            # *self._build_msa_block(128),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),  # 输出: 13x13x256
            nn.ReLU(inplace=True),

            nn.Conv2d(256, 128, kernel_size=3, padding=1),  # 输出: 13x13x128
            nn.ReLU(inplace=True),
            # *self._build_msa_block(128),
            nn.MaxPool2d(kernel_size=3, stride=2),  # 输出: 6x6x128
            * self._build_msa_block(128)
        ]

        # 动态插入注意力模块
        # if use_msa:
        #     self._insert_attention_blocks(
        #         layers=base_layers,
        #         positions=msa_positions,
        #         block_type='msa',
        #         embed_ratio=msa_embed_ratio
        #     )
        if use_mla:
            self._insert_attention_blocks(
                layers=base_layers,
                positions=mla_positions,
                block_type='mla'
            )

        self.features = nn.Sequential(*base_layers)
        self.classifier = self._build_classifier(128 * 6 * 6, num_classes)

        if init_weights:
            self._initialize_weights(use_msa, use_mla)

    def _insert_attention_blocks(self, layers, positions, block_type, embed_ratio=0.25):
        """通用注意力模块插入方法"""
        # 确定目标位置索引
        target_indices = []
        if block_type == 'mla':
            conv_indices = [i for i, layer in enumerate(layers) if isinstance(layer, nn.Conv2d)]
            target_indices = [conv_indices[pos] for pos in positions if pos < len(conv_indices)]
        else:
            target_indices = [pos for pos in positions if pos < len(layers)]

        # # 反向插入避免索引错位
        for idx in sorted(target_indices, reverse=True):
            # if block_type == 'msa':
            #     self._insert_msa_block(layers, idx, embed_ratio)
            if block_type == 'mla':
                self._insert_mla_block(layers, idx)

    # def _insert_msa_block(self, layers, idx, embed_ratio):
    #     """插入多头自注意力模块"""
    #     in_channels = layers[idx].out_channels
    #     embed_dim = max(4, int(in_channels * embed_ratio))
    #     embed_dim = (embed_dim // 4) * 4  # 确保能被头数整除
    #
    #     msa_block = nn.Sequential(
    #         nn.Conv2d(in_channels, embed_dim, 1),
    #         nn.ReLU(inplace=False),
    #         MultiHeadSelfAttention(embed_dim=embed_dim, num_heads=4),
    #         nn.Conv2d(embed_dim, in_channels, 1),
    #         nn.ReLU(inplace=False)
    #     )
    #     layers.insert(idx + 1, msa_block)
    def _build_msa_block(self, channels):
        """安全构建MSA模块"""
        embed_dim = channels // 4
        if (embed_dim % 4 != 0) or (embed_dim < 1):
            return []  # 跳过无效通道数
        return [
            nn.Conv2d(channels, embed_dim, kernel_size=1),  # 通道压缩
            nn.ReLU(inplace=True),
            MultiHeadSelfAttention(embed_dim=embed_dim, num_heads=4),
            nn.Conv2d(embed_dim, channels, kernel_size=1),  # 通道恢复
            nn.ReLU(inplace=True)
        ]

    def _insert_mla_block(self, layers, idx):
        """插入多头潜在注意力模块"""
        out_channels = layers[idx].out_channels
        layers.insert(idx + 1, MultiHeadLatentAttention(out_channels))

    def _build_classifier(self, input_dim, num_classes):
        """构建增强型分类器"""
        return nn.Sequential(
            nn.Dropout(p=0.5),
            nn.Linear(input_dim, 2048),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.5),
            nn.Linear(2048, 2048),  # 增加特征交互层
            nn.ReLU(inplace=True),
            nn.Linear(2048, num_classes)
        )

    def forward(self, x):
        x = self.features(x)
        x = torch.flatten(x, 1)  # 保持batch维度
        x = self.classifier(x)
        return x

    def _initialize_weights(self, use_msa, use_mla):
        """统一初始化策略"""
        for m in self.modules():
            # 卷积层初始化
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

            # 全连接层统一初始化
            elif isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode='fan_in', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

            # 注意力模块特殊初始化
            if use_msa and isinstance(m, MultiHeadSelfAttention):
                nn.init.normal_(m.qkv.weight, mean=0, std=0.02)
                nn.init.normal_(m.proj.weight, mean=0, std=0.02)

            if use_mla and isinstance(m, MultiHeadLatentAttention):
                nn.init.normal_(m.q_proj.weight, mean=0, std=0.02)
                nn.init.normal_(m.kv_proj.weight, mean=0, std=0.02)
                nn.init.normal_(m.latent_proj.weight, mean=0, std=0.02)