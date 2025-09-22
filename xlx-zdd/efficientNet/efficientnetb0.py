# 在代码中添加自动下载逻辑（添加到模型创建部分之前）
import torch.hub
import os

weights_path = "./efficient0.pth"
if not os.path.exists(weights_path):
    print("正在下载预训练权重...")
    url = "https://github.com/lukemelas/EfficientNet-PyTorch/releases/download/1.0/efficientnet-b0-355c32eb.pth"
    torch.hub.download_url_to_file(url, weights_path)
    print("下载完成！")