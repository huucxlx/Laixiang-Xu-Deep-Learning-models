import os
import sys
import json
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import transforms, datasets
from tqdm import tqdm
import matplotlib.pyplot as plt
from model__both_enhanced02 import DualAttentionAlexNet  # 确保这是包含SmartFusion的版本



def denormalize(tensor, mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5)):
    """反归一化用于可视化"""
    tensor = tensor.clone()
    for t, m, s in zip(tensor, mean, std):
        t.mul_(s).add_(m)
    return torch.clamp(tensor, 0, 1)


def main():
    # 训练配置
    config = {
        'phases': {
            'warmup': 30,  # 前30轮只训HybridCS
            'full_train': 70,  # 后70轮联合训练
            'total_epochs': 100
        },
        'lr_config': {
            'hybrid_lr': 1e-4,
            'coord_lr': 5e-5,
            'base_lr': 1e-4
        },
        'regularization': {
            'weight_decay': 1e-5,
            'dropout': 0.5
        },
        'save_dir': 'checkpoints',
        'attn_vis_dir': 'attention_vis'
    }

    # 创建保存目录
    os.makedirs(config['save_dir'], exist_ok=True)
    os.makedirs(config['attn_vis_dir'], exist_ok=True)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"Using {device} device for training")

    # 数据预处理
    data_transform = {
        "train": transforms.Compose([
            transforms.RandomResizedCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ]),
        "val": transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])
    }

    # 数据集路径
    data_root = r"D:\deep_leaning\Alexnet\card-data (2)\card-data"
    train_path = os.path.join(data_root, "train")
    val_path = os.path.join(data_root, "val")

    # 检查路径
    assert os.path.exists(train_path), f"训练集路径不存在: {train_path}"
    assert os.path.exists(val_path), f"验证集路径不存在: {val_path}"

    # 加载数据集
    train_dataset = datasets.ImageFolder(root=train_path, transform=data_transform["train"])
    val_dataset = datasets.ImageFolder(root=val_path, transform=data_transform["val"])

    # 类别映射
    cla_dict = {v: k for k, v in train_dataset.class_to_idx.items()}
    with open('class_indices.json', 'w') as f:
        json.dump(cla_dict, f, indent=4)

    # 数据加载器
    batch_size = 32
    nw = min([os.cpu_count(), batch_size if batch_size > 1 else 0, 8])
    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True, num_workers=nw)
    val_loader = torch.utils.data.DataLoader(
        val_dataset, batch_size=batch_size, shuffle=False, num_workers=nw)

    # 初始化模型
    model = DualAttentionAlexNet(num_classes=len(cla_dict)).to(device)

    # 分层学习率优化器
    optimizer_params = [
        {'params': [p for n, p in model.named_parameters() if 'hybrid' in n],
         'lr': config['lr_config']['hybrid_lr']},
        {'params': [p for n, p in model.named_parameters() if 'coord' in n],
         'lr': config['lr_config']['coord_lr']},
        {'params': [p for n, p in model.named_parameters() if ('hybrid' not in n) and ('coord' not in n)],
         'lr': config['lr_config']['base_lr']}
    ]
    optimizer = optim.AdamW(optimizer_params, weight_decay=config['regularization']['weight_decay'])
    criterion = nn.CrossEntropyLoss()

    # 训练记录
    history = {
        'train_loss': [],
        'train_acc': [],
        'val_loss': [],
        'val_acc': [],
        'attention_weights': []
    }
    best_acc = 0.0

    # 训练循环
    for epoch in range(config['phases']['total_epochs']):
        # === 阶段控制 ===
        if epoch < config['phases']['warmup']:
            # 冻结CoordAttention参数
            for n, p in model.named_parameters():
                if 'coord' in n:
                    p.requires_grad = False
        else:
            # 解冻所有参数
            for p in model.parameters():
                p.requires_grad = True

        # === 训练阶段 ===
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0

        train_bar = tqdm(train_loader, desc=f'Train Epoch [{epoch + 1}/{config["phases"]["total_epochs"]}]',
                         file=sys.stdout)
        for images, labels in train_bar:
            images, labels = images.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()

            # 梯度裁剪
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            running_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

            train_bar.set_postfix({'Loss': loss.item()})

        train_loss = running_loss / len(train_loader)
        train_acc = correct / total
        history['train_loss'].append(train_loss)
        history['train_acc'].append(train_acc)

        # === 验证阶段 ===
        model.eval()
        val_loss = 0.0
        correct = 0
        total = 0

        with torch.no_grad():
            val_bar = tqdm(val_loader, desc='Validating', file=sys.stdout)
            for images, labels in val_bar:
                images, labels = images.to(device), labels.to(device)

                outputs = model(images)
                loss = criterion(outputs, labels)
                val_loss += loss.item()

                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

        val_loss = val_loss / len(val_loader)
        val_acc = correct / total
        history['val_loss'].append(val_loss)
        history['val_acc'].append(val_acc)

        # === 记录注意力权重 ===
        current_weights = {}
        for name, module in model.named_modules():
            if hasattr(module, 'weight') and isinstance(module.weight, nn.Parameter):
                if module.weight.numel() == 2:  # 只记录SmartFusion的权重
                    current_weights[name] = module.weight.data.cpu().numpy()
        history['attention_weights'].append(current_weights)

        # === 保存检查点 ===
        if val_acc > best_acc:
            best_acc = val_acc
            torch.save({
                'epoch': epoch,
                'state_dict': model.state_dict(),
                'optimizer': optimizer.state_dict(),
                'best_acc': best_acc,
                'config': config
            }, os.path.join(config['save_dir'], 'best_model.pth'))

        # === 可视化 ===
        if (epoch + 1) % 10 == 0:
            # 绘制训练曲线
            plt.figure(figsize=(12, 5))
            plt.subplot(1, 2, 1)
            plt.plot(history['train_loss'], label='Train')
            plt.plot(history['val_loss'], label='Val')
            plt.title('Loss Curve')
            plt.legend()

            plt.subplot(1, 2, 2)
            plt.plot(history['train_acc'], label='Train')
            plt.plot(history['val_acc'], label='Val')
            plt.title('Accuracy Curve')
            plt.legend()

            plt.savefig(os.path.join(config['attn_vis_dir'], f'curve_epoch{epoch + 1}.png'))
            plt.close()

            # 注意力权重变化
            if len(history['attention_weights']) > 1:
                weights = np.array([list(w.values())[0] for w in history['attention_weights'] if w])
                plt.figure()
                plt.plot(weights[:, 0], label='HybridCS weight')
                plt.plot(weights[:, 1], label='Coord weight')
                plt.title('Attention Weights Change')
                plt.legend()
                plt.savefig(os.path.join(config['attn_vis_dir'], 'attention_weights.png'))
                plt.close()

        # 打印epoch结果
        print(f'Epoch {epoch + 1}/{config["phases"]["total_epochs"]} | '
              f'Train Loss: {train_loss:.4f} Acc: {train_acc:.4f} | '
              f'Val Loss: {val_loss:.4f} Acc: {val_acc:.4f} | '
              f'Best Acc: {best_acc:.4f}')

        # 打印当前注意力权重
        for name, weight in current_weights.items():
            print(f'{name} weights: {weight}')

    print(f'\nTraining completed! Best val acc: {best_acc:.4f}')


if __name__ == '__main__':
    main()