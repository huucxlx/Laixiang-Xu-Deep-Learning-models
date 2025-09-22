import argparse
import os
import time
import torch
import torch.optim as optim
import torch.optim.lr_scheduler as lr_scheduler
from model import efficientnetv2_s as create_model
from torch.utils.data import DataLoader, Dataset, random_split
from torchvision import transforms
from torch.nn import CrossEntropyLoss
from PIL import Image
from tqdm import tqdm
import json


class ProgressLogger:
    """自定义进度和结果日志记录器"""

    def __init__(self, total_epochs):
        self.total_epochs = total_epochs
        self.epoch_start_time = None
        self.best_acc = 0.0

    def log_epoch_start(self, epoch):
        self.epoch_start_time = time.time()
        print(f"\n{'=' * 40}")
        print(f"Epoch {epoch + 1}/{self.total_epochs}")
        print(f"{'=' * 40}")

    def log_batch(self, epoch, batch_idx, num_batches, loss, acc, lr):
        progress = (batch_idx + 1) / num_batches * 100
        print(f"[train epoch {epoch + 1}] loss: {loss:.3f}, acc: {acc:.3f}: {progress:.0f}%", end='\r')

    def log_epoch_end(self, epoch, train_loss, train_acc, val_loss, val_acc, is_best=False):
        epoch_time = time.time() - self.epoch_start_time
        print(f"\n[train epoch {epoch + 1}] loss: {train_loss:.3f}, acc: {train_acc:.3f}: 100%")
        print(f"[valid epoch {epoch + 1}] loss: {val_loss:.3f}, acc: {val_acc:.3f}: 100%")
        if is_best:
            print(f"★ New best model saved (acc: {val_acc:.3f})")
        print(f"Time: {epoch_time:.0f}s")


class CustomDataset(Dataset):
    """自定义数据集类"""

    def __init__(self, data_path, transform=None):
        self.data = []
        self.class_to_idx = {}
        self.transform = transform

        # 收集图像路径和标签
        for class_idx, class_name in enumerate(sorted(os.listdir(data_path))):
            class_dir = os.path.join(data_path, class_name)
            if os.path.isdir(class_dir):
                self.class_to_idx[class_name] = class_idx
                for img_name in os.listdir(class_dir):
                    if img_name.lower().endswith(('.png', '.jpg', '.jpeg')):
                        img_path = os.path.join(class_dir, img_name)
                        self.data.append((img_path, class_idx))

        if not self.data:
            raise RuntimeError(f"No images found in {data_path}")

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        img_path, label = self.data[idx]
        try:
            img = Image.open(img_path).convert('RGB')
            if self.transform:
                img = self.transform(img)
            return img, label
        except Exception as e:
            print(f"Error loading image {img_path}: {str(e)}")
            # 返回一个空白图像避免训练中断
            blank_img = torch.zeros(3, 320, 320) if self.transform else Image.new('RGB', (320, 320))
            return blank_img, label


def main(args):
    # 设备检测和设置
    device = torch.device(f"cuda:{args.device}" if torch.cuda.is_available() and args.device != 'cpu' else 'cpu')
    print(f"Using device: {device}")
    if device.type == 'cuda':
        print(f"GPU: {torch.cuda.get_device_name(device)}")
        torch.cuda.empty_cache()

    # 确保权重目录存在
    os.makedirs("weights", exist_ok=True)

    # 数据增强
    data_transform = {
        "train": transforms.Compose([
            transforms.RandomResizedCrop(320, scale=(0.7, 1.0)),
            transforms.RandomHorizontalFlip(),
            transforms.RandomVerticalFlip(),
            transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3),
            transforms.RandomRotation(30),
            transforms.RandomAffine(degrees=0, translate=(0.15, 0.15)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
            transforms.RandomErasing(p=0.5, scale=(0.02, 0.2), ratio=(0.3, 3.3)),
        ]),
        "val": transforms.Compose([
            transforms.Resize(384),
            transforms.CenterCrop(320),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
    }

    # 加载完整数据集
    try:
        print(f"Loading dataset from: {args.data_path}")
        full_dataset = CustomDataset(args.data_path)
    except Exception as e:
        print(f"Error loading dataset: {str(e)}")
        return

    # 保存类别到索引的映射
    with open('class_indices.json', 'w') as f:
        json.dump(full_dataset.class_to_idx, f)

    # 自动检测类别数
    args.num_classes = len(full_dataset.class_to_idx)
    print(f"自动检测到类别数: {args.num_classes}")

    # 拆分训练集和验证集
    train_size = int(0.8 * len(full_dataset))
    val_size = len(full_dataset) - train_size
    train_dataset, val_dataset = random_split(full_dataset, [train_size, val_size])

    # 应用不同的transform
    train_dataset.dataset.transform = data_transform["train"]
    val_dataset.dataset.transform = data_transform["val"]

    # 数据加载器 - 优化GPU使用
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=min(4, os.cpu_count()),
        pin_memory=True,  # 加速数据传输到GPU
        persistent_workers=True
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=min(2, os.cpu_count()),
        pin_memory=True
    )

    # 模型初始化
    model = create_model(num_classes=args.num_classes).to(device)

    # 打印模型摘要
    print("\n模型结构摘要:")
    print(model)

    # 检查GPU内存
    if device.type == 'cuda':
        print(f"\nGPU内存使用情况:")
        print(f"分配: {torch.cuda.memory_allocated(device) / 1024 ** 2:.2f} MB")
        print(f"保留: {torch.cuda.memory_reserved(device) / 1024 ** 2:.2f} MB")

    # 加载预训练权重
    if args.weights and os.path.exists(args.weights):
        try:
            state_dict = torch.load(args.weights, map_location=device)
            model.load_state_dict(state_dict)
            print(f"\n成功加载权重: {args.weights}")
        except Exception as e:
            print(f"\n加载权重失败: {str(e)}")

    # 冻结层
    if args.freeze_layers:
        print("\n冻结层配置:")
        for name, param in model.named_parameters():
            if 'head' not in name:  # 假设最后一层名为'head'
                param.requires_grad = False
                print(f"冻结: {name}")

    criterion = CrossEntropyLoss().to(device)

    # 优化器
    optimizer = optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=args.lr,
        weight_decay=0.05,
        betas=(0.9, 0.999)
    )

    # 学习率调度器
    scheduler = lr_scheduler.OneCycleLR(
        optimizer,
        max_lr=args.lr * 1.5,
        steps_per_epoch=len(train_loader),
        epochs=args.epochs,
        pct_start=0.3,
        anneal_strategy='cos',
        final_div_factor=1000
    )

    # 训练循环
    logger = ProgressLogger(args.epochs)
    best_acc = 0.0

    # 添加调试信息
    print("\n===== 开始训练 =====")
    print(f"训练集大小: {len(train_dataset)}")
    print(f"验证集大小: {len(val_dataset)}")
    print(f"批次大小: {args.batch_size}")
    print(f"初始学习率: {args.lr}")

    for epoch in range(args.epochs):
        # 训练阶段
        model.train()
        train_loss, train_correct, train_total = 0, 0, 0
        start_time = time.time()

        # 使用tqdm显示进度条
        pbar = tqdm(enumerate(train_loader), total=len(train_loader),
                    desc=f'Epoch {epoch + 1}/{args.epochs}', leave=False)

        for batch_idx, (images, labels) in pbar:
            images, labels = images.to(device), labels.to(device)

            # 前向传播
            outputs = model(images)
            loss = criterion(outputs, labels)

            # 反向传播
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # 统计信息
            train_loss += loss.item()
            _, predicted = outputs.max(1)
            train_total += labels.size(0)
            train_correct += predicted.eq(labels).sum().item()

            # 更新进度条
            pbar.set_postfix({
                'loss': f'{train_loss / (batch_idx + 1):.3f}',
                'acc': f'{train_correct / train_total:.3f}'
            })

        # 计算训练指标
        train_loss /= len(train_loader)
        train_acc = train_correct / train_total

        # 验证阶段
        val_loss, val_acc = validate(model, val_loader, device, criterion)

        # 保存最佳模型
        if val_acc > best_acc:
            best_acc = val_acc
            torch.save(model.state_dict(), "weights/best_model.pth")

        # 打印epoch结果 - 修改为图片中的格式
        print(f"[train epoch {epoch + 1}] loss: {train_loss:.3f}, acc: {train_acc:.3f}: 100%")
        print(f"[valid epoch {epoch + 1}] loss: {val_loss:.3f}, acc: {val_acc:.3f}: 100%")

        # 打印时间信息
        epoch_time = time.time() - start_time
        batches_per_sec = len(train_loader) / epoch_time
        print(
            f"{len(train_loader)}/{len(train_loader)} [{epoch_time // 60:.0f}m{epoch_time % 60:.0f}s<0m, {batches_per_sec:.2f}it/s]")

    print("\n进程已结束，退出代码为 0")

def validate(model, val_loader, device, criterion):
    model.eval()
    val_loss, correct, total = 0, 0, 0
    with torch.no_grad():
        for images, labels in val_loader:
            images = images.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)
            outputs = model(images)
            val_loss += criterion(outputs, labels).item()
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()
    return val_loss / len(val_loader), correct / total


def parse_opt():
    parser = argparse.ArgumentParser(description='PyTorch EfficientNetV2 Training')
    parser.add_argument('--num-classes', type=int, default=5,
                        help='number of classes (will auto detect if not specified)')
    parser.add_argument('--epochs', type=int, default=50,
                        help='number of total epochs to run')
    parser.add_argument('--batch-size', type=int, default=32,
                        help='total batch size for all GPUs')
    parser.add_argument('--lr', type=float, default=1e-3,
                        help='initial learning rate')
    parser.add_argument('--lrf', type=float, default=0.01,
                        help='final learning rate (OneCycleLR)')
    parser.add_argument('--data-path', type=str,
                        default=r'D:\Test11_efficientnetV2 _first\train',
                        help='dataset path')
    parser.add_argument('--weights', type=str, default='',
                        help='initial weights path')
    parser.add_argument('--freeze-layers', action='store_true',
                        help='freeze all layers except head')
    parser.add_argument('--device', type=str, default='0',
                        help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
    return parser.parse_args()


if __name__ == '__main__':
    opt = parse_opt()

    # 处理路径中的空格问题
    opt.data_path = opt.data_path.strip()

    # 打印配置信息
    print("\n===== 配置信息 =====")
    print(f"数据集路径: {opt.data_path}")
    print(f"设备: {'GPU:' + str(opt.device) if opt.device != 'cpu' else 'CPU'}")
    print(f"批次大小: {opt.batch_size}")
    print(f"初始学习率: {opt.lr}")
    print(f"训练周期: {opt.epochs}")

    main(opt)