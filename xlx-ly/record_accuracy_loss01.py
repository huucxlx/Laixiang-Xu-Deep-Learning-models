import os
import json
import torch
import torch.nn as nn
from torchvision import transforms, datasets
from tqdm import tqdm
#from model__both_enhanced import DualAttentionAlexNet
# model_hybridcs import LicensePlateAlexNetHybridCS
#from model_cood_attention import LicensePlateAlexNet
#from model_optimized import LicensePlateAlexNet
#from model import AlexNet
from model__both_enhanced02 import DualAttentionAlexNet


def evaluate_model(model, data_loader, criterion, device):
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0
    with torch.no_grad():
        for images, labels in tqdm(data_loader, desc="Evaluating"):
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)
            running_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    loss = running_loss / len(data_loader)
    accuracy = correct / total
    return loss, accuracy

def main():
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"Using {device} device for evaluation")

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

    # 数据集路径配置
    data_root = r"D:\deep_leaning\Alexnet\card-data (2)\card-data"
    train_path = os.path.join(data_root, "train")
    val_path = os.path.join(data_root, "val")

    print(f"\n使用数据集路径: {data_root}")
    print(f"训练集路径: {train_path}")
    print(f"验证集路径: {val_path}")

    # 检查路径是否存在
    if not os.path.exists(train_path):
        print(f"\n错误: 训练集路径不存在: {train_path}")
        return

    if not os.path.exists(val_path):
        print(f"\n错误: 验证集路径不存在: {val_path}")
        return

    # 加载数据集
    train_dataset = datasets.ImageFolder(root=train_path, transform=data_transform["train"])
    val_dataset = datasets.ImageFolder(root=val_path, transform=data_transform["val"])

    # 数据加载器
    batch_size = 32
    nw = min([os.cpu_count(), batch_size if batch_size > 1 else 0, 8])
    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=batch_size, shuffle=False, num_workers=nw)
    val_loader = torch.utils.data.DataLoader(
        val_dataset, batch_size=batch_size, shuffle=False, num_workers=nw)

    # 加载类别映射
    with open('class_indices.json', 'r') as f:
        cla_dict = json.load(f)

    # 初始化模型
    model = DualAttentionAlexNet(num_classes=len(cla_dict)).to(device)

    # 加载训练好的权重
    weights_path = r'D:\deep_leaning\Alexnet\checkpoints\best_moble02.pth'
    if not os.path.exists(weights_path):
        print(f"\n错误: 权重文件不存在: {weights_path}")
        return

    # 加载所有保存的检查点
    try:
        checkpoint = torch.load(weights_path, map_location=device, weights_only=True)
        if isinstance(checkpoint, dict) and 'state_dict' in checkpoint:
            model.load_state_dict(checkpoint['state_dict'])
            print(f"加载检查点权重，epoch: {checkpoint.get('epoch', '未知')}")
        else:
            model.load_state_dict(checkpoint)
            print("加载模型权重")
    except Exception as e:
        print(f"加载权重时出错: {str(e)}")
        return

    criterion = nn.CrossEntropyLoss()

    # 创建结果目录 - 修复了路径不一致的问题
    results_dir = 'training_results07'
    os.makedirs(results_dir, exist_ok=True)

    # 评估配置
    epochs = 100
    results = []

    print("\n开始评估每一轮的表现...")
    for epoch in range(1, epochs + 1):
        # 评估训练集
        train_loss, train_accuracy = evaluate_model(model, train_loader, criterion, device)

        # 评估验证集
        val_loss, val_accuracy = evaluate_model(model, val_loader, criterion, device)

        # 记录结果
        results.append({
            'epoch': epoch,
            'train_loss': train_loss,
            'train_accuracy': train_accuracy,
            'val_loss': val_loss,
            'val_accuracy': val_accuracy
        })

        # 打印当前轮次结果
        print(f"Epoch {epoch:3d}/{epochs} | "
              f"Train Loss: {train_loss:.4f} | Train Acc: {train_accuracy:.4f} | "
              f"Val Loss: {val_loss:.4f} | Val Acc: {val_accuracy:.4f}")

        # 每10轮保存一次中间结果
        if epoch % 10 == 0 or epoch == epochs:
            result_file = os.path.join(results_dir, f'epoch_{epoch}_results.txt')
            with open(result_file, 'w') as f:
                f.write("Epoch,Train Loss,Train Accuracy,Val Loss,Val Accuracy\n")
                for res in results:
                    f.write(f"{res['epoch']},{res['train_loss']:.6f},{res['train_accuracy']:.6f},"
                            f"{res['val_loss']:.6f},{res['val_accuracy']:.6f}\n")
            print(f"已保存当前结果到 {result_file}")

    # 保存最终完整结果
    final_result_file = os.path.join(results_dir, 'final_training_results.csv')
    with open(final_result_file, 'w') as f:
        f.write("Epoch,Train Loss,Train Accuracy,Val Loss,Val Accuracy\n")
        for res in results:
            f.write(f"{res['epoch']},{res['train_loss']:.6f},{res['train_accuracy']:.6f},"
                    f"{res['val_loss']:.6f},{res['val_accuracy']:.6f}\n")

    print("\n评估完成!")
    print(f"最终结果已保存到 {final_result_file}")

if __name__ == '__main__':
    main()