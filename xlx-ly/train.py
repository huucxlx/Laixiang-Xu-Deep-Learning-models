import os
import sys
import json
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim  # 修复问题1：添加optim导入
from torchvision import transforms, datasets
from tqdm import tqdm
import matplotlib.pyplot as plt
from model import AlexNet
#from model_optimized import LicensePlateAlexNet  # 假设你的简化版模型类名是 LicensePlateAlexNet
#from model_cood_attention import LicensePlateAlexNet
#from model_hybridcs import LicensePlateAlexNetHybridCS
#from model__both_enhanced import DualAttentionAlexNet
from model_LiteCoord_attention import LicensePlateAlexNet
#from model__both_enhanced02 import DualAttentionAlexNet



def main():
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

    # 数据集路径配置 ==============================================
    # 修复问题5：拼写错误 Alexnet -> AlexNet
    data_root = r"D:\deep_leaning\Alexnet\data"  # 修正后的路径

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
    # ===========================================================

    # 加载数据集
    train_dataset = datasets.ImageFolder(root=train_path, transform=data_transform["train"])
    val_dataset = datasets.ImageFolder(root=val_path, transform=data_transform["val"])

    train_num = len(train_dataset)
    val_num = len(val_dataset)

    # 类别映射
    class_dict = train_dataset.class_to_idx
    cla_dict = {v: k for k, v in class_dict.items()}
    with open('class_02indices.json', 'w') as f:
        json.dump(cla_dict, f, indent=4)

    # 数据加载器
    batch_size = 32
    nw = min([os.cpu_count(), batch_size if batch_size > 1 else 0, 8])
    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True, num_workers=nw)
    val_loader = torch.utils.data.DataLoader(
        val_dataset, batch_size=batch_size, shuffle=False, num_workers=nw)

    # 初始化模型
    model = AlexNet(num_classes=len(cla_dict)).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-4)

    # 训练参数
    epochs = 100
    save_path = r'AlexNet_Plant.pth'
    best_acc = 0.0

    # 记录训练过程
    train_loss_list = []
    train_acc_list = []
    val_loss_list = []
    val_acc_list = []

    print("\n开始训练...")
    for epoch in range(epochs):
        # 训练阶段
        model.train()
        running_loss = 0.0
        correct_train = 0
        total_train = 0

        train_bar = tqdm(train_loader, file=sys.stdout)
        for images, labels in train_bar:
            images, labels = images.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total_train += labels.size(0)
            # 修复问题2：使用torch.sum()替代.sum()
            correct_train += torch.sum(predicted == labels).item()

            train_bar.desc = f"Train Epoch [{epoch + 1}/{epochs}] Loss: {loss:.3f}"

        # 计算训练指标
        train_loss = running_loss / len(train_loader)
        train_acc = correct_train / total_train
        train_loss_list.append(train_loss)
        train_acc_list.append(train_acc)

        # 验证阶段
        model.eval()
        val_loss = 0.0
        correct_val = 0
        total_val = 0

        with torch.no_grad():
            val_bar = tqdm(val_loader, file=sys.stdout)
            for images, labels in val_bar:
                images, labels = images.to(device), labels.to(device)

                outputs = model(images)
                loss = criterion(outputs, labels)
                val_loss += loss.item()

                _, predicted = torch.max(outputs.data, 1)
                total_val += labels.size(0)
                # 修复问题3：使用torch.sum()替代.sum()
                correct_val += torch.sum(predicted == labels).item()

        # 计算验证指标
        val_loss = val_loss / len(val_loader)
        val_acc = correct_val / total_val
        val_loss_list.append(val_loss)
        val_acc_list.append(val_acc)

        # 打印epoch结果
        print(f"Epoch [{epoch + 1}/{epochs}] "
              f"Train Loss: {train_loss:.4f} Acc: {train_acc:.4f} | "
              f"Val Loss: {val_loss:.4f} Acc: {val_acc:.4f}")

        # 保存最佳模型
        if val_acc > best_acc:
            best_acc = val_acc
            torch.save(model.state_dict(), save_path)

    # 训练结果
    print("\n训练完成!")
    print(f"最佳验证准确率: {best_acc:.4f}")
    print(f"平均训练准确率: {np.mean(train_acc_list):.4f}")
    print(f"平均验证准确率: {np.mean(val_acc_list):.4f}")

    # 绘制训练曲线
    plt.figure(figsize=(12, 5))

    # Loss曲线
    plt.subplot(1, 2, 1)
    plt.plot(train_loss_list, label='Train Loss')
    plt.plot(val_loss_list, label='Val Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training and Validation Loss')
    plt.legend()

    # Accuracy曲线
    plt.subplot(1, 2, 2)
    plt.plot(train_acc_list, label='Train Acc')
    plt.plot(val_acc_list, label='Val Acc')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.title('Training and Validation Accuracy')
    plt.legend()

    plt.tight_layout()
    plt.savefig('training_metrics011.png')

    # 修复问题4：确保文件末尾有空行
    print("\n训练曲线已保存为 'training_metrics11.png'")


if __name__ == '__main__':
    main()