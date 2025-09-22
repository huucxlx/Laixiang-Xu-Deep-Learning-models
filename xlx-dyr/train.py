import torch
import torch.nn as nn
from torchvision import transforms, datasets
from torch.utils.data import DataLoader, random_split
import torch.optim as optim
from model import mobilenetv3_small
import os
import json
import csv
from datetime import datetime
import pandas as pd
from tqdm import tqdm


def main():
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # 数据预处理
    data_transform = {
        "train": transforms.Compose([
            transforms.RandomResizedCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),
        "val": transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),
        "test": transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
    }

    # 数据集路径
    image_path = r"D:\myself\mobilNetV3\car_datas（he）"

    # 加载完整数据集
    full_dataset = datasets.ImageFolder(root=image_path, transform=data_transform["train"])

    # 获取数据集大小
    dataset_size = len(full_dataset)
    print(f"Total dataset size: {dataset_size}")

    # 划分数据集
    train_size = int(0.6 * dataset_size)
    val_size = int(0.2 * dataset_size)
    test_size = dataset_size - train_size - val_size

    train_dataset, val_dataset, test_dataset = random_split(
        full_dataset, [train_size, val_size, test_size]
    )
    print(f"Train size: {len(train_dataset)}, Val size: {len(val_dataset)}, Test size: {len(test_dataset)}")

    # 获取分类的名称
    flower_list = full_dataset.class_to_idx
    cla_dict = dict((val, key) for key, val in flower_list.items())
    print("Class mapping:", cla_dict)

    # 创建数据加载器 (Windows下建议num_workers=0)
    batch_size = 16
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=0)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=0)

    # 定义模型
    net = mobilenetv3_small(num_classes=len(cla_dict)).to(device)
    print("Model created with DCN layers")

    # 定义损失函数和优化器
    loss_function = nn.CrossEntropyLoss()
    optimizer = optim.Adam(net.parameters(), lr=0.0005, weight_decay=1e-4)

    # 设置存储路径
    save_dir = r"D:\myself\mobilNetV3\results"
    os.makedirs(save_dir, exist_ok=True)
    model_save_path = os.path.join(save_dir, "best_model.pth")
    csv_save_path = os.path.join(save_dir, "training_log.csv")
    log_file = os.path.join(save_dir, "training_log.txt")

    # 初始化日志文件
    def log_message(message):
        print(message)
        with open(log_file, 'a') as f:
            f.write(message + '\n')

    with open(csv_save_path, mode='w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['epoch', 'train_loss', 'val_loss', 'train_acc', 'val_acc', 'timestamp'])

    best_acc = 0.0

    # 训练循环
    for epoch in range(50):
        # Training phase
        net.train()
        running_loss = 0.0
        running_corrects = 0

        train_bar = tqdm(train_loader, desc=f'Train Epoch {epoch + 1}', leave=False)
        for images, labels in train_bar:
            images, labels = images.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = net(images)
            loss = loss_function(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * images.size(0)
            _, preds = torch.max(outputs, 1)
            running_corrects += torch.sum(preds == labels.data)

            train_bar.set_postfix(loss=loss.item())

        train_loss = running_loss / len(train_dataset)
        train_acc = running_corrects.double() / len(train_dataset)

        # Validation phase
        net.eval()
        val_loss = 0.0
        val_corrects = 0

        val_bar = tqdm(val_loader, desc=f'Val Epoch {epoch + 1}', leave=False)
        with torch.no_grad():
            for images, labels in val_bar:
                images, labels = images.to(device), labels.to(device)
                outputs = net(images)
                loss = loss_function(outputs, labels)

                val_loss += loss.item() * images.size(0)
                _, preds = torch.max(outputs, 1)
                val_corrects += torch.sum(preds == labels.data)

                val_bar.set_postfix(loss=loss.item())

        val_loss = val_loss / len(val_dataset)
        val_acc = val_corrects.double() / len(val_dataset)

        # 记录当前时间戳
        current_time = datetime.now().strftime('%Y-%m-%d %H:%M:%S')

        # 保存到CSV
        with open(csv_save_path, mode='a', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([epoch + 1, train_loss, val_loss, train_acc.item(), val_acc.item(), current_time])

        log_message(f"[epoch {epoch + 1}] train_loss: {train_loss:.4f}  val_loss: {val_loss:.4f}  "
                    f"train_acc: {train_acc:.4f}  val_acc: {val_acc:.4f}")

        # 保存最佳模型
        if val_acc > best_acc:
            best_acc = val_acc
            torch.save({
                'epoch': epoch + 1,
                'model_state_dict': net.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_acc': val_acc,
                'class_mapping': cla_dict
            }, model_save_path)
            log_message(f"New best model saved with val_acc: {val_acc:.4f}")

    # 测试模型
    log_message("\nTesting best model...")
    checkpoint = torch.load(model_save_path)
    net.load_state_dict(checkpoint['model_state_dict'])
    net.eval()

    test_corrects = 0
    test_results = []
    test_loss = 0.0

    test_bar = tqdm(test_loader, desc='Testing', leave=False)
    with torch.no_grad():
        for images, labels in test_bar:
            images, labels = images.to(device), labels.to(device)
            outputs = net(images)
            loss = loss_function(outputs, labels)

            test_loss += loss.item() * images.size(0)
            _, preds = torch.max(outputs, 1)
            test_corrects += torch.sum(preds == labels.data)

            # 保存测试结果
            for i in range(len(labels)):
                test_results.append({
                    'true_label': labels[i].item(),
                    'true_class': cla_dict[labels[i].item()],
                    'pred_label': preds[i].item(),
                    'pred_class': cla_dict[preds[i].item()],
                    'correct': int(labels[i] == preds[i])
                })

    test_loss = test_loss / len(test_dataset)
    test_acc = test_corrects.double() / len(test_dataset)
    log_message(f"\nTest Results - Loss: {test_loss:.4f}  Accuracy: {test_acc:.4f}")

    # 保存测试结果到CSV
    test_results_df = pd.DataFrame(test_results)
    test_results_df.to_csv(os.path.join(save_dir, 'test_results.csv'), index=False)

    # 保存类别映射
    with open(os.path.join(save_dir, 'class_mapping.json'), 'w') as f:
        json.dump(cla_dict, f)

    log_message("\nTraining and evaluation completed successfully!")
    log_message(f"Best validation accuracy: {best_acc:.4f}")
    log_message(f"Test accuracy: {test_acc:.4f}")


if __name__ == '__main__':
    main()