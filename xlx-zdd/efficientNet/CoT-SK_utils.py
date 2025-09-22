import os
import sys
import json
import pickle
import random
from tqdm import tqdm
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from torch.cuda.amp import GradScaler, autocast

def read_split_data(root: str = r"D:\car-data", val_rate: float = 0.2):
    random.seed(0)
    print(f"正在读取数据集，根目录: {root}")
    print(f"目录是否存在: {os.path.exists(root)}")

    if not os.path.exists(root):
        print(f"错误: 目录 {root} 不存在!")
        raise FileNotFoundError(f"dataset root: {root} does not exist.")

    # 支持的图片格式（扩展了更多常见格式）
    supported = [".jpg", ".JPG", ".jpeg", ".JPEG", ".png", ".PNG", ".bmp", ".BMP"]

    # 自动检测是否为二级目录结构
    is_two_level = any(
        os.path.isdir(os.path.join(root, d, subd))
        for d in os.listdir(root)
        if os.path.isdir(os.path.join(root, d))
        for subd in os.listdir(os.path.join(root, d))
    )

    train_images_path = []
    train_images_label = []
    val_images_path = []
    val_images_label = []
    every_class_num = []

    if is_two_level:
        print("检测到二级目录结构（root/split/class/images）")

        # 首先收集所有类别（跨train/val目录）
        classes = set()
        for split_dir in os.listdir(root):
            split_path = os.path.join(root, split_dir)
            if not os.path.isdir(split_path):
                continue

            for class_name in os.listdir(split_path):
                class_path = os.path.join(split_path, class_name)
                if os.path.isdir(class_path):
                    classes.add(class_name)

        # 创建类别索引
        classes = sorted(list(classes))
        class_indices = {cls: i for i, cls in enumerate(classes)}
        print(f"找到的类别: {classes}")

        # 收集所有图片路径并按类别分组
        class_images = {cls: [] for cls in classes}
        for split_dir in os.listdir(root):
            split_path = os.path.join(root, split_dir)
            if not os.path.isdir(split_path):
                continue

            for class_name in os.listdir(split_path):
                class_path = os.path.join(split_path, class_name)
                if not os.path.isdir(class_path) or class_name not in class_indices:
                    continue

                images = [os.path.join(class_path, img) for img in os.listdir(class_path)
                          if os.path.splitext(img)[-1] in supported]
                class_images[class_name].extend(images)

        # 对每个类别分别划分训练集和验证集
        for class_name, images in class_images.items():
            images.sort()
            image_class = class_indices[class_name]
            every_class_num.append(len(images))

            # 按比例随机采样验证样本
            val_path = random.sample(images, k=int(len(images) * val_rate))

            for img_path in images:
                if img_path in val_path:
                    val_images_path.append(img_path)
                    val_images_label.append(image_class)
                else:
                    train_images_path.append(img_path)
                    train_images_label.append(image_class)

    else:
        print("检测到一级目录结构（root/class/images）")
        # 原始的一级目录处理逻辑
        data = [cla for cla in os.listdir(root)
                if os.path.isdir(os.path.join(root, cla))]
        data.sort()
        class_indices = {k: v for v, k in enumerate(data)}
        print(f"找到的类别: {data}")

        for cla in data:
            cla_path = os.path.join(root, cla)
            images = [os.path.join(root, cla, i) for i in os.listdir(cla_path)
                      if os.path.splitext(i)[-1] in supported]
            images.sort()
            image_class = class_indices[cla]
            every_class_num.append(len(images))
            val_path = random.sample(images, k=int(len(images) * val_rate))

            for img_path in images:
                if img_path in val_path:
                    val_images_path.append(img_path)
                    val_images_label.append(image_class)
                else:
                    train_images_path.append(img_path)
                    train_images_label.append(image_class)

    # 保存类别索引
    json_str = json.dumps({v: k for k, v in class_indices.items()}, indent=4)
    with open('class_indices.json', 'w') as json_file:
        json_file.write(json_str)

    print("共找到 {} 个类别".format(len(class_indices)))
    print("数据集总图片数: {}".format(sum(every_class_num)))
    print("训练集图片数: {}".format(len(train_images_path)))
    print("验证集图片数: {}".format(len(val_images_path)))

    if not train_images_path:
        raise ValueError("训练集图片数量为0，请检查数据集路径和结构！")
    if not val_images_path:
        raise ValueError("验证集图片数量为0，请检查数据集路径和结构！")

    # 绘制类别分布图
    plot_image = False
    if plot_image:
        plt.figure(figsize=(10, 6))
        plt.bar(range(len(class_indices)), every_class_num, align='center')
        plt.xticks(range(len(class_indices)), list(class_indices.keys()), rotation=45)
        for i, v in enumerate(every_class_num):
            plt.text(x=i, y=v + 5, s=str(v), ha='center')
        plt.xlabel('Image Class')
        plt.ylabel('Number of Images')
        plt.title('Class Distribution')
        plt.tight_layout()
        plt.show()

    return train_images_path, train_images_label, val_images_path, val_images_label


def plot_data_loader_image(data_loader):
    """可视化数据加载器中的图像"""
    batch_size = data_loader.batch_size
    plot_num = min(batch_size, 4)

    json_path = './class_indices.json'
    assert os.path.exists(json_path), json_path + " does not exist."
    json_file = open(json_path, 'r')
    class_indices = json.load(json_file)

    for data in data_loader:
        images, labels = data
        plt.figure(figsize=(12, 6))
        for i in range(plot_num):
            # [C, H, W] -> [H, W, C]
            img = images[i].numpy().transpose(1, 2, 0)
            # 反Normalize操作
            img = (img * [0.229, 0.224, 0.225] + [0.485, 0.456, 0.406]) * 255
            label = labels[i].item()
            plt.subplot(1, plot_num, i + 1)
            plt.xlabel(class_indices[str(label)])
            plt.xticks([])
            plt.yticks([])
            plt.imshow(img.astype('uint8'))
        plt.tight_layout()
        plt.show()
        break  # 只显示第一个batch


def train_one_epoch(model, optimizer, data_loader, device, epoch,
                    use_amp=False, clip_grad=None, print_freq=10):
    """支持SKNet和CoTAttention的训练epoch"""
    model.train()
    loss_function = nn.CrossEntropyLoss()
    scaler = GradScaler(enabled=use_amp)

    total_loss = 0
    total_samples = 0
    pbar = tqdm(enumerate(data_loader), total=len(data_loader),
                desc=f"Epoch {epoch}", file=sys.stdout)

    for step, (images, targets) in pbar:
        images, targets = images.to(device), targets.to(device)

        # 混合精度训练
        with autocast(enabled=use_amp):
            outputs = model(images)
            loss = loss_function(outputs, targets)

        # 反向传播
        scaler.scale(loss).backward()

        # 梯度裁剪（特别适用于注意力模型）
        if clip_grad is not None:
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), clip_grad)

        scaler.step(optimizer)
        scaler.update()
        optimizer.zero_grad()

        # 更新统计信息
        batch_size = images.size(0)
        total_loss += loss.item() * batch_size
        total_samples += batch_size

        if step % print_freq == 0:
            pbar.set_postfix({
                "loss": f"{total_loss / total_samples:.4f}",
                "lr": f"{optimizer.param_groups[0]['lr']:.6f}"
            })

    return total_loss / total_samples


@torch.no_grad()
def evaluate(model, data_loader, device, return_attention=False):
    """增强的验证函数，支持SKNet和CoTAttention"""
    model.eval()
    correct = 0
    total = 0
    attention_maps = []

    pbar = tqdm(data_loader, desc="Validating", file=sys.stdout)
    for images, targets in pbar:
        images, targets = images.to(device), targets.to(device)

        # 获取模型输出（可能包含注意力图）
        outputs = model(images)
        if isinstance(outputs, tuple):  # 如果模型返回(output, attention)
            preds, attn = outputs
            if return_attention:
                attention_maps.append(attn.cpu())
        else:
            preds = outputs

        # 计算准确率
        _, predicted = torch.max(preds.data, 1)
        total += targets.size(0)
        correct += (predicted == targets).sum().item()

        pbar.set_postfix({"acc": f"{100 * correct / total:.2f}%"})

    acc = correct / total
    return (acc, attention_maps) if return_attention else acc


# -------------------- 模型工具 --------------------
def create_optimizer(model, lr=1e-3, weight_decay=1e-4, model_type="efficient"):
    """创建优化器，针对不同模型类型进行优化"""
    if model_type.startswith("sknet"):
        # SKNet需要不同的学习率设置
        params = [
            {"params": [p for n, p in model.named_parameters() if "conv2" not in n], "lr": lr},
            {"params": [p for n, p in model.named_parameters() if "conv2" in n], "lr": lr * 0.1}
        ]
        return torch.optim.AdamW(params, weight_decay=weight_decay)
    else:
        return torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)


def save_checkpoint(state, filename="checkpoint.pth"):
    """保存模型检查点"""
    torch.save(state, filename)


def load_checkpoint(model, checkpoint_path, device):
    """加载模型检查点"""
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint["state_dict"])
    return model


# -------------------- 注意力可视化 --------------------
def visualize_attention(model, data_loader, device, num_images=4):
    """可视化SKNet或CoTAttention的注意力图"""
    model.eval()
    images, _ = next(iter(data_loader))
    images = images[:num_images].to(device)

    # 获取注意力图
    with torch.no_grad():
        if hasattr(model, "get_attention_maps"):
            _, attention_maps = model(images, return_attention=True)
        else:
            outputs = model(images)
            attention_maps = None

    # 可视化
    plt.figure(figsize=(15, 5 * num_images))
    for i in range(min(num_images, images.size(0))):
        # 原始图像
        img = images[i].cpu().numpy().transpose(1, 2, 0)
        img = (img * [0.229, 0.224, 0.225] + [0.485, 0.456, 0.406]) * 255

        plt.subplot(num_images, 2, 2 * i + 1)
        plt.imshow(img.astype("uint8"))
        plt.title("Original Image")
        plt.axis("off")

        # 注意力图
        if attention_maps is not None:
            attn = attention_maps[i].mean(dim=0).cpu().numpy()
            plt.subplot(num_images, 2, 2 * i + 2)
            plt.imshow(attn, cmap="hot")
            plt.title("Attention Map")
            plt.axis("off")

    plt.tight_layout()
    plt.show()


# -------------------- 辅助函数 --------------------
def write_pickle(list_info: list, file_name: str):
    """将列表信息写入pickle文件"""
    with open(file_name, "wb") as f:
        pickle.dump(list_info, f)


def read_pickle(file_name: str) -> list:
    """从pickle文件读取列表信息"""
    with open(file_name, "rb") as f:
        return pickle.load(f)