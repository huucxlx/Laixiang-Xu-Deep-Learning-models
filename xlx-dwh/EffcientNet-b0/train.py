import os
import argparse

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
from torchvision import transforms

from model import efficientnet_b0 as create_model
from my_dataset import MyDataSet
from utils import train_one_epoch, evaluate  # 移除不再使用的 read_split_data
# 在训练代码开头添加
import torch.autograd
torch.autograd.set_detect_anomaly(True)



# 检查设备
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
# 在训练代码开头添加异常检测
torch.autograd.set_detect_anomaly(True)

# 运行后会显示更详细的错误定位
def get_image_paths_and_labels(data_dir):
    """从数据集目录生成图像路径和标签（类别为子目录名）"""
    classes = [cls for cls in os.listdir(data_dir) if os.path.isdir(os.path.join(data_dir, cls))]
    classes.sort()
    class_to_idx = {cls: idx for idx, cls in enumerate(classes)}

    image_paths = []
    labels = []
    for cls in classes:
        cls_dir = os.path.join(data_dir, cls)
        for img_name in os.listdir(cls_dir):
            if img_name.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp', '.gif')):
                img_path = os.path.join(cls_dir, img_name)
                image_paths.append(img_path)
                labels.append(class_to_idx[cls])
    return image_paths, labels


def prepare_data(args):
    # 直接读取用户指定的训练集和验证集路径
    train_image_paths, train_labels = get_image_paths_and_labels(args.train_data_path)
    val_image_paths, val_labels = get_image_paths_and_labels(args.val_data_path)

    img_size = 224  # EfficientNet-B0 输入尺寸
    num_model = "B0"

    data_transform = {
        "train": transforms.Compose([
            # 强制Resize到固定尺寸（224x224），解决尺寸不一致问题
            transforms.Resize((img_size, img_size)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),
        "val": transforms.Compose([
            transforms.Resize((img_size, img_size)),
            transforms.CenterCrop(img_size),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
    }

    # 实例化数据集（传入路径列表和标签列表）
    train_dataset = MyDataSet(
        images_path=train_image_paths,
        images_class=train_labels,
        transform=data_transform["train"]
    )

    val_dataset = MyDataSet(
        images_path=val_image_paths,
        images_class=val_labels,
        transform=data_transform["val"]
    )

    batch_size = args.batch_size
    nw = 4
    print('Using {} dataloader workers every process'.format(nw))

    # 移除自定义 collate_fn（若 MyDataSet 中没有特殊需求，使用默认即可）
    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        pin_memory=True,
        num_workers=nw
    )

    val_loader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        pin_memory=True,
        num_workers=nw
    )

    return train_loader, val_loader


def train_model(model, optimizer, train_loader, val_loader, device, args, tb_writer):
    best_acc = 0.0
    best_epoch = 0

    for epoch in range(args.epochs):
        train_loss, train_acc = train_one_epoch(
            model=model,
            optimizer=optimizer,
            data_loader=train_loader,
            device=device,
            epoch=epoch
        )

        val_loss, val_acc = evaluate(
            model=model,
            data_loader=val_loader,
            device=device
        )

        if val_acc > best_acc:
            best_acc = val_acc
            best_epoch = epoch
            torch.save(model.state_dict(), "./weights/best.pth")
            print(f"New best model saved at epoch {epoch} with val_acc: {val_acc:.3f}")

        print(
            f"[epoch {epoch}] train_loss: {train_loss:.3f}, train_acc: {train_acc:.3f}, val_loss: {val_loss:.3f}, val_acc: {val_acc:.3f}")

        tags = ["train_loss", "train_accuracy", "val_loss", "val_accuracy", "learning_rate"]
        tb_writer.add_scalar(tags[0], train_loss, epoch)
        tb_writer.add_scalar(tags[1], train_acc, epoch)
        tb_writer.add_scalar(tags[2], val_loss, epoch)
        tb_writer.add_scalar(tags[3], val_acc, epoch)
        tb_writer.add_scalar(tags[4], optimizer.param_groups[0]["lr"], epoch)

    print(f"\nTraining completed. Best accuracy: {best_acc:.3f} at epoch {best_epoch}")
    tb_writer.close()


def main(args):
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")

    print(args)
    print('Start Tensorboard with "tensorboard --logdir=runs", view at http://localhost:6006/')
    tb_writer = SummaryWriter()
    if not os.path.exists("./weights"):
        os.makedirs("./weights")

    train_loader, val_loader = prepare_data(args)

    model = create_model(num_classes=args.num_classes).to(device)
    if args.weights != "":
        if os.path.exists(args.weights):
            weights_dict = torch.load(args.weights, map_location=device)
            load_weights_dict = {k: v for k, v in weights_dict.items() if model.state_dict()[k].numel() == v.numel()}
            print(model.load_state_dict(load_weights_dict, strict=False))
        else:
            raise FileNotFoundError(f"not found weights file: {args.weights}")

    if args.freeze_layers:
        for name, para in model.named_parameters():
            if ("features.top" not in name) and ("classifier" not in name):
                para.requires_grad_(False)
            else:
                print(f"training {name}")

    pg = [p for p in model.parameters() if p.requires_grad]
    optimizer = optim.SGD(pg, lr=args.lr, momentum=0.9, weight_decay=1E-5)

    train_model(model, optimizer, train_loader, val_loader, device, args, tb_writer)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--num_classes', type=int, default=4)
    parser.add_argument('--epochs', type=int, default=30)
    parser.add_argument('--batch-size', type=int, default=32)  # 建议根据显存调整，如64/32
    parser.add_argument('--lr', type=float, default=0.0001)
    parser.add_argument('--train_data_path', type=str,
                        default=r'D:\pycharm\pycharmprojects\learn-pytorch\EffcientNet-b0\data\soybean\train',
                        help='Path to the training dataset (class subdirectories inside)')
    parser.add_argument('--val_data_path', type=str,
                        default=r'D:\pycharm\pycharmprojects\learn-pytorch\EffcientNet-b0\data\soybean\val',
                        help='Path to the validation dataset (class subdirectories inside)')
    parser.add_argument('--weights', type=str, default='', help='initial weights path')
    parser.add_argument('--freeze-layers', type=bool, default=False)
    parser.add_argument('--device', default='cuda:0', help='device id (i.e. 0 or 0,1 or cpu)')

    opt = parser.parse_args()
    main(opt)