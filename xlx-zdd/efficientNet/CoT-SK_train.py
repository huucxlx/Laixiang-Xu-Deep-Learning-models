import os
import math
import argparse
import torch
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
from torchvision import transforms
import torch.optim.lr_scheduler as lr_scheduler

import torch.nn as nn
from models import efficient0
from models import sknet50, sknet101
from my_dataset import MyDataSet
from utils import read_split_data, train_one_epoch, evaluate


def create_model(model_name='efficient0', num_classes=1000, pretrained=False):
    """模型工厂函数，支持多种模型"""
    model_dict = {
        'efficient0': efficient0,
        'sknet50': sknet50,
        'sknet101': sknet101
    }

    if model_name not in model_dict:
        raise ValueError(f"Unsupported model: {model_name}")

    model = model_dict[model_name](num_classes=num_classes)

    # SKNet特殊初始化
    if model_name.startswith('sknet'):
        for m in model.modules():
            if isinstance(m, (nn.Conv2d, nn.Linear)):
                nn.init.kaiming_normal_(m.weight, mode='fan_out')
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    return model


def convert_pretrained_keys(pretrained_dict, model_dict, model_type='efficient'):
    """增强的权重转换函数，支持多种模型"""
    converted_dict = {}
    key_mappings = {}

    # 不同模型的键名映射规则
    if model_type.startswith('sknet'):
        key_mappings = {
            'features.stem.0.weight': 'conv1.weight',
            'features.stem.1.weight': 'bn1.weight',
            # 添加更多SKNet特有的映射...
        }
    else:  # efficientnet
        key_mappings = {
            '_conv_stem.weight': 'features.stem_conv.0.weight',
            '_bn0.weight': 'features.stem_conv.1.weight',
            # 添加更多EfficientNet的映射...
        }

    # 尝试转换每个键
    for pretrained_key, pretrained_value in pretrained_dict.items():
        # 优先检查映射表
        if pretrained_key in key_mappings:
            model_key = key_mappings[pretrained_key]
            if model_key in model_dict:
                converted_dict[model_key] = pretrained_value
                continue

        # 尝试自动转换
        model_key = pretrained_key
        if model_type.startswith('sknet'):
            model_key = model_key.replace('features.', '')
        else:
            model_key = model_key.replace('_blocks.', 'features.')

        if model_key in model_dict and pretrained_value.shape == model_dict[model_key].shape:
            converted_dict[model_key] = pretrained_value

    return converted_dict


def create_optimizer(model, lr=0.01, model_type='efficient'):
    """创建优化器，针对不同模型优化"""
    if model_type.startswith('sknet'):
        # SKNet需要不同的学习率设置
        params = [
            {"params": [p for n, p in model.named_parameters() if "attention" not in n], "lr": lr},
            {"params": [p for n, p in model.named_parameters() if "attention" in n], "lr": lr * 0.1}
        ]
        return optim.SGD(params, momentum=0.9, weight_decay=1e-4)
    else:
        return optim.SGD(model.parameters(), lr=lr, momentum=0.9, weight_decay=1e-4)


def main(args):
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")

    # 初始化TensorBoard
    print('Start Tensorboard with "tensorboard --logdir=runs", view at http://localhost:6006/')
    tb_writer = SummaryWriter()
    if os.path.exists("./weights") is False:
        os.makedirs("./weights")

    # 数据加载
    train_images_path, train_images_label, val_images_path, val_images_label = read_split_data(args.data_path)

    # 图像变换
    img_size = 224  # 统一使用224x224输入
    data_transform = {
        "train": transforms.Compose([
            transforms.RandomResizedCrop(img_size),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),
        "val": transforms.Compose([
            transforms.Resize(img_size),
            transforms.CenterCrop(img_size),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
    }

    # 数据集
    train_dataset = MyDataSet(images_path=train_images_path,
                              images_class=train_images_label,
                              transform=data_transform["train"])
    val_dataset = MyDataSet(images_path=val_images_path,
                            images_class=val_images_label,
                            transform=data_transform["val"])

    # 数据加载器
    batch_size = args.batch_size
    nw = min([os.cpu_count(), batch_size if batch_size > 1 else 0, 8])
    train_loader = torch.utils.data.DataLoader(train_dataset,
                                               batch_size=batch_size,
                                               shuffle=True,
                                               pin_memory=True,
                                               num_workers=nw,
                                               collate_fn=train_dataset.collate_fn)
    val_loader = torch.utils.data.DataLoader(val_dataset,
                                             batch_size=batch_size,
                                             shuffle=False,
                                             pin_memory=True,
                                             num_workers=nw,
                                             collate_fn=val_dataset.collate_fn)

    # 初始化模型
    model = create_model(model_name=args.model, num_classes=args.num_classes).to(device)

    # 加载预训练权重
    if args.weights != "":
        if os.path.exists(args.weights):
            weights_dict = torch.load(args.weights, map_location=device)

            # 转换权重键名
            converted_weights = convert_pretrained_keys(
                weights_dict,
                model.state_dict(),
                model_type=args.model
            )

            # 加载权重
            load_info = model.load_state_dict(converted_weights, strict=False)
            print("\n权重加载结果:")
            print(f"缺失的键: {load_info.missing_keys}")
            print(f"意外的键: {load_info.unexpected_keys}")
        else:
            raise FileNotFoundError(f"not found weights file: {args.weights}")

    # 冻结层
    if args.freeze_layers:
        for name, para in model.named_parameters():
            # 不同模型的冻结策略
            if args.model.startswith('sknet'):
                if "fc" not in name:  # 只训练最后的全连接层
                    para.requires_grad_(False)
            else:  # efficientnet
                if ("features.top" not in name) and ("classifier" not in name):
                    para.requires_grad_(False)
            if para.requires_grad:
                print(f"训练层: {name}")

    # 优化器和学习率调度器
    optimizer = create_optimizer(model, lr=args.lr, model_type=args.model)
    lf = lambda x: ((1 + math.cos(x * math.pi / args.epochs)) / 2) * (1 - args.lrf) + args.lrf
    scheduler = lr_scheduler.LambdaLR(optimizer, lr_lambda=lf)

    # 训练循环
    for epoch in range(args.epochs):
        # 训练一个epoch
        mean_loss = train_one_epoch(
            model=model,
            optimizer=optimizer,
            data_loader=train_loader,
            device=device,
            epoch=epoch,
            use_amp=args.amp,  # 是否使用混合精度
            clip_grad=args.clip_grad  # 梯度裁剪
        )

        scheduler.step()

        # 验证
        acc = evaluate(
            model=model,
            data_loader=val_loader,
            device=device
        )

        # 记录日志
        print(f"[epoch {epoch}] accuracy: {acc:.3f}")
        tb_writer.add_scalar("loss", mean_loss, epoch)
        tb_writer.add_scalar("accuracy", acc, epoch)
        tb_writer.add_scalar("learning_rate", optimizer.param_groups[0]["lr"], epoch)

        # 保存模型
        if (epoch + 1) % args.save_freq == 0:
            torch.save(model.state_dict(), f"./weights/model-{epoch}.pth")

    tb_writer.close()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, default='efficient0',
                        choices=['efficient0', 'sknet50', 'sknet101'],
                        help='model architecture')
    parser.add_argument('--num-classes', type=int, default=4)
    parser.add_argument('--epochs', type=int, default=30)
    parser.add_argument('--batch-size', type=int, default=16)
    parser.add_argument('--lr', type=float, default=0.01)
    parser.add_argument('--lrf', type=float, default=0.01)
    parser.add_argument('--data-path', type=str, default='D:\\car-data/')
    parser.add_argument('--weights', type=str, default='')
    parser.add_argument('--freeze-layers', type=bool, default=False)
    parser.add_argument('--device', default='cuda:0')
    parser.add_argument('--amp', action='store_true', help='use mixed precision')
    parser.add_argument('--clip-grad', type=float, default=5.0,
                        help='gradient clipping max norm')
    parser.add_argument('--save-freq', type=int, default=5,
                        help='save frequency (epochs)')

    args = parser.parse_args()
    main(args)