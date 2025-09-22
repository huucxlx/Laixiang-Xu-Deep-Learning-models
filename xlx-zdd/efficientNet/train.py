import os
import math
import argparse

import torch
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
from torchvision import transforms
import torch.optim.lr_scheduler as lr_scheduler

from original_m import efficient0 as create_model
from my_dataset import MyDataSet
from utils import read_split_data, train_one_epoch, evaluate


def convert_pretrained_keys(pretrained_dict, model_dict):
    """
    更健壮的键名转换函数
    Args:
        pretrained_dict: 预训练权重的state_dict
        model_dict: 模型的state_dict
    Returns:
        转换后的state_dict
    """
    converted_dict = {}
    unmatched_keys = []

    # 打印调试信息
    print("\n==== 键名匹配调试信息 ====")
    print("预训练权重中的前5个键:", list(pretrained_dict.keys())[:5])
    print("模型中的前5个键:", list(model_dict.keys())[:5])

    # 定义键名映射规则
    key_mappings = {
        '_conv_stem.weight': 'features.stem_conv.0.weight',
        '_bn0.weight': 'features.stem_conv.1.weight',
        '_bn0.bias': 'features.stem_conv.1.bias',
        '_bn0.running_mean': 'features.stem_conv.1.running_mean',
        '_bn0.running_var': 'features.stem_conv.1.running_var',
        # 添加更多映射规则...
    }

    # 尝试转换每个键
    for pretrained_key in pretrained_dict.keys():
        # 先检查是否有直接映射
        if pretrained_key in key_mappings:
            model_key = key_mappings[pretrained_key]
            if model_key in model_dict:
                converted_dict[model_key] = pretrained_dict[pretrained_key]
                print(f"匹配成功: {pretrained_key} -> {model_key}")
                continue

        # 尝试自动转换（更通用的规则）
        model_key = pretrained_key
        model_key = model_key.replace('_conv_stem', 'features.stem_conv.0')
        model_key = model_key.replace('_bn0', 'features.stem_conv.1')

        if model_key in model_dict:
            converted_dict[model_key] = pretrained_dict[pretrained_key]
            print(f"自动匹配: {pretrained_key} -> {model_key}")
        else:
            unmatched_keys.append(pretrained_key)
            print(f"未匹配键: {pretrained_key} (尝试映射到 {model_key})")

    print(f"\n总键数: {len(pretrained_dict)}, 成功匹配: {len(converted_dict)}, 未匹配: {len(unmatched_keys)}")
    print("==== 调试信息结束 ====\n")

    return converted_dict


def main(args):
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")

    print(args)
    print('Start Tensorboard with "tensorboard --logdir=runs", view at http://localhost:6006/')
    tb_writer = SummaryWriter()
    if os.path.exists("./weights") is False:
        os.makedirs("./weights")

    # 使用新的read_split_data函数加载数据
    train_images_path, train_images_label, val_images_path, val_images_label = read_split_data(args.data_path)

    img_size = {"B0": 224,
                "B1": 240,
                "B2": 260,
                "B3": 300,
                "B4": 380,
                "B5": 456,
                "B6": 528,
                "B7": 600}
    num_model = "B0"

    data_transform = {
        "train": transforms.Compose([transforms.RandomResizedCrop(img_size[num_model]),
                                     transforms.RandomHorizontalFlip(),
                                     transforms.ToTensor(),
                                     transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])]),
        "val": transforms.Compose([transforms.Resize(img_size[num_model]),
                                   transforms.CenterCrop(img_size[num_model]),
                                   transforms.ToTensor(),
                                   transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])}

    # 实例化训练数据集
    train_dataset = MyDataSet(images_path=train_images_path,
                              images_class=train_images_label,
                              transform=data_transform["train"])

    # 实例化验证数据集
    val_dataset = MyDataSet(images_path=val_images_path,
                            images_class=val_images_label,
                            transform=data_transform["val"])

    batch_size = args.batch_size
    nw = min([os.cpu_count(), batch_size if batch_size > 1 else 0, 8])
    print('Using {} dataloader workers every process'.format(nw))
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
    model = create_model(num_classes=args.num_classes).to(device)

    # 加载预训练权重
    if args.weights != "":
        if os.path.exists(args.weights):
            # 加载预训练权重
            weights_dict = torch.load(args.weights, map_location=device)

            # 转换键名
            converted_weights = convert_pretrained_keys(weights_dict, model.state_dict())

            # 加载转换后的权重
            load_info = model.load_state_dict(converted_weights, strict=False)

            # 打印加载结果
            print("\n权重加载结果:")
            print(f"缺失的键: {load_info.missing_keys}")
            print(f"意外的键: {load_info.unexpected_keys}")
        else:
            raise FileNotFoundError("not found weights file: {}".format(args.weights))

    # 是否冻结权重
    if args.freeze_layers:
        for name, para in model.named_parameters():
            if ("features.top" not in name) and ("classifier" not in name):
                para.requires_grad_(False)
            else:
                print("training {}".format(name))

    pg = [p for p in model.parameters() if p.requires_grad]
    optimizer = optim.SGD(pg, lr=args.lr, momentum=0.9, weight_decay=1E-4)
    lf = lambda x: ((1 + math.cos(x * math.pi / args.epochs)) / 2) * (1 - args.lrf) + args.lrf
    scheduler = lr_scheduler.LambdaLR(optimizer, lr_lambda=lf)

    for epoch in range(args.epochs):
        # train
        mean_loss = train_one_epoch(model=model,
                                    optimizer=optimizer,
                                    data_loader=train_loader,
                                    device=device,
                                    epoch=epoch)

        scheduler.step()

        # validate
        acc = evaluate(model=model,
                       data_loader=val_loader,
                       device=device)
        print("[epoch {}] accuracy: {}".format(epoch, round(acc, 3)))
        tags = ["loss", "accuracy", "learning_rate"]
        tb_writer.add_scalar(tags[0], mean_loss, epoch)
        tb_writer.add_scalar(tags[1], acc, epoch)
        tb_writer.add_scalar(tags[2], optimizer.param_groups[0]["lr"], epoch)

        torch.save(model.state_dict(), "./weights/model-{}.pth".format(epoch))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--num_classes', type=int, default=4)
    parser.add_argument('--epochs', type=int, default=30)
    parser.add_argument('--batch-size', type=int, default=16)
    parser.add_argument('--lr', type=float, default=0.01)
    parser.add_argument('--lrf', type=float, default=0.01)
    parser.add_argument('--data-path', type=str,
                        default=r"D:\car-data/",
                        help='path to dataset directory')
    parser.add_argument('--weights', type=str, default='./efficient0.pth',
                        help='initial weights path')
    parser.add_argument('--freeze-layers', type=bool, default=False)
    parser.add_argument('--device', default='cuda:0', help='device id (i.e. 0 or 0,1 or cpu)')

    opt = parser.parse_args()
    main(opt)