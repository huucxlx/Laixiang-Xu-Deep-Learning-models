import argparse
import math
import os

import torch
import torch.optim as optim
import torch.optim.lr_scheduler as lr_scheduler
from model import efficientnetv2_s as create_model
from my_dataset import MyDataSet
from torch.utils.tensorboard import SummaryWriter
from torchvision import transforms
from utils import evaluate, read_split_data, train_one_epoch
import matplotlib.pyplot as plt
from utils import train_one_epoch, evaluate


def main(args):
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")

    # 初始化存储训练/验证指标的列表
    train_losses = []
    train_accuracies = []
    val_losses = []
    val_accuracies = []

    print(args)
    print('Start Tensorboard with "tensorboard --logdir=runs", view at http://localhost:6006/')
    tb_writer = SummaryWriter()
    if os.path.exists("./weights") is False:
        os.makedirs("./weights")

    train_images_path, train_images_label, val_images_path, val_images_label = read_split_data(
        args.data_path)

    img_size = {"s": [300, 384],  # train_size, val_size
                "m": [384, 480],
                "l": [384, 480]}
    num_model = "s"

    data_transform = {
        "train": transforms.Compose([#transforms.RandomResizedCrop(img_size[num_model][0]),
                                     #transforms.RandomHorizontalFlip(),
                                     transforms.Resize(img_size[num_model][1]),
                                          # 中心裁剪
                                     #transforms.CenterCrop(img_size[num_model][1]),
                                     transforms.ToTensor(),
                                     transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])]),

        #"val": transforms.Compose([transforms.Resize(img_size[num_model][1]),
                                   #transforms.CenterCrop(img_size[num_model][1]),
                                   #transforms.ToTensor(),
                                   #transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])])}
        "val": transforms.Compose([

            # 随机灰度化，以 0.1 的概率将图像转换为灰度图
            #transforms.RandomGrayscale(p=0.1),
            #transforms.GaussianBlur(kernel_size=3, sigma=(0.1, 2.0)),
            #transforms.GaussianBlur(kernel_size=3, sigma=(0.1, 2.0)),
            # 随机旋转，在 -10 到 10 度之间随机旋转图像
            #transforms.RandomRotation(degrees=5),
            # 调整图像大小
            transforms.Resize(img_size[num_model][1]),
            # 中心裁剪
            transforms.CenterCrop(img_size[num_model][1]),
            # 转换为张量
            transforms.ToTensor(),
            # 归一化
            transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
        ])
    }
    '''
        "val": transforms.Compose([transforms.Resize(img_size[num_model][1]),
                                 transforms.CenterCrop(img_size[num_model][1]),
                                  transforms.RandomResizedCrop(img_size[num_model][1]),  # 添加随机裁剪
                                   transforms.RandomHorizontalFlip(p=0.5),                 # 添加随机翻转
                                   transforms.ToTensor(),
                                   transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])])}
    
        # 验证集预处理应固定，避免随机性
        "val": transforms.Compose([
            transforms.Resize(img_size[num_model][1]),
            transforms.CenterCrop(img_size[num_model][1]),  # 固定中心裁剪
            transforms.ToTensor(),
            transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])

    ])}
    
    data_transform = {
        "train": transforms.Compose([
            transforms.Resize((img_size[num_model][0], img_size[num_model][0])),
            transforms.CenterCrop(img_size[num_model][0]),
            transforms.ToTensor(),
            transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
        ]),
        "val": transforms.Compose([
            transforms.Resize((img_size[num_model][1], img_size[num_model][1])),
            transforms.CenterCrop(img_size[num_model][1]),
            transforms.ToTensor(),
            transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
        ])}
        '''
    # 实例化训练数据集
    train_dataset = MyDataSet(images_path=train_images_path,
                              images_class=train_images_label,
                              transform=data_transform["train"])

    # 实例化验证数据集
    val_dataset = MyDataSet(images_path=val_images_path,
                            images_class=val_images_label,
                            transform=data_transform["val"])

    batch_size = args.batch_size
    # number of workers
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

    # 如果存在预训练权重则载入
    model = create_model(num_classes=args.num_classes).to(device)
    if args.weights != "":
        if os.path.exists(args.weights):
            weights_dict = torch.load(args.weights, map_location=device)

            # load_weights_dict = {k: v for k, v in weights_dict.items()
            #                     if model.state_dict()[k].numel() == v.numel()}

            # 修改后：跳过SE模块和分类头
            load_weights_dict = {
                k: v for k, v in weights_dict.items()
                if ("se" not in k) and ("head" not in k) and (model.state_dict()[k].numel() == v.numel())
            }

            print(model.load_state_dict(load_weights_dict, strict=False))
        else:
            raise FileNotFoundError(
                "not found weights file: {}".format(args.weights))

    # 是否冻结权重
    if args.freeze_layers:
        for name, para in model.named_parameters():
            # 除head外，其他权重全部冻结
            if "head" not in name:
                para.requires_grad_(False)
            else:
                print("training {}".format(name))

    pg = [p for p in model.parameters() if p.requires_grad]
    optimizer = optim.SGD(pg, lr=args.lr, momentum=0.9, weight_decay=1E-4)

    # Scheduler https://arxiv.org/pdf/1812.01187.pdf
    # 余弦退火
    def lf(x):
        return ((1 + math.cos(x * math.pi / args.epochs)) / 2) * \
            (1 - args.lrf) + args.lrf  # cosine

    scheduler = lr_scheduler.LambdaLR(optimizer, lr_lambda=lf)

    for epoch in range(args.epochs):
        # 训练并记录指标
        train_loss, train_acc = train_one_epoch(model=model,
                                                optimizer=optimizer,
                                                data_loader=train_loader,
                                                device=device,
                                                epoch=epoch)
        train_losses.append(train_loss)
        train_accuracies.append(train_acc)

        # 验证并记录指标
        val_loss, val_acc = evaluate(model=model,
                                     data_loader=val_loader,
                                     device=device,
                                     epoch=epoch)
        val_losses.append(val_loss)
        val_accuracies.append(val_acc)

        # 打印指标
        print(f"Epoch {epoch}: Train Loss={train_loss:.4f}, Val Loss={val_loss:.4f}")
        print(f"          Train Acc={train_acc:.4f}, Val Acc={val_acc:.4f}")

        # 训练结束后绘制曲线
        plt.figure(figsize=(12, 5))

        # 绘制损失曲线
        plt.subplot(1, 2, 1)
        plt.plot(train_losses, label='Train Loss', marker='o')
        plt.plot(val_losses, label='Validation Loss', marker='o')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.title('Training and Validation Loss')
        plt.grid(True)
        plt.legend()

        # 绘制准确率曲线
        plt.subplot(1, 2, 2)
        plt.plot(train_accuracies, label='Train Accuracy', marker='o')
        plt.plot(val_accuracies, label='Validation Accuracy', marker='o')
        plt.xlabel('Epoch')
        plt.ylabel('Accuracy')
        plt.title('Training and Validation Accuracy')
        plt.grid(True)
        plt.legend()

        # 保存图片到指定目录
        save_dir = r"D:\gz\Strawberry草莓2类总计4498/results"
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        plt.savefig(os.path.join(save_dir, 'training_metrics.png'))
        plt.close()  # 关闭图像，避免内存泄漏

        tags = ["train_loss", "train_acc",
                "val_loss", "val_acc", "learning_rate"]
        tb_writer.add_scalar(tags[0], train_loss, epoch)
        tb_writer.add_scalar(tags[1], train_acc, epoch)
        tb_writer.add_scalar(tags[2], val_loss, epoch)
        tb_writer.add_scalar(tags[3], val_acc, epoch)
        tb_writer.add_scalar(tags[4], optimizer.param_groups[0]["lr"], epoch)

        torch.save(model.state_dict(), "./weights/model-{}.pth".format(epoch))


def parse_opt(known=False):
    parser = argparse.ArgumentParser()
    parser.add_argument('--num_classes', type=int, default=5)
    parser.add_argument('--epochs', type=int, default=50)
    parser.add_argument('--batch-size', type=int, default=32)
    parser.add_argument('--lr', type=float, default=0.03)
    parser.add_argument('--lrf', type=float, default=0.01)
    #parser.add_argument('--weight_decay', type=float, default=0.001)
    parser.add_argument('--data-path', type=str, default=r"D:\gz\小麦\论文\论文中的数据小麦\train")  # 车牌路劲"card-data"
    parser.add_argument('--weights', type=str,
                        default='\gz\gzz\Classification-of-flowers-Master\Efficientnet\weights\pre_efficientnetv2-s.pth',
                        help='initial weights path')
    parser.add_argument('--freeze-layers', type=bool, default=True)
    parser.add_argument('--device', default='cuda:0', help='device id (i.e. 0 or 0,1 or cpu)')

    return parser.parse_known_args()[0] if known else parser.parse_args()


if __name__ == '__main__':
    opt = parse_opt()
    main(opt)
