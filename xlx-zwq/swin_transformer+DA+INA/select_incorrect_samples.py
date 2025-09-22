"""
该脚本能够把验证集中预测错误的图片挑选出来，并记录在error_records.csv中
"""
import os
import json
import argparse
import sys

import torch
from torchvision import transforms
from tqdm import tqdm

from my_dataset import MyDataSet
from model import swin_tiny_patch4_window7_224 as create_model
from utils import read_split_data  # 假设此函数可获取完整数据


def main(args):
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")

    # 直接获取全部数据 --------------------------------------------------
    # 假设通过设置 split_rate=0 获取全部数据作为验证集
    try:
        _, _, all_images_path, all_images_label = read_split_data(args.data_path, split_rate=0)
    except TypeError:
        # 如果 split_rate 参数不存在，合并训练集和验证集
        train_images_path, train_images_label, val_images_path, val_images_label = read_split_data(args.data_path)
        all_images_path = train_images_path + val_images_path
        all_images_label = train_images_label + val_images_label

    # 打印数据集信息
    print(f"Total images: {len(all_images_path)}")

    # 图像预处理（保持原尺寸设置）
    img_size = 224
    data_transform = transforms.Compose([
        transforms.Resize(int(img_size * 1.143)),
        transforms.CenterCrop(img_size),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    # 实例化完整数据集 --------------------------------------------------
    full_dataset = MyDataSet(
        images_path=all_images_path,
        images_class=all_images_label,
        transform=data_transform
    )

    # 数据加载器配置
    batch_size = args.batch_size
    nw = min([os.cpu_count(), batch_size if batch_size > 1 else 0, 8])
    print(f'Using {nw} dataloader workers')

    data_loader = torch.utils.data.DataLoader(
        full_dataset,
        batch_size=batch_size,
        shuffle=False,  # 必须保持顺序以正确记录文件路径
        pin_memory=True,
        num_workers=nw,
        collate_fn=full_dataset.collate_fn
    )

    # 模型加载（保持原逻辑）
    model = create_model(num_classes=args.num_classes).to(device)
    assert os.path.exists(args.weights), f"权重文件 {args.weights} 不存在"
    model.load_state_dict(torch.load(args.weights, map_location=device))

    # 加载类别标签
    with open('./class_indices.json', "r") as f:
        class_indict = json.load(f)

    # 预测与记录 --------------------------------------------------------
    model.eval()
    error_records = []
    class_stats = {cls: {"correct": 0, "total": 0} for cls in class_indict.values()}

    with torch.no_grad():
        pbar = tqdm(data_loader, desc="Processing", file=sys.stdout)
        for batch_idx, (images, labels) in enumerate(pbar):
            images = images.to(device)
            labels = labels.to(device)

            outputs = model(images)
            preds = torch.argmax(outputs, dim=1)

            # 记录本批次结果
            for i in range(len(labels)):
                global_idx = batch_idx * batch_size + i
                true_label = labels[i].item()
                pred_label = preds[i].item()
                true_cls = class_indict[str(true_label)]
                pred_cls = class_indict[str(pred_label)]

                # 更新统计
                class_stats[true_cls]["total"] += 1
                if true_label == pred_label:
                    class_stats[true_cls]["correct"] += 1
                else:
                    error_records.append(
                        f"{all_images_path[global_idx]} | {true_cls} | {pred_cls}"
                    )

    # 保存错误记录
    with open("error_records.csv", "w") as f:
        f.write("file_path,true_label,predicted_label\n")
        f.write("\n".join(error_records))

    # 打印统计信息
    print("\n分类统计:")
    total_correct = sum(stats["correct"] for stats in class_stats.values())
    total_samples = sum(stats["total"] for stats in class_stats.values())
    for cls, stats in class_stats.items():
        acc = stats["correct"] / stats["total"] if stats["total"] > 0 else 0
        print(f"{cls}: {stats['correct']}/{stats['total']} ({acc:.2%})")
    print(f"\n总体准确率: {total_correct / total_samples:.2%}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--num_classes', type=int, default=4)
    parser.add_argument('--batch-size', type=int, default=2)
    parser.add_argument('--data-path', type=str, default=r"D:\split_datasets\test2")
    parser.add_argument('--weights', type=str, default='./weights/best_model.pth', help='initial weights path')
    parser.add_argument('--device', default='cuda:0', help='device id (i.e. 0 or 0,1 or cpu)')

    args = parser.parse_args()
    main(args)