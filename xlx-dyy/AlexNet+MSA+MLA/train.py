import os
import sys
import json

import torch
import torch.nn as nn
from torchvision import transforms, datasets, utils
# import matplotlib.pyplot as plt
import numpy as np
import torch.optim as optim
from tqdm import tqdm

from model import AlexNet


def main():
    torch.cuda.empty_cache()  # 释放未使用的显存

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    # device = "cpu"
    print("using {} device.".format(device))

    data_transform = {
        "train": transforms.Compose([
            transforms.Resize(256),  # 缩小输入尺寸
            transforms.RandomCrop(224),  # 减小裁剪尺寸
            # 移除RandomHorizontalFlip
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ]),
        "val": transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])
    }
    data_root = r'D:/team/alexnet learning/deep-learning-for-image-processing-master-juanji-msa/data_set/car_data'
    train_dataset = datasets.ImageFolder(os.path.join(data_root, 'train'), data_transform['train'])
    train_num = len(train_dataset)

    flower_list = train_dataset.class_to_idx
    cla_dict = dict((val, key) for key, val in flower_list.items())
    with open('class_indices.json', 'w') as json_file:
        json.dump(cla_dict, json_file, indent=4)

    batch_size = 16
    nw = min([os.cpu_count(), batch_size if batch_size > 1 else 0, 2])
    print('Using {} dataloader workers every process'.format(nw))

    train_loader = torch.utils.data.DataLoader(train_dataset,
                                               batch_size=batch_size, shuffle=True,
                                               num_workers=nw)

    validate_dataset = datasets.ImageFolder(os.path.join(data_root, 'val'), data_transform['val'])
    val_num = len(validate_dataset)
    validate_loader = torch.utils.data.DataLoader(validate_dataset,
                                                  batch_size=batch_size, shuffle=False,
                                                  num_workers=nw)

    print("using {} images for training, {} images for validation.".format(train_num, val_num))

    # 模型初始化
    net = AlexNet(num_classes=4, init_weights=True)
    net.to(device)
    print(net)
    loss_function = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(net.parameters(), lr=0.0002, weight_decay=0.01)
    # scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', patience=3, factor=0.5)

    # 早停参数设置
    epochs = 30
    patience = 5  # 允许连续不提升的epoch数
    best_acc = 0.0
    counter_no_improve = 0  # 未提升计数器
    early_stop = False

    save_path = './AlexNet.pth'
    results_file = 'training_results.csv'
    with open(results_file, 'w') as f:
        f.write('Epoch,Train Loss,Train Acc,Val Loss,Val Acc\n')

    for epoch in range(epochs):
        if early_stop:
            print("Early stopping triggered!")
            break

        # train
        net.train()
        running_loss = 0.0
        train_correct = 0
        train_bar = tqdm(train_loader, file=sys.stdout)
        for step, data in enumerate(train_bar):
            images, labels = data
            optimizer.zero_grad()
            outputs = net(images.to(device))
            loss = loss_function(outputs, labels.to(device))
            loss.backward()
            optimizer.step()

            batch_size = images.size(0)
            running_loss += loss.item() * batch_size
            _, predicted = torch.max(outputs, 1)
            train_correct += (predicted == labels.to(device)).sum().item()
            train_bar.desc = "train epoch[{}/{}] loss:{:.3f}".format(epoch + 1, epochs, loss)

        train_loss = running_loss / train_num
        train_acc = train_correct / train_num

        # validate
        net.eval()
        val_loss = 0.0
        val_correct = 0
        with torch.no_grad():
            val_bar = tqdm(validate_loader, file=sys.stdout)
            for val_data in val_bar:
                val_images, val_labels = val_data
                outputs = net(val_images.to(device))
                loss = loss_function(outputs, val_labels.to(device))
                batch_size = val_labels.size(0)
                val_loss += loss.item() * batch_size
                predict_y = torch.max(outputs, dim=1)[1]
                val_correct += (predict_y == val_labels.to(device)).sum().item()

        val_loss = val_loss / val_num
        val_acc = val_correct / val_num
        # scheduler.step(val_acc)
        print('[epoch %d] train_loss: %.3f train_acc: %.3f val_loss: %.3f val_accuracy: %.3f' %
              (epoch + 1, train_loss, train_acc, val_loss, val_acc))

        # 写入CSV文件
        with open(results_file, 'a') as f:
            f.write(f'{epoch + 1},{train_loss:.4f},{train_acc:.4f},{val_loss:.4f},{val_acc:.4f}\n')

        # 早停机制判断
        if val_acc > best_acc:
            best_acc = val_acc
            torch.save(net.state_dict(), save_path)
            counter_no_improve = 0  # 重置计数器
        else:
            counter_no_improve += 1
            print(f'Validation accuracy did not improve for {counter_no_improve}/{patience} epochs')

            if counter_no_improve >= patience:
                early_stop = True
                print(f"Early stopping after {epoch + 1} epochs with best acc: {best_acc:.4f}")

    print('Finished Training')


if __name__ == '__main__':
    main()
