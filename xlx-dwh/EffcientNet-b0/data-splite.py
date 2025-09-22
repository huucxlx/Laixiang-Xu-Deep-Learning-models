import os
import random
import shutil
from sklearn.model_selection import train_test_split

def stratified_split_dataset(data_dir, train_ratio=0.60, val_ratio=0.20, test_ratio=0.20):
    """
    分层划分数据集为训练集、验证集和测试集
    :param data_dir: 数据集根目录，每个类别为一个子目录
    :param train_ratio: 训练集比例
    :param val_ratio: 验证集比例
    :param test_ratio: 测试集比例
    """
    assert train_ratio + val_ratio + test_ratio == 1, "比例之和必须为 1"

    classes = os.listdir(data_dir)
    for cls in classes:
        cls_dir = os.path.join(data_dir, cls)
        if os.path.isdir(cls_dir):
            images = os.listdir(cls_dir)

            # 第一次划分，划分出训练集和临时集（验证集 + 测试集）
            train_images, temp_images = train_test_split(images, train_size=train_ratio, stratify=[cls] * len(images))

            # 计算验证集和测试集的比例
            remaining_ratio = val_ratio + test_ratio
            val_ratio_adj = val_ratio / remaining_ratio

            # 第二次划分，从临时集中划分出验证集和测试集
            val_images, test_images = train_test_split(temp_images, train_size=val_ratio_adj, stratify=[cls] * len(temp_images))

            # 创建训练集、验证集和测试集的目录
            train_dir = os.path.join(data_dir, 'train', cls)
            val_dir = os.path.join(data_dir, 'val', cls)
            test_dir = os.path.join(data_dir, 'test', cls)

            os.makedirs(train_dir, exist_ok=True)
            os.makedirs(val_dir, exist_ok=True)
            os.makedirs(test_dir, exist_ok=True)

            # 移动图像到相应的目录
            for img in train_images:
                src = os.path.join(cls_dir, img)
                dst = os.path.join(train_dir, img)
                shutil.copyfile(src, dst)

            for img in val_images:
                src = os.path.join(cls_dir, img)
                dst = os.path.join(val_dir, img)
                shutil.copyfile(src, dst)

            for img in test_images:
                src = os.path.join(cls_dir, img)
                dst = os.path.join(test_dir, img)
                shutil.copyfile(src, dst)

# 使用示例
data_dir = r'D:\pycharm\pycharmprojects\learn-pytorch\EffcientNet-b0\data\soybean'
stratified_split_dataset(data_dir)