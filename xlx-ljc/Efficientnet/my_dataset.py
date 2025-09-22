import torch
from PIL import Image
from torch.utils.data import Dataset
import warnings  # 用于提示非RGB图像被转换


class MyDataSet(Dataset):
    """自定义数据集（支持自动处理非RGB图像）"""

    def __init__(self, images_path: list, images_class: list, transform=None):
        self.images_path = images_path
        self.images_class = images_class
        self.transform = transform

    def __len__(self):
        return len(self.images_path)

    def __getitem__(self, item):
        img_path = self.images_path[item]
        img = Image.open(img_path)

        # 自动将非RGB图像转换为RGB（例如灰度图、RGBA图）
        if img.mode != 'RGB':
            # 发出警告提示哪些图像被转换（可选）

            img = img.convert('RGB')  # 关键修改：强制转换为RGB

        label = self.images_class[item]

        if self.transform is not None:
            img = self.transform(img)

        return img, label

    @staticmethod
    def collate_fn(batch):
        images, labels = tuple(zip(*batch))
        images = torch.stack(images, dim=0)
        labels = torch.as_tensor(labels)
        return images, labels