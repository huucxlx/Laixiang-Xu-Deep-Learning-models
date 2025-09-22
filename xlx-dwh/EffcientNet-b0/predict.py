'''
import os
import json

import torch
from PIL import Image
from torchvision import transforms
import matplotlib.pyplot as plt

from model import efficientnet_b0 as create_model


def main():
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    img_size = {"B0": 224,
                "B1": 240,
                "B2": 260,
                "B3": 300,
                "B4": 380,
                "B5": 456,
                "B6": 528,
                "B7": 600}
    num_model = "B0"

    data_transform = transforms.Compose(
        [transforms.Resize(img_size[num_model]),
         transforms.CenterCrop(img_size[num_model]),
         transforms.ToTensor(),
         transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])

    # load image
    img_path = r'D:\pycharm\pycharmprojects\learn-pytorch\EffcientNet-b0\data\Citrus柑橘4类总计6262\Black spot-181\CitrusBlackSpot(2).png'
    assert os.path.exists(img_path), "file: '{}' dose not exist.".format(img_path)
    img = Image.open(img_path)
    plt.imshow(img)
    # [N, C, H, W]
    img = data_transform(img)
    # expand batch dimension
    img = torch.unsqueeze(img, dim=0)

    # read class_indict
    json_path = r'D:\pycharm\pycharmprojects\learn-pytorch\EffcientNet-b0\class_indices.json'
    assert os.path.exists(json_path), "file: '{}' dose not exist.".format(json_path)

    with open(json_path, "r") as f:
        class_indict = json.load(f)

    # create model
    model = create_model(num_classes=4).to(device)
    # load model weights
    model_weight_path = r'D:\pycharm\pycharmprojects\learn-pytorch\EffcientNet-b0\weights\model-23.pth'
    model.load_state_dict(torch.load(model_weight_path, map_location=device,weights_only=True))
    model.eval()
    with torch.no_grad():
        # predict class
        output = torch.squeeze(model(img.to(device))).cpu()
        predict = torch.softmax(output, dim=0)
        predict_cla = torch.argmax(predict).numpy()

    print_res = "class: {}   prob: {:.3}".format(class_indict[str(predict_cla)],
                                                 predict[predict_cla].numpy())
    plt.title(print_res)
    for i in range(len(predict)):
        print("class: {:10}   prob: {:.3}".format(class_indict[str(i)],
                                                  predict[i].numpy()))
    plt.show()


if __name__ == '__main__':
    main()
'''
import os
import json
import torch
import numpy as np
from torchvision import transforms, datasets
from torch.utils.data import DataLoader
from sklearn.metrics import confusion_matrix, classification_report
import seaborn as sns
import matplotlib.pyplot as plt
from model import efficientnet_b0 as create_model


def main():
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # 加载类别信息
    json_path = r'D:\pycharm\pycharmprojects\learn-pytorch\EffcientNet-b0\class_indices.json'
    with open(json_path, "r") as f:
        class_indict = json.load(f)

    # 图像预处理（需要与训练时保持一致）
    img_size = 224
    data_transform = transforms.Compose([
        transforms.Resize(img_size),
        transforms.CenterCrop(img_size),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    # 加载模型
    model = create_model(num_classes=len(class_indict)).to(device)
    model_weight_path = r'D:\pycharm\pycharmprojects\learn-pytorch\EffcientNet-b0\weights\best.pth'
    model.load_state_dict(torch.load(model_weight_path, map_location=device))
    model.eval()

    # 加载测试集
    test_dir = r'D:\pycharm\pycharmprojects\learn-pytorch\EffcientNet-b0\data\soybean\test'
    assert os.path.exists(test_dir), f"Test directory {test_dir} not found."

    test_dataset = datasets.ImageFolder(root=test_dir, transform=data_transform)
    test_loader = DataLoader(test_dataset, batch_size=16, shuffle=False, num_workers=2)

    # 进行预测
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for images, labels in test_loader:
            images = images.to(device)
            outputs = model(images)
            _, preds = torch.max(outputs, 1)

            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.numpy())

    # 生成评估报告
    print("\nClassification Report:")
    print(classification_report(all_labels, all_preds, target_names=class_indict.values()))

    # 计算整体准确率
    accuracy = np.mean(np.array(all_preds) == np.array(all_labels))
    print(f"\nOverall Accuracy: {accuracy:.4f}")

    # 生成混淆矩阵
    cm = confusion_matrix(all_labels, all_preds)
    plt.figure(figsize=(12, 10))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=class_indict.values(),
                yticklabels=class_indict.values())
    plt.xlabel('Predicted Labels')
    plt.ylabel('True Labels')
    plt.title('Confusion Matrix')
    plt.show()

    # 各类别准确率
    class_acc = cm.diagonal() / cm.sum(axis=1)
    print("\nClass-wise Accuracy:")
    for class_name, acc in zip(class_indict.values(), class_acc):
        print(f"{class_name}: {acc:.4f}")


if __name__ == '__main__':
    main()