import os
import json

import torch
from PIL import Image
from torchvision import transforms
import matplotlib.pyplot as plt

from model_optimized import LicensePlateAlexNet
#from model_optimized import LicensePlateAlexNet
#from model__both_enhanced02 import DualAttentionAlexNet


def main():
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"Using {device} device for inference")

    # 图像预处理
    data_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])

    # 加载测试图像（修改为你需要测试的车牌图片路径）
    img_path = r"D:\deep_leaning\Alexnet\card-data (2)\card-data\test\yellow card\IMG_20250401_114827.jpg"
    assert os.path.exists(img_path), f"File '{img_path}' does not exist."

    img = Image.open(img_path)
    plt.imshow(img)

    # 图像预处理并添加batch维度
    img_tensor = data_transform(img)
    img_tensor = torch.unsqueeze(img_tensor, dim=0)

    # 加载类别标签（确保这是你训练时生成的json文件）
    json_path = './class_indices.json'
    assert os.path.exists(json_path), f"Class indices file '{json_path}' does not exist."

    with open(json_path, "r") as f:
        class_indict = json.load(f)

    # 检查类别数量是否匹配
    num_classes = len(class_indict)
    print(f"Loaded {num_classes} classes: {class_indict}")

    # 初始化模型（修改num_classes为你的车牌类别数）
    model = LicensePlateAlexNet(num_classes=4).to(device)

    # 加载训练好的权重（修改为你的车牌模型权重路径）
    weights_path = r"D:\deep_leaning\Alexnet\AlexNet_Simplified_Plate.pth"
    assert os.path.exists(weights_path), f"Model weights '{weights_path}' does not exist."

    model.load_state_dict(torch.load(weights_path, map_location=device, weights_only=True))
    print(f"Loaded weights from {weights_path}")

    # 推理预测
    model.eval()
    with torch.no_grad():
        output = torch.squeeze(model(img_tensor.to(device))).cpu()
        predict = torch.softmax(output, dim=0)
        predict_cla = torch.argmax(predict).item()

    # 显示预测结果
    print("\nPrediction Result:")
    print(f"Most likely class: {class_indict[str(predict_cla)]} with probability: {predict[predict_cla].item():.3f}")

    print("\nAll class probabilities:")
    for i in range(len(predict)):
        print(f"Class {i:2} ({class_indict[str(i)]:15}): {predict[i].item():.4f}")

    # 在图像上显示预测结果
    plt.title(f"Prediction: {class_indict[str(predict_cla)]} ({predict[predict_cla].item():.2f})")
    plt.axis('off')
    plt.savefig('result.png')


if __name__ == '__main__':
    main()