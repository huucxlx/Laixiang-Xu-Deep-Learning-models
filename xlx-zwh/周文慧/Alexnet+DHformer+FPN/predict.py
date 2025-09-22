import os
import json
import torch
import numpy as np
from PIL import Image
from torchvision import transforms
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
from model import AlexNet
import pandas as pd


def evaluate_dataset():
    # 硬件设置
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # 图像预处理（必须与训练时一致）
    data_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.5] * 3, [0.5] * 3)
    ])

    # 加载类别映射
    json_path = "./class_indices.json"
    assert os.path.exists(json_path), "类别文件不存在"
    with open(json_path, "r") as f:
        class_indict = json.load(f)
    idx_to_class = {v: k for k, v in class_indict.items()}  # 反转字典用于查找真实标签

    # 初始化模型
    model = AlexNet(num_classes=len(class_indict)).to(device)
    weights_path = "./AlexNet.pth"
    assert os.path.exists(weights_path), "模型权重文件不存在"
    model.load_state_dict(torch.load(weights_path, map_location=device))
    model.eval()

    # 数据集路径（假设结构：dataset/test/class_name/*.jpg）
    dataset_root = r"E:\Alexnet1\car-data\test"
    assert os.path.exists(dataset_root), "数据集路径不存在"

    # 结果存储
    all_true = []
    all_pred = []
    results = []
    error_files = []

    # 遍历数据集（支持多级子目录）
    for root, dirs, files in os.walk(dataset_root):
        for file in files:
            if file.lower().endswith(('.png', '.jpg', '.jpeg')):
                img_path = os.path.join(root, file)
                try:
                    # 获取真实标签（从目录名获取）
                    true_class = os.path.basename(root)  # 假设子目录名为真实类别
                    true_label = idx_to_class[true_class]  # 转换为数字标签

                    # 预测处理
                    img = Image.open(img_path).convert('RGB')
                    img_tensor = data_transform(img).unsqueeze(0).to(device)

                    with torch.no_grad():
                        output = model(img_tensor)
                        predict = torch.softmax(output, dim=1)
                        pred_label = torch.argmax(predict).item()
                        prob = predict[0][pred_label].item()

                    # 记录结果
                    all_true.append(int(true_label))
                    all_pred.append(pred_label)
                    results.append({
                        "filename": file,
                        "true_class": true_class,
                        "pred_class": class_indict[str(pred_label)],
                        "confidence": f"{prob:.4f}",
                        "correct": (true_class == class_indict[str(pred_label)])})
                except Exception as e:
                    error_files.append({"file": file, "error": str(e)})

    # 生成评估报告
    print("\n======== 数据集评估报告 ========")
    print(f"总样本数: {len(results)}")
    print(f"错误文件数: {len(error_files)}")

    # 在生成评估报告的部分添加以下代码
    correct_predictions = sum(1 for item in results if item["correct"])
    print(f"正确预测数: {correct_predictions}")
    print(f"分类错误数: {len(results) - correct_predictions}")

    # 分类报告
    print("\n分类指标:")
    print(classification_report(all_true, all_pred, target_names=class_indict.values()))

    # 混淆矩阵
    cm = confusion_matrix(all_true, all_pred)
    cm_df = pd.DataFrame(cm, index=class_indict.values(), columns=class_indict.values())
    print("\n混淆矩阵:")
    print(cm_df)

    # 保存详细结果到CSV
    df = pd.DataFrame(results)
    df.to_csv("dataset_predictions.csv", index=False)

    # 保存错误日志
    if error_files:
        error_df = pd.DataFrame(error_files)
        error_df.to_csv("prediction_errors.csv", index=False)

    # 计算整体准确率
    accuracy = np.sum(np.array(all_true) == np.array(all_pred)) / len(all_true)
    print(f"\n整体准确率: {accuracy:.4f}")


if __name__ == '__main__':
    evaluate_dataset()