import os
import json
import torch
from PIL import Image
from torchvision import transforms
import matplotlib.pyplot as plt
#from model import AlexNet
#from model_optimized import LicensePlateAlexNet
#from model_cood_attention import LicensePlateAlexNet
#from model_hybridcs import LicensePlateAlexNetHybridCS
#from model__both_enhanced02 import DualAttentionAlexNet
from model_LiteCoord_attention import LicensePlateAlexNet
from collections import defaultdict


def predict_single_image(model, device, data_transform, img_path):
    """预测单张图片并返回预测类别索引"""
    img = Image.open(img_path).convert("RGB")  # 统一变量名，确保RGB格式 确保转换为 3 通道
    img_tensor = data_transform(img)
    img_tensor = torch.unsqueeze(img_tensor, dim=0)

    model.eval()
    with torch.no_grad():
        output = torch.squeeze(model(img_tensor.to(device))).cpu()
        predict = torch.softmax(output, dim=0)
        predict_cla = torch.argmax(predict).item()

    return predict_cla


def main():
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"Using {device} device for inference")

    # 图像预处理
    data_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])

    # 加载类别标签
    json_path = './class_indices.json'
    assert os.path.exists(json_path), f"Class indices file '{json_path}' does not exist."

    with open(json_path, "r") as f:
        class_indict = json.load(f)

    # 创建反向映射：类别名 -> 类别索引
    class_name_to_idx = {v: k for k, v in class_indict.items()}
    num_classes = len(class_indict)
    print(f"Loaded {num_classes} classes: {class_indict}")

    # 初始化模型
    model = LicensePlateAlexNet(num_classes=num_classes).to(device)

    # 加载训练好的权重
    weights_path = r"D:\deep_leaning\Alexnet\AlexNet_Simplified_Plate_LiteCoord.pth"
    assert os.path.exists(weights_path), f"Model weights '{weights_path}' does not exist."
    checkpoint = torch.load(weights_path, map_location=device, weights_only=True)
    model.load_state_dict(checkpoint['state_dict'])  # 只加载模型权重部分
    print(f"Loaded weights from {weights_path}")

    # 设置测试数据集文件夹路径（包含按类别分组的子文件夹）
    test_data_dir = r"D:\deep_leaning\Alexnet\card-data (2)\card-data\test"
    assert os.path.isdir(test_data_dir), f"Test data folder '{test_data_dir}' does not exist."

    # 统计结果
    confusion_matrix = torch.zeros(num_classes, num_classes)
    class_correct = list(0. for _ in range(num_classes))
    class_total = list(0. for _ in range(num_classes))
    results = []

    # 遍历每个类别文件夹
    for class_name in os.listdir(test_data_dir):
        class_dir = os.path.join(test_data_dir, class_name)
        if not os.path.isdir(class_dir):
            continue

        # 获取真实类别索引
        true_class_idx = int(class_name_to_idx[class_name])

        # 遍历当前类别下的所有图片
        image_files = [f for f in os.listdir(class_dir)
                       if f.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp'))]

        print(f"\nProcessing class '{class_name}' with {len(image_files)} images...")

        for img_file in image_files:
            img_path = os.path.join(class_dir, img_file)
            try:
                # 预测图片
                pred_class_idx = predict_single_image(model, device, data_transform, img_path)

                # 更新统计
                class_total[true_class_idx] += 1
                if pred_class_idx == true_class_idx:
                    class_correct[true_class_idx] += 1
                confusion_matrix[true_class_idx][pred_class_idx] += 1

                # 记录结果
                results.append({
                    'image_path': img_path,
                    'true_class': class_name,
                    'predicted_class': class_indict[str(pred_class_idx)],
                    'correct': pred_class_idx == true_class_idx
                })

            except Exception as e:
                print(f"Error processing image {img_file}: {str(e)}")

    # 计算并打印总体准确率
    total_images = sum(class_total)
    total_correct = sum(class_correct)
    overall_accuracy = total_correct / total_images if total_images > 0 else 0
    print(f"\n{'=' * 50}")
    print(f"Overall Accuracy: {overall_accuracy:.2%} ({total_correct}/{total_images})")
    print(f"{'=' * 50}\n")

    # 打印每个类别的准确率
    print("\nClass-wise Accuracy:")
    for i in range(num_classes):
        accuracy = class_correct[i] / class_total[i] if class_total[i] > 0 else 0
        print(f"{class_indict[str(i)]:15}: {accuracy:.2%} ({int(class_correct[i])}/{int(class_total[i])})")

    # 打印混淆矩阵
    print("\nConfusion Matrix:")
    print("True \\ Pred", end="")
    for i in range(num_classes):
        print(f"{class_indict[str(i)]:>15}", end="")
    print()

    for i in range(num_classes):
        print(f"{class_indict[str(i)]:15}", end="")
        for j in range(num_classes):
            print(f"{confusion_matrix[i][j]:>15.0f}", end="")
        print(f"  | {class_indict[str(i)]}")

    # 保存详细结果到文件
    os.makedirs("prediction07_results", exist_ok=True)
    results_file = "prediction07_results/detailed_results.csv"

    with open(results_file, "w", encoding="utf-8") as f:
        f.write("Image Path,True Class,Predicted Class,Correct\n")
        for result in results:
            f.write(f"{result['image_path']},{result['true_class']},{result['predicted_class']},{result['correct']}\n")

    print(f"\nDetailed results saved to {results_file}")


if __name__ == '__main__':
    main()