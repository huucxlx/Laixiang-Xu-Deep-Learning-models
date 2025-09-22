import os
import json
import torch
from PIL import Image
import matplotlib.pyplot as plt
from torchvision import transforms
from model import mobilenetv3_small


def load_finetuned_model(weights_path, device):
    """加载微调后的完整模型"""
    if not os.path.exists(weights_path):
        raise FileNotFoundError(f"Model weights not found at {weights_path}")

    checkpoint = torch.load(weights_path, map_location=device)

    # 从checkpoint获取类别数
    num_classes = checkpoint.get('num_classes', None)
    if num_classes is None:
        # 如果checkpoint中没有明确存储类别数，尝试从state_dict推断
        last_layer_key = [k for k in checkpoint.keys() if 'classifier' in k or 'fc' in k][-1]
        num_classes = checkpoint[last_layer_key].shape[0]

    # 初始化模型
    model = mobilenetv3_small(num_classes=num_classes).to(device)

    # 加载状态字典
    if 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
    else:
        # 处理可能的key前缀问题
        state_dict = checkpoint
        # 移除可能的'module.'前缀(如果是多GPU训练保存的)
        state_dict = {k.replace('module.', ''): v for k, v in state_dict.items()}
        model.load_state_dict(state_dict)

    model.eval()
    print(f"Successfully loaded finetuned model with {num_classes} classes")

    # 返回类别映射
    class_mapping = checkpoint.get('class_mapping', None)
    return model, class_mapping


def predict_and_visualize_single_image(img_path, model, device, class_indict, transform):
    """预测单张图片并可视化结果"""
    try:
        img = Image.open(img_path).convert('RGB')
        plt_img = img.copy()  # 保存原始图像用于显示

        # 应用预处理
        img_tensor = transform(img).unsqueeze(0).to(device)

        with torch.no_grad():
            output = model(img_tensor)
            probabilities = torch.softmax(output, dim=1)
            top_probs, top_classes = torch.topk(probabilities, k=3)  # 获取top3预测

        # 准备结果显示
        result_text = "Top Predictions:\n"
        for i, (prob, class_idx) in enumerate(zip(top_probs[0], top_classes[0])):
            class_name = class_indict.get(str(class_idx.item()), f"Class {class_idx.item()}")
            result_text += f"{i + 1}. {class_name}: {prob.item():.2%}\n"

        # 可视化
        plt.figure(figsize=(12, 6))

        # 原始图像
        plt.subplot(1, 2, 1)
        plt.imshow(plt_img)
        plt.title(f"Input Image\n{os.path.basename(img_path)}")
        plt.axis('off')

        # 预测结果
        plt.subplot(1, 2, 2)
        plt.text(0.1, 0.5, result_text,
                 fontsize=12,
                 bbox=dict(facecolor='white', alpha=0.8),
                 verticalalignment='center')
        plt.axis('off')

        plt.tight_layout()
        plt.show()

        return class_indict.get(str(top_classes[0, 0].item())), top_probs[0, 0].item()

    except Exception as e:
        print(f"Error processing {img_path}: {str(e)}")
        return None, None


def load_class_mapping(json_path=None, checkpoint=None):
    """加载类别映射"""
    class_mapping = None

    # 首先尝试从checkpoint加载
    if checkpoint is not None:
        class_mapping = checkpoint.get('class_mapping', None)

    # 如果checkpoint中没有，尝试从JSON文件加载
    if class_mapping is None and json_path is not None:
        if os.path.exists(json_path):
            with open(json_path, "r") as f:
                class_mapping = json.load(f)

    if class_mapping is None:
        raise ValueError("Class mapping not found in checkpoint or JSON file")

    # 确保映射格式正确
    if isinstance(class_mapping, dict):
        # 检查是否需要反转字典
        if all(isinstance(k, str) for k in class_mapping.keys()):
            return class_mapping
        else:
            return {str(v): k for k, v in class_mapping.items()}
    else:
        raise ValueError("Invalid class mapping format")


def main():
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # 数据预处理 - 必须与训练时一致！
    data_transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    # 路径设置 - 修改为您自己的路径
    img_path = r"C:\Users\duanyiru\Desktop\深度学习论文\lan.png"  # 测试图片路径
    weights_path = r"D:\myself\mobilNetV3\results\best_model.pth"  # 模型权重路径
    json_path = r"D:\myself\mobilNetV3\results\class_mapping.json"  # 类别映射路径(可选)

    # 检查文件存在
    if not os.path.exists(img_path):
        raise FileNotFoundError(f"Image not found: {img_path}")
    if not os.path.exists(weights_path):
        raise FileNotFoundError(f"Model weights not found: {weights_path}")

    # 加载模型和类别映射
    model, class_mapping = load_finetuned_model(weights_path, device)
    class_indict = load_class_mapping(json_path, checkpoint={'class_mapping': class_mapping})

    print("Available classes:", class_indict)

    # 执行预测
    class_name, prob = predict_and_visualize_single_image(
        img_path, model, device, class_indict, data_transform)

    if class_name:
        print("\nFinal Prediction:")
        print(f"- Image: {os.path.basename(img_path)}")
        print(f"- Predicted Class: {class_name}")
        print(f"- Probability: {prob:.2%}")

        # 打印所有类别概率（诊断用）
        print("\nFull class probabilities:")
        with torch.no_grad():
            img = Image.open(img_path).convert('RGB')
            img_tensor = data_transform(img).unsqueeze(0).to(device)
            output = model(img_tensor)
            probabilities = torch.softmax(output, dim=1)

            for class_idx in range(probabilities.size(1)):
                class_name = class_indict.get(str(class_idx), f"Class {class_idx}")
                print(f"{class_name}: {probabilities[0, class_idx].item():.4f}")


if __name__ == '__main__':
    try:
        main()
    except Exception as e:
        print(f"Error occurred: {str(e)}")
        print("Possible solutions:")
        print("1. Check all file paths are correct")
        print("2. Ensure model architecture matches the saved weights")
        print("3. Verify your class mapping file format")
        print("4. Make sure input image is valid RGB image")