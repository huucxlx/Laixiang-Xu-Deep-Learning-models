import os
import json
import torch
from PIL import Image
from torchvision import transforms
from tqdm import tqdm
from Model.EfficientNet import efficient0
from Model.SKNet import sknet50


def load_model(model_name, num_classes, weight_path, device):
    """加载模型工厂函数，支持多种模型类型"""
    model_dict = {
        'efficient0': efficient0,
        'sknet50': sknet50
    }

    if model_name not in model_dict:
        raise ValueError(f"Unsupported model: {model_name}")

    model = model_dict[model_name](num_classes=num_classes).to(device)

    # 加载权重
    if os.path.exists(weight_path):
        state_dict = torch.load(weight_path, map_location=device)
        if 'state_dict' in state_dict:  # 如果是完整检查点
            model.load_state_dict(state_dict['state_dict'])
        else:  # 如果是纯权重文件
            model.load_state_dict(state_dict)
    else:
        raise FileNotFoundError(f"Model weights not found at {weight_path}")

    model.eval()
    return model


def main():
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # 配置参数
    config = {
        "model_name": "efficient0",  # 可切换为"efficient0"
        "img_size": 224,
        "weight_path": "./weights/model-29.pth",
        "test_dir": "D:\\car-data\\val"
    }

    # 图像预处理
    data_transform = transforms.Compose([
        transforms.Resize(config["img_size"]),
        transforms.CenterCrop(config["img_size"]),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    # 加载类别标签
    json_path = './class_indices.json'
    assert os.path.exists(json_path), f"file: '{json_path}' does not exist."
    with open(json_path, "r") as f:
        class_indict = json.load(f)

    # 初始化模型
    model = load_model(
        model_name=config["model_name"],
        num_classes=len(class_indict),
        weight_path=config["weight_path"],
        device=device
    )

    # 统计结果
    correct = 0
    total = 0
    class_correct = {class_name: 0 for class_name in class_indict.values()}
    class_total = {class_name: 0 for class_name in class_indict.values()}

    # 遍历测试集
    for class_name in os.listdir(config["test_dir"]):
        class_dir = os.path.join(config["test_dir"], class_name)
        if not os.path.isdir(class_dir):
            continue

        true_class_idx = [k for k, v in class_indict.items() if v == class_name][0]

        # 使用进度条显示
        image_files = [f for f in os.listdir(class_dir) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
        for img_file in tqdm(image_files, desc=f'Processing {class_name}'):
            img_path = os.path.join(class_dir, img_file)

            try:
                img = Image.open(img_path).convert('RGB')  # 确保RGB格式
                img = data_transform(img)
                img = torch.unsqueeze(img, dim=0).to(device)

                with torch.no_grad():
                    # 处理可能返回的注意力图
                    output = model(img)
                    if isinstance(output, tuple):  # 如果模型返回(output, attention)
                        output = output[0]
                    output = torch.squeeze(output)
                    predict = torch.softmax(output, dim=0)
                    predict_cla = torch.argmax(predict).item()

                # 更新统计
                total += 1
                class_total[class_name] += 1
                if str(predict_cla) == true_class_idx:
                    correct += 1
                    class_correct[class_name] += 1

            except Exception as e:
                print(f"\nError processing {img_path}: {str(e)}")
                continue

    # 打印结果
    print(f"\nModel: {config['model_name']}")
    print(f"Total Accuracy: {100 * correct / total:.2f}% ({correct}/{total})")

    print("\nClass-wise Accuracy:")
    for class_name in sorted(class_indict.values()):
        if class_total[class_name] > 0:
            acc = 100 * class_correct[class_name] / class_total[class_name]
            print(f"{class_name:20s}: {acc:.2f}% ({class_correct[class_name]}/{class_total[class_name]})")


if __name__ == '__main__':
    main()