import os
import json
import torch
from PIL import Image
from torchvision import transforms
from tqdm import tqdm
from original_m import efficient0 as create_model


def main():
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # 图像预处理设置
    img_size = {"B0": 224}
    data_transform = transforms.Compose([
        transforms.Resize(img_size["B0"]),
        transforms.CenterCrop(img_size["B0"]),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    # 加载类别标签
    json_path = './class_indices.json'
    assert os.path.exists(json_path), f"file: '{json_path}' does not exist."
    with open(json_path, "r") as f:
        class_indict = json.load(f)

    # 初始化模型
    model = create_model(num_classes=len(class_indict)).to(device)
    model_weight_path = "./weights/model-29.pth"
    model.load_state_dict(torch.load(model_weight_path, map_location=device))
    model.eval()

    # 设置测试文件夹路径（包含子文件夹，每个子文件夹是一个类别）
    test_dir = "D:\\car-data\\val"  # 假设子文件夹名为类别名
    assert os.path.exists(test_dir), f"dir: '{test_dir}' does not exist."

    # 统计结果
    correct = 0
    total = 0
    class_correct = {class_name: 0 for class_name in class_indict.values()}
    class_total = {class_name: 0 for class_name in class_indict.values()}

    # 遍历测试集
    for class_name in os.listdir(test_dir):
        class_dir = os.path.join(test_dir, class_name)
        if not os.path.isdir(class_dir):
            continue

        true_class_idx = [k for k, v in class_indict.items() if v == class_name][0]

        # 使用进度条显示
        image_files = [f for f in os.listdir(class_dir) if f.endswith(('.jpg', '.jpeg', '.png'))]
        for img_file in tqdm(image_files, desc=f'Processing {class_name}'):
            img_path = os.path.join(class_dir, img_file)

            try:
                img = Image.open(img_path)
                img = data_transform(img)
                img = torch.unsqueeze(img, dim=0).to(device)

                with torch.no_grad():
                    output = torch.squeeze(model(img))
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

    # 打印总体准确率
    print(f"\nTotal Accuracy: {100 * correct / total:.2f}% ({correct}/{total})")

    # 打印每个类别的准确率
    print("\nClass-wise Accuracy:")
    for class_name in class_indict.values():
        if class_total[class_name] > 0:
            acc = 100 * class_correct[class_name] / class_total[class_name]
            print(f"{class_name}: {acc:.2f}% ({class_correct[class_name]}/{class_total[class_name]})")


if __name__ == '__main__':
    main()