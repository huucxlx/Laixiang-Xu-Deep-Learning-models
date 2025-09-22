import os
import json
import torch
from PIL import Image
from torchvision import transforms
from model import AlexNet


def main():
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # 数据预处理
    data_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])

    # 加载类别标签
    json_path = './class_indices.json'
    assert os.path.exists(json_path), f"file: '{json_path}' does not exist."
    with open(json_path, "r") as f:
        class_indict = json.load(f)

    # 创建模型并加载权重
    model = AlexNet(num_classes=4).to(device)
    weights_path = "./AlexNet.pth"
    assert os.path.exists(weights_path), f"file: '{weights_path}' does not exist."
    model.load_state_dict(torch.load(weights_path))
    model.eval()

    # 测试文件夹路径
    test_dir = "D:\\team\\alexnet learning\\deep-learning-for-image-processing-master-juanji-msa\\data_set\\car_data\\test"
    assert os.path.exists(test_dir), f"folder: '{test_dir}' does not exist."

    # 初始化统计变量
    correct = 0
    total = 0
    supported_formats = ('.png', '.jpg', '.jpeg')

    # 遍历测试集
    for class_name in os.listdir(test_dir):
        class_dir = os.path.join(test_dir, class_name)
        if not os.path.isdir(class_dir):
            continue

        for img_name in os.listdir(class_dir):
            img_path = os.path.join(class_dir, img_name)

            # 跳过非图片文件
            if not img_name.lower().endswith(supported_formats):
                print(f"跳过不支持的文件: {img_name}")
                continue

            try:
                # 图像预处理
                img = Image.open(img_path).convert('RGB')
                img_tensor = data_transform(img).unsqueeze(0).to(device)

                # 推理预测
                with torch.no_grad():
                    output = model(img_tensor)
                    predict = torch.softmax(output, dim=1)
                    confidence, pred_idx = torch.max(predict, 1)

                # 转换结果
                confidence = confidence.item() * 100  # 转换为百分比
                pred_label = class_indict[str(pred_idx.item())]
                true_label = class_name
                is_correct = (true_label == pred_label)

                # 更新统计
                total += 1
                if is_correct:
                    correct += 1
                current_acc = 100.0 * correct / total

                # 输出详细信息
                print(f"图片: {img_name:15} | 真实: {true_label:10} | 预测: {pred_label:10} | "
                      f"置信度: {confidence:5.2f}% | 当前准确率: {current_acc:6.3f}%")

            except Exception as e:
                print(f"处理图片 {img_name} 时出错: {str(e)}")
                continue

    # 最终统计结果
    final_accuracy = 100.0 * correct / total if total != 0 else 0.0
    print("\n测试结果汇总:")
    print(f"总测试样本数: {total}")
    print(f"正确预测数: {correct}")
    print(f"最终测试准确率: {final_accuracy:.3f}%")


if __name__ == '__main__':
    main()
