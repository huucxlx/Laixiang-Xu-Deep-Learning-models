from concurrent.futures import ThreadPoolExecutor
import json
import os
import threading

import matplotlib.pyplot as plt
import torch
from model import efficientnetv2_s as create_model
from PIL import Image
from torchvision import transforms

# create model
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
model = create_model(num_classes=4).to(device)
# load model weights
model_weight_path = "weights/model-8.pth"
model.load_state_dict(torch.load(model_weight_path, map_location=device))
model.eval()

# read class_indict
json_path = 'class_indices.json'
assert os.path.exists(json_path), f"file: '{json_path}' does not exist."
with open(json_path, "r") as f:
    class_indict = json.load(f)

# read img_dir
path = r"C:\Users\cai\Desktop\car\test"
dirs = os.listdir(path)
imgs = []
true_labels = []
for dir in dirs:
    path_flower = os.path.join(path, dir)
    if os.path.isdir(path_flower):
        files = os.listdir(path_flower)
        for file in files:
            img_path = os.path.join(path_flower, file)
            imgs.append(img_path)
            true_labels.append(dir)

img_size = {"s": [300, 384],  # train_size, val_size
            "m": [384, 480],
            "l": [384, 480]}
num_model = "s"

data_transform = transforms.Compose(
    [transforms.Resize(img_size[num_model][1]),
     transforms.CenterCrop(img_size[num_model][1]),
     transforms.ToTensor(),
     transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])])

# 用于线程安全的锁
lock = threading.Lock()

# 存储每个类别的预测数量、正确预测数量
class_pred_count = {class_name: 0 for class_name in class_indict.values()}
class_correct_count = {class_name: 0 for class_name in class_indict.values()}
total_correct = 0

# 美化输出
def main(img_index, img_path, true_label):
    global total_correct
    # load image
    assert os.path.exists(img_path), f"file: '{img_path}' does not exist."
    img = Image.open(img_path)
    plt.imshow(img)

    # 预处理
    img = data_transform(img)
    img = torch.unsqueeze(img, dim=0)

    with torch.no_grad():
        # 预测
        output = torch.squeeze(model(img.to(device))).cpu()
        predict = torch.softmax(output, dim=0)
        predict_cla = torch.argmax(predict).numpy()
        predicted_class_name = class_indict[str(predict_cla)]

    # 美化输出
    print(f"\n{'=' * 40}")
    print(f"IMAGE #{img_index + 1}: {os.path.basename(img_path)}")
    print(f"{'-' * 40}")
    print(f"{'Class':<15} | {'Probability':<10} | {'Status':<6}")
    print(f"{'-' * 40}")

    for i in range(len(predict)):
        status = "★" if i == predict_cla else ""
        print(f"{class_indict[str(i)]:<15} | {predict[i].numpy():<10.4f} | {status:<6}")

    print(f"{'=' * 40}\n")

    # 线程安全地更新统计信息
    with lock:
        class_pred_count[true_label] += 1
        if predicted_class_name == true_label:
            class_correct_count[true_label] += 1
            total_correct += 1

if __name__ == '__main__':
    with ThreadPoolExecutor(10) as t:
        for index, (img, true_label) in enumerate(zip(imgs, true_labels)):
            t.submit(main, index, img, true_label)

    # 输出结果总结
    print("\nResults Summary:")
    print(f"{'-' * 45}")
    print(f"{'Class':<15} | {'Predicted Count':<15} | {'Correct Count':<15} | {'Accuracy'}")
    print(f"{'-' * 45}")
    total_pred = len(imgs)
    for class_name in class_indict.values():
        pred_count = class_pred_count[class_name]
        correct_count = class_correct_count[class_name]
        accuracy = correct_count / pred_count if pred_count > 0 else 0
        print(f"{class_name:<15} | {pred_count:<15} | {correct_count:<15} | {accuracy:.2%}")
    print(f"{'-' * 45}")
    overall_accuracy = total_correct / total_pred
    print(f"Overall Accuracy: {overall_accuracy:.2%}")