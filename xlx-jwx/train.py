import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import transforms, datasets
from torch.utils.data import DataLoader
from model import MobileNetV2
from torchvision.models import resnet50, ResNet50_Weights
import os
import json
from tqdm import tqdm

# 配置与路径
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
data_root = "D:/py/MobileNetV3-Pytorch-master/data/car_data"
save_path = "./checkpoints/mobilenetV2_distilled.pth"
os.makedirs(os.path.dirname(save_path), exist_ok=True)

# 数据增强
data_transform = {
    "train": transforms.Compose([
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
    "val": transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
    "test": transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
}

# 数据加载
train_dataset = datasets.ImageFolder(os.path.join(data_root, "train"), data_transform["train"])
train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True, num_workers=4, pin_memory=True)
#3.21——多线程，内存锁页加速（不开启内存锁页加速会导致线程问题报错）
val_dataset = datasets.ImageFolder(os.path.join(data_root, "val"), data_transform["val"])
val_loader = DataLoader(val_dataset, batch_size=16, shuffle=False, num_workers=4, pin_memory=True)

test_dataset = datasets.ImageFolder(os.path.join(data_root, "test"), data_transform["test"])
test_loader = DataLoader(test_dataset, batch_size=16, shuffle=False, num_workers=4, pin_memory=True)
# 定义蒸馏损失函数
class DistillLoss(nn.Module):
    def __init__(self, alpha=0.5, T=4):
        super().__init__()
        self.alpha = alpha
        self.T = T
        self.ce = nn.CrossEntropyLoss()
        self.kl = nn.KLDivLoss(reduction='batchmean')

    def forward(self, student_logits, teacher_logits, labels):
        ce_loss = self.ce(student_logits, labels)
        soft_loss = self.kl(
            nn.functional.log_softmax(student_logits / self.T, dim=1),
            nn.functional.softmax(teacher_logits / self.T, dim=1)
        ) * (self.T ** 2)
        return self.alpha * ce_loss + (1 - self.alpha) * soft_loss


# 对抗攻击函数
def fgsm_attack(images, labels, model, epsilon=0.01):
    images = images.to(device).detach().requires_grad_(True)
    outputs = model(images)
    loss = nn.CrossEntropyLoss()(outputs, labels.to(device))
    loss.backward()
    perturbed = images + epsilon * images.grad.sign()#3.20——切断梯度回转
    return torch.clamp(perturbed, 0, 1).detach()


# 早停机制
class EarlyStopping:
    def __init__(self, patience=21, delta=0.001):
        self.patience = patience
        self.delta = delta
        self.counter = 0
        self.best_loss = float('inf')

    def __call__(self, val_loss, model):
        if val_loss < self.best_loss - self.delta:#3.20——直观比较损失下降
            self.best_loss = val_loss
            self.counter = 0
            torch.save(model.state_dict(), save_path)
        else:
            self.counter += 1
            if self.counter >= self.patience:
                return True
        return False


if __name__ == '__main__':
    # 初始化模型
    teacher_model = resnet50(weights=ResNet50_Weights.IMAGENET1K_V1)#3.21——显式指定权重加载
    teacher_model.fc = nn.Linear(teacher_model.fc.in_features, 4)
    teacher_model = teacher_model.to(device)
    for param in teacher_model.parameters():
        param.requires_grad = False

    student_model = MobileNetV2(num_classes=4).to(device)
    optimizer = optim.Adam(student_model.parameters(), lr=0.0001, weight_decay=1e-5)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.1)
    early_stopping = EarlyStopping()
    val_accuracies = []
    # 训练循环
    for epoch in range(20):
        student_model.train()
        train_loss = 0.0
        for images, labels in tqdm(train_loader, desc=f"Epoch {epoch + 1} Training"):
            perturbed = fgsm_attack(images, labels, student_model)
            optimizer.zero_grad()
            student_logits = student_model(perturbed)
            with torch.no_grad():
                teacher_logits = teacher_model(images.to(device))
            loss = DistillLoss()(student_logits, teacher_logits, labels.to(device))
            loss.backward()
            optimizer.step()
            train_loss += loss.item()

        # 验证
        student_model.eval()
        val_loss, correct = 0.0, 0
        with torch.no_grad():
            for images, labels in tqdm(val_loader, desc="Validating"):
                outputs = student_model(images.to(device))#3.20——只评估学生模型（去冗余）
                val_loss += DistillLoss()(outputs, teacher_model(images.to(device)), labels.to(device)).item()
                correct += (outputs.argmax(1) == labels.to(device)).sum().item()

        avg_val_loss = val_loss / len(val_loader)
        val_acc = correct / len(val_dataset)
        scheduler.step()
        val_accuracies.append(val_acc)
        print(
            f"Epoch {epoch + 1}: Train Loss: {train_loss / len(train_loader):.4f} | Val Loss: {avg_val_loss:.4f} | Val Acc: {val_acc:.2%}")

        if early_stopping(avg_val_loss, student_model):
            print("Early stopping triggered!")
            break

    print("Training Complete!\n")
    avg_val_acc = sum(val_accuracies) / len(val_accuracies)
    print(f"Average Validation Accuracy: {avg_val_acc:.2%}\n")
    # 测试评估
    student_model.eval()
    test_acc = 0.0
    with torch.no_grad():
        for images, labels in tqdm(test_loader, desc="Testing"):
            outputs = student_model(images.to(device))
            test_acc += (outputs.argmax(1) == labels.to(device)).sum().item()

    test_accuracy = test_acc / len(test_dataset)
    print(f"\nTest Accuracy: {test_accuracy:.2%}")