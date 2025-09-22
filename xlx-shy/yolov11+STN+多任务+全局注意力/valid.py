from ultralytics import YOLO

# Load a model
model = YOLO("D:/LiousChen/Develop/Model/yolov11/v11Text/ultralytics-8.3.39/runs/detect/train9_addCDFA_10/weights/best.pt")

if __name__ == '__main__':
    results = model.val(
        data='./data.yaml',
        imgsz=640,  # 输入图像的大小
        batch=16,   # 批次大小
        conf=0.25,  # 置信度阈值
        device=0,   # 使用的设备，0表示使用GPU，-1表示使用CPU
        project='./runs/detect',  # 保存验证结果的目录
        name='val7_addCDFA_10',  # 验证结果的名称
    )
