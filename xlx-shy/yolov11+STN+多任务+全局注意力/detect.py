from ultralytics import YOLO

# Load a model
model = YOLO("D:/LiousChen/Develop/Model/yolov11/v11Text/ultralytics-8.3.39/runs/detect/train5/weights/best.pt")

results = model(
    "D:/LiousChen/Develop/Model/yolov11/v11Text/ultralytics-8.3.39/datasets_clear/test/images",
    conf=0.25,  # 置信度阈值
    save=False
)

