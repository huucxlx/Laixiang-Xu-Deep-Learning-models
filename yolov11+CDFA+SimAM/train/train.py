from ultralytics import YOLO

model = YOLO("yolo11n.yaml")  # 加载预训练模型

if __name__ == '__main__':
    results = model.train(

        # --- 核心训练参数 ---
        data='./data.yaml',  # 数据集配置文件的路径
        epochs=100,  # 训练的轮数
        imgsz=640,  # 输入图像的大小
        batch=16,  # 批次大小

        # --- 硬件加速配置 ---
        device=0,  # 使用的设备，0表示使用GPU，-1表示使用CPU
        workers=6,  # 数据加载的工作进程数量

        # --- 模型保存与监控 ---
        project='./runs/detect',  # 保存训练结果的目录
        name='train14_addCDFA_100',  # 训练结果的名称
    )
