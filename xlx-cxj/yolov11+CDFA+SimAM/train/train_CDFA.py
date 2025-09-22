from ultralytics import YOLO

model = YOLO("yolo11n-CDFA.yaml")  # 加载预训练模型

if __name__ == '__main__':
    results = model.train(

        # --- 核心训练参数 ---
        data='./data.yaml',  # 数据集配置文件的路径
        epochs=100,  # 训练的轮数
        imgsz=640,  # 输入图像的大小
        batch=16,  # 批次大小
        optimizer='AdamW',  # 优化器类型

        # --- 学习率优化配置 ---
        lr0=0.01,  # 初始学习率
        lrf=0.01,  # 最终学习率
        warmup_epochs=15,  # 学习率预热阶段（缓解初始震荡）
        warmup_momentum=0.8, # 学习率预热阶段的动量
        warmup_bias_lr=0.1, # 偏置项预热学习率（避免初始阶段过激更新）
        weight_decay=0.0005,  # 权重衰减（L2正则化）

        # --- 硬件加速配置 ---
        device=0,  # 使用的设备，0表示使用GPU，-1表示使用CPU
        workers=6,  # 数据加载的工作进程数量

        # --- 模型保存与监控 ---
        project='./runs/detect',  # 保存训练结果的目录
        name='train14_addCDFA_100',  # 训练结果的名称
    )
