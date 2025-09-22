from ultralytics import YOLO
# 原始模型
model = YOLO("yolov11n.yaml")
print(f"Original Params: {sum(p.numel() for p in model.parameters()) / 1e6:.2f}M")

# 修改后模型
model_cbam = YOLO("yolov11_cbam.yaml")
print(f"CBAM Version Params: {sum(p.numel() for p in model_cbam.parameters()) / 1e6:.2f}M")

# 预期输出示例：
# Original Params: 7.21M
# CBAM Version Params: 7.38M (+2.3%)