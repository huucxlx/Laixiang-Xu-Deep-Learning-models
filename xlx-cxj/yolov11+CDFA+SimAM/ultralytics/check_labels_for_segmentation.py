import os

def check_labels_for_segmentation(labels_dir):
    """
    检查标签目录中的每个标签文件是否包含分割标注格式。

    Args:
        labels_dir (str): 标签文件所在的目录路径。

    Returns:
        list: 包含分割标注格式的文件路径。
    """
    segmentation_files = []  # 用于存储包含分割标注的文件

    for filename in os.listdir(labels_dir):
        # 构建完整的标签文件路径
        filepath = os.path.join(labels_dir, filename)
        if not filepath.endswith('.txt'):
            continue  # 跳过非文本文件

        with open(filepath, 'r') as f:
            lines = f.readlines()

        for line in lines:
            parts = line.strip().split()  # 分割每一行
            if len(parts) > 5:  # 分割标注通常会有多边形点集，字段数大于5
                # 检查是否含有多边形点集
                try:
                    floats = [float(x) for x in parts[5:]]  # 从第6个字段开始应为多边形点坐标
                    if len(floats) % 2 == 0:  # 检查点集是否为偶数对 (x, y)
                        segmentation_files.append(filepath)
                        break  # 当前文件已经确定为分割标注格式，无需再检查其余行
                except ValueError:
                    pass  # 如果无法转换为浮点数，跳过该行

    return segmentation_files


# 使用示例
if __name__ == "__main__":
    labels_directory = r"D:\LiousChen\Develop\Model\yolov11\v11Text\ultralytics-8.3.39\datasets3\train\labels"  # 替换为你的 labels 文件夹路径
    segmentation_files = check_labels_for_segmentation(labels_directory)

    if segmentation_files:
        print("以下文件包含分割标注格式：")
        for file in segmentation_files:
            print(file)
    else:
        print("未找到包含分割标注格式的文件。")