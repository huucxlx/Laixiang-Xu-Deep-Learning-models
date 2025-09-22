import os

def convert_segmentation_to_box(labels_dir):
    """
    将分割标注文件转换为框标注文件，并覆盖原文件。

    Args:
        labels_dir (str): 分割标注文件的目录路径。

    Returns:
        None
    """
    for filename in os.listdir(labels_dir):
        input_path = os.path.join(labels_dir, filename)

        if not filename.endswith('.txt'):
            continue  # 跳过非文本文件

        with open(input_path, 'r') as infile:
            lines = infile.readlines()

        with open(input_path, 'w') as outfile:
            for line in lines:
                parts = line.strip().split()
                if len(parts) <= 5:
                    # 如果不是分割标注格式，则直接写入原内容
                    outfile.write(line)
                    continue

                class_id = parts[0]
                polygon_points = list(map(float, parts[5:]))
                x_coords = polygon_points[::2]  # 偶数索引为x坐标
                y_coords = polygon_points[1::2]  # 奇数索引为y坐标

                # 计算外接矩形
                x_min, x_max = min(x_coords), max(x_coords)
                y_min, y_max = min(y_coords), max(y_coords)
                x_center = (x_min + x_max) / 2
                y_center = (y_min + y_max) / 2
                width = x_max - x_min
                height = y_max - y_min

                # 转换为框标注格式
                box_annotation = f"{class_id} {x_center:.6f} {y_center:.6f} {width:.6f} {height:.6f}\n"
                outfile.write(box_annotation)

    print(f"转换完成，原标签文件已更新。")

# 使用示例
if __name__ == "__main__":
    labels_directory = r"D:\LiousChen\Develop\Model\yolov11\v11Text\ultralytics-8.3.39\datasets2\valid\labels"  # 替换为你的分割标注目录路径
    convert_segmentation_to_box(labels_directory)