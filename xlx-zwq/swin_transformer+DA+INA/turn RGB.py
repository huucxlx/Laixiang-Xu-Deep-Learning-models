import os
from PIL import Image
from tqdm import tqdm


def convert_images_to_rgb(input_folder, output_folder):
    """
    批量将图像转换为 RGB 模式并保存到指定输出文件夹。

    Args:
        input_folder (str): 输入文件夹路径（包含原始图像）。
        output_folder (str): 输出文件夹路径（保存转换后的 RGB 图像）。
    """
    # 确保输出文件夹存在
    os.makedirs(output_folder, exist_ok=True)

    # 遍历输入文件夹中的所有文件
    for root, _, files in os.walk(input_folder):
        for file in tqdm(files, desc="处理文件"):
            if not file.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp')):
                continue  # 跳过非图像文件

            # 构建输入和输出路径
            input_path = os.path.join(root, file)
            relative_path = os.path.relpath(root, input_folder)
            output_subfolder = os.path.join(output_folder, relative_path)
            os.makedirs(output_subfolder, exist_ok=True)
            output_path = os.path.join(output_subfolder, file)

            try:
                # 打开图像并转换为 RGB 模式
                with Image.open(input_path) as img:
                    img_rgb = img.convert('RGB')
                    # 保存为 RGB 图像
                    img_rgb.save(output_path)
            except Exception as e:
                print(f"处理文件 {input_path} 时出错: {e}")


# 使用方法
input_folder = r"D:\split_datasets\test\green car"  # 原始数据集路径
output_folder = r"D:\split_datasets\test2\green car"  # 转换后的数据集路径

convert_images_to_rgb(input_folder, output_folder)