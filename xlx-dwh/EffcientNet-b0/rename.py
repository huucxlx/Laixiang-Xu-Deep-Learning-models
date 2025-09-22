import os
from datetime import datetime

def batch_rename_images(folder_path, prefix="image_", start_num=1, digits=4):
    """
    批量重命名图片文件
    :param folder_path: 图片文件夹路径
    :param prefix: 新文件名前缀（例如 "cat_"）
    :param start_num: 起始序号（默认从1开始）
    :param digits: 序号位数（默认4位，如0001）
    """
    try:
        # 检查文件夹是否存在
        if not os.path.exists(folder_path):
            raise FileNotFoundError(f"文件夹 {folder_path} 不存在")

        # 获取所有图片文件（按创建时间排序）
        files = [f for f in os.listdir(folder_path) if f.lower().endswith(('.jpg', '.png', '.jpeg'))]
        files.sort(key=lambda x: os.path.getctime(os.path.join(folder_path, x)))  # 按创建时间排序

        if not files:
            print("警告：文件夹中没有图片文件！")
            return

        # 开始重命名
        count = start_num
        for filename in files:
            # 获取文件扩展名（保留原始大小写）
            _, ext = os.path.splitext(filename)
            # 生成新文件名（例如 image_0001.jpg）
            new_name = f"{prefix}{count:0{digits}d}{ext}"
            # 重命名文件
            old_path = os.path.join(folder_path, filename)
            new_path = os.path.join(folder_path, new_name)
            os.rename(old_path, new_path)
            print(f"重命名: {filename} -> {new_name}")
            count += 1

        print(f"\n完成！共重命名 {len(files)} 个文件")

    except Exception as e:
        print(f"错误: {e}")

if __name__ == "__main__":
    # ！！！！！！！！！！ 在这里修改参数 ！！！！！！！！！！！！！
    FOLDER_PATH = "D:\pycharm\pycharmprojects\learn-pytorch\EffcientNet-b0\data\soybean\mosaic"  # 替换为你的图片文件夹路径
    PREFIX = "mosaic_"   # 新文件名前缀（例如 "dog_", "cat_"）
    START_NUM = 1      # 起始编号
    DIGITS = 4         # 序号位数（4位生成0001，3位生成001）

    # 执行重命名
    batch_rename_images(FOLDER_PATH, PREFIX, START_NUM, DIGITS)