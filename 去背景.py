import cv2
import numpy as np
import os

# 输入文件夹路径
file_dir = r"D:\ufv\2025 winnter\project\jzv2\pine"

# 输出文件夹路径（确保文件夹存在）
output_dir = r"D:\ufv\2025 winnter\project\PTCV2\tre\qu"
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

# 遍历文件夹中的所有图片
for img_name in os.listdir(file_dir):
    # 检查文件是否是图像文件（例如 jpg, jpeg, png）
    if img_name.endswith(('.jpg', '.jpeg', '.png')):
        # 构建图像文件的完整路径
        image_path = os.path.join(file_dir, img_name)

        # 读取图像
        image = cv2.imread(image_path)

        # 检查图像是否成功加载
        if image is None:
            print(f"Failed to load image: {img_name}")
            continue

        # 将图像转换为灰度图
        gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        # 创建一个掩码，忽略所有大于120的像素
        mask = gray_image > 230

        # 设置超过230的像素为0（黑色）
        gray_image[mask] = 0

        # 保存处理后的图片
        output_path = os.path.join(output_dir, img_name)
        cv2.imwrite(output_path, gray_image)

        print(f"Processed image saved: {output_path}")

print("所有图片处理完成！")