import cv2
import numpy as np
import os

# 输入文件夹路径
file_dir = r"D:\ufv\2025 winnter\project\jzv2\spruce"
# 输出文件夹路径（确保文件夹存在）
output_dir = r"D:\ufv\2025 winnter\project\jzv2\spruce1"
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

# 遍历文件夹中的所有图片
for img_name in os.listdir(file_dir):
    if img_name.endswith(('.jpg', '.jpeg', '.png')):
        image_path = os.path.join(file_dir, img_name)

        image = cv2.imread(image_path)
        if image is None:
            print(f"Failed to load image: {img_name}")
            continue

        # 创建掩膜，检测白色背景（白色范围阈值可调）
        # 白色阈值范围：大于 (240, 240, 240) 认为是白色
        lower_white = np.array([240, 240, 240], dtype=np.uint8)
        upper_white = np.array([255, 255, 255], dtype=np.uint8)

        mask = cv2.inRange(image, lower_white, upper_white)  # 白色区域为255，其它为0

        # 创建全黑背景
        black_bg = np.zeros_like(image)

        # 将白色区域替换为黑色，其他区域保留原图
        image_with_black_bg = np.where(mask[:, :, np.newaxis] == 255, black_bg, image)

        output_path = os.path.join(output_dir, img_name)
        cv2.imwrite(output_path, image_with_black_bg)
        print(f"Processed image saved: {output_path}")

print("所有图片处理完成！")
