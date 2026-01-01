import os
import yaml
import shutil

# 设置你要提取的目标类别
target_class = "pine"

# 读取 data.yaml 获取类名
with open("data.yaml", "r") as f:
    data_cfg = yaml.safe_load(f)

names = data_cfg["names"]  # ['larch', 'pine', 'poplar', 'snag', 'spruce']

# 定义训练集路径
train_path = data_cfg["train"].replace("/images", "")
label_dir = os.path.join(train_path, "labels")
img_dir = os.path.join(train_path, "images")

# 目标文件夹路径，保存提取的图片
output_dir = r"C:\Users\lenovo\Desktop\pine\pine230"  # 你可以根据需求修改此路径

# 如果目标文件夹不存在，则创建它
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

# 存储匹配的图片路径
matched_images = []

for label_file in os.listdir(label_dir):
    if not label_file.endswith(".txt"):
        continue

    label_path = os.path.join(label_dir, label_file)
    with open(label_path, "r") as f:
        lines = f.readlines()

    # 提取所有 class_id
    class_ids = {int(line.strip().split()[0]) for line in lines if line.strip()}
    class_names = [names[i] for i in class_ids]

    # 如果目标类别在其中，记录图片路径
    if target_class in class_names:
        image_name = label_file.replace(".txt", ".jpg")  # 改成 .png 如果你图片是 PNG
        image_path = os.path.join(img_dir, image_name)

        # 将匹配的图片复制到目标文件夹
        shutil.copy(image_path, os.path.join(output_dir, image_name))
        matched_images.append(image_path)

# 输出结果
print(f"\n✅ 共找到 {len(matched_images)} 张图片包含 '{target_class}'，并已保存到 '{output_dir}'：")
for img in matched_images:
    print(img)
