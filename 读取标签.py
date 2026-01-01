import os
from collections import Counter

# 标签文件夹路径
label_dir = r"D:\ufv\2025 summer\sci\org.v1i.yolov12 - 副本\valid\labels"

# 类别名称（按你的 class_id 顺序）
class_names = ['fir', 'pine', 'spruce', 'trembling aspen']

# 创建一个计数器，用来统计每个 class_id 出现的次数
class_counter = Counter()

# 遍历标签文件夹中的所有 .txt 文件
for file_name in os.listdir(label_dir):
    if file_name.endswith(".txt"):
        file_path = os.path.join(label_dir, file_name)

        with open(file_path, 'r') as f:
            for line in f:
                if line.strip():  # 跳过空行
                    parts = line.strip().split()
                    class_id = int(parts[0])  # YOLO 标签格式的第一列是 class_id
                    class_counter[class_id] += 1

# 输出每类树的数量
print("每类树木的数量统计如下：")
for class_id, count in class_counter.items():
    class_name = class_names[class_id] if class_id < len(class_names) else f"class_{class_id}"
    print(f"{class_name}（ID: {class_id}）：{count} 棵")
