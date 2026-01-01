import os

def rename_files(folder_path):
    # 获取文件列表，并按名称中的数字排序
    files = sorted(os.listdir(folder_path), key=lambda x: int(''.join(filter(str.isdigit, x))) if any(c.isdigit() for c in x) else float('inf'))

    for index, file_name in enumerate(files, start=105):
        old_path = os.path.join(folder_path, file_name)
        if os.path.isfile(old_path):
            # 获取文件扩展名
            ext = os.path.splitext(file_name)[1]
            # 生成新文件名（如 0001.jpg, 0002.png）
            new_name = f"{index:04d}{ext}"
            new_path = os.path.join(folder_path, new_name)
            os.rename(old_path, new_path)
            print(f"Renamed: {file_name} -> {new_name}")

# 设置要重命名的文件夹路径
folder = r"C:\Users\lenovo\Desktop\j\pine"  # 修改为你的文件夹路径
rename_files(folder)
