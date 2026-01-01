# -*- coding: utf-8 -*-
"""
Created on Thu Sep 28 20:07:41 2023

@author: Lenovo
"""

import numpy as np
import os
from osgeo import gdal
import matplotlib.pyplot as plt

file_dir = r"C:\Users\lenovo\Desktop\report\cj\fir/"
save_dir = r"C:\Users\lenovo\Desktop\report\cj\different angle"
os.makedirs(save_dir, exist_ok=True)

# 三个对比角度 + 当前角度
views = [
    ("current", 75, -120),   # 当前角度 (-120°, 75°)
    ("compare_90", 75, 90),  # 比较角度 (90°, 75°)
    ("compare_120", 75, 120) # 比较角度 (120°, 75°)
]


# 读取图像并提取信息
for img_name in os.listdir(file_dir):
    img_path = os.path.join(file_dir, img_name)
    dataset = gdal.Open(img_path)
    if dataset is None:
        print(f"无法读取 {img_name}")
        continue

    band = dataset.GetRasterBand(1)
    data = band.ReadAsArray()

    ny, nx = data.shape
    x = np.linspace(0, 1, nx)
    y = np.linspace(0, 1, ny)
    xv, yv = np.meshgrid(x, y)
    data = np.where(data > 255, np.nan, data)

    for view_name, elev, az in views:
        fig = plt.figure(figsize=(10, 10), dpi=200)
        ax = fig.add_subplot(111, projection='3d')
        ax.plot_surface(xv, yv, data, cmap="Greens", linewidth=0)

        # 设置观察视角
        ax.view_init(elev, az)


        # 保存文件
        save_name = f"{os.path.splitext(img_name)[0]}_{view_name}.png"
        plt.savefig(os.path.join(save_dir, save_name))
        plt.close(fig)

    print(f"{img_name} 已保存四个角度的对比图")
