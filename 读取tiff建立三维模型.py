# -*- coding: utf-8 -*-
"""
Created on Thu Sep 28 20:07:41 2023

@author: Lenovo
"""

import numpy as np #导入NumPy库，用于进行数值运算（如矩阵操作、数组生成等）
import os.path #导入os.path模块，用于处理文件和路径操作
from osgeo import gdal# 从OSGeo库中导入gdal模块，用于读取遥感影像数据
import matplotlib.pyplot as plt# 导入Matplotlib用于绘图，包括3D绘图

file_dir =  r"C:\Users\lenovo\Desktop\report\cj\fir/"
save_dir =  r"D:\ufv\2025 winnter\project\PTCV2\tre"

#读取图像并提取信息
for img_name in os.listdir(file_dir): # 遍历 file_dir 文件夹中所有图像文件
    img_path = file_dir +img_name# 获取每张图像的完整路径
    dataset = gdal.Open(img_path)# 使用GDAL打开图像文件，返回dataset对象

    #宽度和高度定义了图像的像素维度；
    #波段数代表该遥感图像是几波段的；
    #仿射矩阵和投影信息用于后续空间定位，但在本项目中未使用，仅保留。
    img_XSize = dataset.RasterXSize #列数
    img_YSize = dataset.RasterYSize #行数
    img_bands =dataset.RasterCount #波段数
    img_geotrans = dataset.GetGeoTransform() #仿射矩阵
    img_proj =dataset.GetProjection() #地图投影信息

#读取一个波段，其参数为波段的索引号，波段索引号从1开始（绿色）
    band = dataset.GetRasterBand(1)
    #将读取的数据转化为数组形式
    data = band.ReadAsArray()
    #根据图像的大小生成归一化的坐标范围
    ny,nx = data.shape # 获取图像行列数（高度和宽度）
    x = np.linspace(0,1,nx)# 在x方向生成均匀分布的坐标值（0到1）
    y = np.linspace(0,1,ny)# 在y方向生成均匀分布的坐标值（0到1）
    xv,yv = np.meshgrid(x,y) #进行网格化
    data = np.where(data>255,np.nan,data)
    #创建一个 10x10 英寸，分辨率为 200 的图像画布；
    # 创建一个三维坐标轴 ax；
    # 使用 plot_surface() 将图像的数据绘制成一个 3D 表面，其中绿色表示像素值高低。
    fig = plt.figure(figsize=(10,10),dpi=200)#设置输出的图像大小和像素
    ax =fig.add_subplot(111,projection='3d')#创建三维坐标轴
    data3d = ax.plot_surface(xv,yv,data,cmap="Greens",linewidth=0)#画PTC
#设置观察视角
    az,elev =-120,75
    ax.view_init(elev,az)
    plt.axis()
    plt.savefig(os.path.join(save_dir,img_name))