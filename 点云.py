import numpy as np
import os
from osgeo import gdal
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker

# 输入输出目录
file_dir = r"C:\Users\lenovo\Desktop\report\cj\fir/"
save_dir = r"C:\Users\lenovo\Desktop\report\cj\Nadir_sparse_tree_clear"
os.makedirs(save_dir, exist_ok=True)

# 生成 Nadir 索引（中心向外）
def generate_nadir_indices(data):
    ny, nx = data.shape
    ys, xs = np.where(~np.isnan(data))
    zs = data[ys, xs]
    cy, cx = ny // 2, nx // 2
    distances = np.sqrt((ys - cy)**2 + (xs - cx)**2)
    sort_idx = np.argsort(distances)
    return xs[sort_idx], ys[sort_idx], zs[sort_idx]

# 绘制灰度点云
def plot_nadir_points_gray(ax, xs_sorted, ys_sorted, zs_sorted, step, prev_indices=None):
    ny, nx = np.max(ys_sorted)+1, np.max(xs_sorted)+1
    ax.set_xlim(0,1)
    ax.set_ylim(0,1)

    # 去掉背景和异常高值
    if step < 5:
        valid_mask = zs_sorted <= 150  # 前四步去掉 >150
    else:
        valid_mask = zs_sorted <= 230  # 第五步去掉 >230
    xs_sorted = xs_sorted[valid_mask]
    ys_sorted = ys_sorted[valid_mask]
    zs_sorted = zs_sorted[valid_mask]

    # 点选择数量
    if step == 2:
        n_add = 5  # 第二步 3-5 个点
    else:
        add_fraction_map = {1:1, 3:0.0015, 4:0.007, 5:0.015}  # step1直接1个点
        add_fraction = add_fraction_map.get(step,0.01)
        n_add = max(1,int(len(xs_sorted)*add_fraction))

    # 累计选择
    if prev_indices is None:
        indices = [0]
    else:
        remaining = np.setdiff1d(np.arange(len(xs_sorted)), prev_indices)
        if len(remaining)==0:
            indices = prev_indices
        else:
            if step==2:
                selected = np.random.choice(remaining, min(n_add,len(remaining)), replace=False)
            else:
                selected = np.linspace(0,len(remaining)-1,n_add,dtype=int)
            indices = np.sort(np.concatenate([prev_indices, remaining[selected]]))

    xs_draw = xs_sorted[indices]/(nx-1)
    ys_draw = ys_sorted[indices]/(ny-1)
    zs_draw = zs_sorted[indices]

    # Z轴缩放和 colorbar
    if step < 5:
        zmax = max(3, np.nanmax(zs_draw))  # 前四步自动缩小
        ax.set_zlim(0, zmax)
        vmin, vmax = 0, zmax
    else:
        ax.set_zlim(0, 255)
        vmin, vmax = 0, 255

    # Z轴整数刻度，减少重叠
    ax.zaxis.set_major_locator(mticker.MaxNLocator(nbins=5, integer=True))

    # 绘制灰度点
    sc = ax.scatter(xs_draw, ys_draw, zs_draw, c=zs_draw, s=50, cmap='gray_r', vmin=vmin, vmax=vmax)

    # colorbar
    cbar = plt.colorbar(sc, ax=ax, shrink=0.6, pad=0.1)
    cbar.set_label(f"Height (0-{int(vmax)})")
    return indices

# 遍历生成五步 Nadir 灰度点云
for img_name in os.listdir(file_dir):
    img_path = os.path.join(file_dir,img_name)
    dataset = gdal.Open(img_path)
    if dataset is None:
        print(f"无法读取 {img_name}")
        continue

    band = dataset.GetRasterBand(1)
    data = band.ReadAsArray()
    data = np.where(data <= 0, np.nan, data)  # 去掉背景

    xs_sorted,ys_sorted,zs_sorted = generate_nadir_indices(data)
    prev_indices = None

    for step in range(1,6):
        fig = plt.figure(figsize=(8,8),dpi=200)
        ax = fig.add_subplot(111,projection='3d')
        prev_indices = plot_nadir_points_gray(ax,xs_sorted,ys_sorted,zs_sorted,step,prev_indices)
        ax.view_init(elev=75, azim=-120)
        plt.tight_layout()
        out_path = os.path.join(save_dir,f"{os.path.splitext(img_name)[0]}_nadir_gray_step{step}.png")
        plt.savefig(out_path)
        plt.close(fig)

    print(f"{img_name} 的五步稀疏 Nadir 灰度点云图已保存（清晰版）")
