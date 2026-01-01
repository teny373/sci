import numpy as np
import os
from osgeo import gdal
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker

file_dir = r"C:\Users\lenovo\Desktop\report\cj\fir/"
save_dir = r"C:\Users\lenovo\Desktop\report\cj\1_auto_z_filtered_v4"
os.makedirs(save_dir, exist_ok=True)

def generate_ordered_indices(data):
    ny, nx = data.shape
    ys, xs = np.where(~np.isnan(data))
    zs = data[ys, xs]

    cy, cx = ny // 2, nx // 2
    distances = np.sqrt((ys - cy)**2 + (xs - cx)**2)

    sort_idx = np.argsort(distances)
    xs_sorted = xs[sort_idx]
    ys_sorted = ys[sort_idx]
    zs_sorted = zs[sort_idx]

    return xs_sorted, ys_sorted, zs_sorted

def plot_ptc_super_sparse_cumulative(ax, xs_sorted, ys_sorted, zs_sorted, step, prev_indices=None):
    ny = np.max(ys_sorted) + 1
    nx = np.max(xs_sorted) + 1
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)

    # 每步新增比例
    add_fraction_map = {1: 0.001, 2: 0.001, 3: 0.005, 4: 0.01, 5: 0.02}
    add_fraction = add_fraction_map[step]
    n_add = max(1, int(len(xs_sorted) * add_fraction))

    if prev_indices is None:
        indices = [0]
    else:
        remaining = np.setdiff1d(np.arange(len(xs_sorted)), prev_indices)
        if len(remaining) == 0:
            indices = prev_indices
        else:
            indices_add = np.linspace(0, len(remaining)-1, n_add, dtype=int)
            indices = np.sort(np.concatenate([prev_indices, remaining[indices_add]]))

    xs_draw = xs_sorted[indices]
    ys_draw = ys_sorted[indices]
    zs_draw = zs_sorted[indices]

    # 去掉异常高值
    if step <= 3:
        zs_draw = np.where(zs_draw > 150, np.nan, zs_draw)
    elif step == 4:
        zs_draw = np.where(zs_draw > 240, np.nan, zs_draw)
    else:
        zs_draw = np.where(zs_draw > 230, np.nan, zs_draw)

    # Z轴缩放，前四步自动，最后一步固定 0-255
    if step < 5:
        zmax = max(3, np.nanmax(zs_draw))
        ax.set_zlim(0, zmax)
        # 设定整数刻度，不超过 5 个
        ax.zaxis.set_major_locator(mticker.MaxNLocator(nbins=5, integer=True))
    else:
        ax.set_zlim(0, 255)
        ax.zaxis.set_major_locator(mticker.MaxNLocator(nbins=6, integer=True))

    # 绘制绿色柱子
    for x0_pix, y0_pix, z0 in zip(xs_draw, ys_draw, zs_draw):
        if np.isnan(z0):
            continue
        x0 = x0_pix / (nx - 1)
        y0 = y0_pix / (ny - 1)
        ax.plot([x0, x0], [y0, y0], [0, z0], color="green", linewidth=1)

    return indices

# 遍历图像生成五步累积超稀疏 PTC
for img_name in os.listdir(file_dir):
    img_path = os.path.join(file_dir, img_name)
    dataset = gdal.Open(img_path)
    if dataset is None:
        print(f"无法读取 {img_name}")
        continue

    band = dataset.GetRasterBand(1)
    data = band.ReadAsArray()
    data = np.where(data > 255, np.nan, data)

    xs_sorted, ys_sorted, zs_sorted = generate_ordered_indices(data)
    prev_indices = None

    for step in range(1, 6):
        fig = plt.figure(figsize=(8, 8), dpi=200)
        ax = fig.add_subplot(111, projection="3d")
        prev_indices = plot_ptc_super_sparse_cumulative(ax, xs_sorted, ys_sorted, zs_sorted, step, prev_indices)
        ax.view_init(elev=75, azim=-120)
        plt.tight_layout()
        out_path = os.path.join(save_dir, f"{os.path.splitext(img_name)[0]}_step{step}.png")
        plt.savefig(out_path)
        plt.close(fig)

    print(f"{img_name} 的五步累积超稀疏 PTC 图已保存（Z轴自动缩放，最后一步固定0-255）")
