import numpy as np
import os
from osgeo import gdal
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker

file_dir = r"C:\Users\lenovo\Desktop\report\cj\fir/"
save_dir = r"C:\Users\lenovo\Desktop\report\cj\Nadir_dual_views"
os.makedirs(save_dir, exist_ok=True)

def generate_nadir_indices(data):
    ny, nx = data.shape
    ys, xs = np.where(~np.isnan(data))
    zs = data[ys, xs]
    cy, cx = ny // 2, nx // 2
    distances = np.sqrt((ys - cy)**2 + (xs - cx)**2)
    sort_idx = np.argsort(distances)
    return xs[sort_idx], ys[sort_idx], zs[sort_idx]

def select_indices(xs_sorted, zs_sorted, prev_indices, step):
    # 第二步固定 6 个点
    if step == 2:
        n_add = 6
    else:
        add_fraction_map = {1:1, 3:0.0015, 4:0.007, 5:0.015}
        n_add = max(1,int(len(xs_sorted)*add_fraction_map.get(step,0.01)))

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
            indices = np.sort(np.concatenate([prev_indices, selected]))
    return indices

def plot_nadir_dual_views(xs_sorted, ys_sorted, zs_sorted, indices, step):
    ny, nx = np.max(ys_sorted)+1, np.max(xs_sorted)+1
    xs_draw = xs_sorted[indices]/(nx-1)
    ys_draw = ys_sorted[indices]/(ny-1)
    zs_draw = zs_sorted[indices]

    # 去掉异常值
    if step < 5:
        zs_draw = np.where(zs_draw > 150, np.nan, zs_draw)
    else:
        zs_draw = np.where(zs_draw > 230, np.nan, zs_draw)

    # Z轴缩放
    if step < 5:
        zmax = max(3, np.nanmax(zs_draw))
        zlim = (0, zmax)
    else:
        zlim = (0, 255)

    # --- 点云图 ---
    fig1 = plt.figure(figsize=(10,8), dpi=200)
    ax1 = fig1.add_subplot(111, projection='3d')
    ax1.set_xlim(0,1)
    ax1.set_ylim(0,1)
    ax1.set_zlim(*zlim)
    ax1.zaxis.set_major_locator(mticker.MaxNLocator(nbins=5, integer=True))
    sc = ax1.scatter(xs_draw, ys_draw, zs_draw, c=zs_draw, s=50, cmap='gray_r', vmin=zlim[0], vmax=zlim[1])
    cbar = plt.colorbar(sc, ax=ax1, shrink=0.6, pad=0.1)
    cbar.set_label(f"Height (0-{int(zlim[1])})")
    ax1.view_init(elev=75, azim=-120)
    plt.tight_layout()

    # --- 柱状图 ---
    fig2 = plt.figure(figsize=(8,8), dpi=200)
    ax2 = fig2.add_subplot(111, projection='3d')
    ax2.set_xlim(0,1)
    ax2.set_ylim(0,1)
    ax2.set_zlim(*zlim)
    ax2.zaxis.set_major_locator(mticker.MaxNLocator(nbins=5, integer=True))
    for x0, y0, z0 in zip(xs_draw, ys_draw, zs_draw):
        if np.isnan(z0):
            continue
        ax2.plot([x0, x0], [y0, y0], [0, z0], color="green", linewidth=1)
    ax2.view_init(elev=75, azim=-120)
    plt.tight_layout()

    return fig1, fig2

# 遍历图像
for img_name in os.listdir(file_dir):
    img_path = os.path.join(file_dir, img_name)
    dataset = gdal.Open(img_path)
    if dataset is None:
        print(f"无法读取 {img_name}")
        continue

    band = dataset.GetRasterBand(1)
    data = band.ReadAsArray()
    data = np.where(data>255,np.nan,data)
    data = np.where(data<=0,np.nan,data)

    xs_sorted, ys_sorted, zs_sorted = generate_nadir_indices(data)
    prev_indices = None

    for step in range(1,6):
        indices = select_indices(xs_sorted, zs_sorted, prev_indices, step)
        prev_indices = indices
        fig1, fig2 = plot_nadir_dual_views(xs_sorted, ys_sorted, zs_sorted, indices, step)

        out_path1 = os.path.join(save_dir, f"{os.path.splitext(img_name)[0]}_pointcloud_step{step}.png")
        fig1.savefig(out_path1)
        plt.close(fig1)

        out_path2 = os.path.join(save_dir, f"{os.path.splitext(img_name)[0]}_pillar_step{step}.png")
        fig2.savefig(out_path2)
        plt.close(fig2)

    print(f"{img_name} 的五步 Nadir 点云和柱状图已保存")
