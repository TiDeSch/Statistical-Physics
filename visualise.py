import numpy as np
import pickle
import glob
import os
import re
from PIL import Image
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
import scienceplots
plt.style.use('science')
plt.rcParams.update({'font.size': 20})

"""
Visualise frames from either contiouum models, isolines/contours or experenemtal fields. 
"""
    
def add_scale_bar(ax, bar_length, label, location='upper right', color='black', pad=0.05):
    # axis limits
    x0, x1 = ax.get_xlim()
    y0, y1 = ax.get_ylim()
    width = x1 - x0
    height = y1 - y0

    if location == 'lower right':
        x_end = x1 - pad * width
        x_start = x_end - bar_length
        y_bar = y0 + pad * height
    elif location == 'lower left':
        x_start = x0 + pad * width
        x_end = x_start + bar_length
        y_bar = y0 + pad * height
    elif location == 'upper right':
        x_end = x1 - pad * width
        x_start = x_end - bar_length
        y_bar = y1 - 2*pad * height
    elif location == 'upper left':
        x_start = x0 + pad * width
        x_end = x_start + bar_length
        y_bar = y1 - 2*pad * height

    bar_thickness = height * 0.01 

    ax.add_patch(plt.Rectangle(
        (x_start, y_bar), bar_length, bar_thickness,
        color=color, transform=ax.transData, clip_on=False
    ))
    with plt.rc_context({'font.weight': 'bold', 'text.usetex': False}):
        ax.text(
            (x_start + x_end) / 2, y_bar + bar_thickness * 3,
            label,
            ha='center', va='bottom',
            color=color, fontsize=14,
            transform=ax.transData
        )

def make_label(name, Type=None):
    if Type == "ModelH":
        if any(char.isdigit() for char in name):
            custom_label = fr"$\zeta = {name}$"
        else:
            custom_label = name

    elif Type in ("AMB", "AMB+", "Kennedy"):
        name = re.sub(r'K[\d.]+', '', name).strip()
        greek_map = {
            "zeta": r"\zeta",
            "alpha": r"\alpha",
            "beta": r"\beta",
            "gamma": r"\gamma",
            "delta": r"\delta",
            "epsilon": r"\epsilon",
            "lambda": r"\lambda",
            "mu": r"\mu",
            "sigma": r"\sigma",
            "omega": r"\omega",
            "kappa": r"\kappa",
        }
        matches = re.findall(r'([a-zA-Z]+)([\d.]+)', name)
        if matches:
            parts = []
            for key, val in matches:
                latex_key = greek_map.get(key.lower(), key)
                val = val.rstrip('0').rstrip('.')
                parts.append(rf"{latex_key} = {val}")
            custom_label = "$" + ", ".join(parts) + "$"
        else:
            custom_label = name
            
    else:
        custom_label = name
    return custom_label

def plot_ModelH(names, Type):
    arrays = []
    for name in names:
        parent = f"{Type_dir}/Data_{name}"
        subfolders = sorted([f.path for f in os.scandir(parent) if f.is_dir()])
        files = sorted(glob.glob(f"{subfolders[0]}/*.npy"))
        arrays.append(np.load(files[-1]))

    # Settings -----
    n = len(names)
    vmax = max(np.max(np.abs(a)) for a in arrays)
    vmin = -vmax
    labels = [make_label(nm, Type) for nm in names]
    fs = plt.rcParams['font.size']

    panel_base_w = 6 # width per panel in inches
    fig_w = panel_base_w * n
    fig_h = 7
    fig = plt.figure(figsize=(fig_w, fig_h))

    left = 0.03
    right = 0.90
    cbar_gap = 0.008
    cbar_w = 0.011
    gap_between = 0.01
    total_w = right - left
    panel_w = (total_w - (n - 1) * gap_between) / n
    lefts = [left + i * (panel_w + gap_between) for i in range(n)]

    bottom = 0.05
    top = 0.82
    panel_h = top - bottom
    bar_h = 0.03
    bar_gap = 0.008
    lbl_gap = 0.006

    axes = []
    for lft in lefts:
        ax = fig.add_axes([lft, bottom, panel_w, panel_h])
        axes.append(ax)

    for ax, data in zip(axes, arrays):
        im = ax.imshow(data, origin='lower', cmap='RdBu_r', vmin=vmin, vmax=vmax)
        ax.set_axis_off()
        add_scale_bar(ax, bar_length=len_scale, label=f'{len_scale} su', location='lower left')
    fig.canvas.draw()

    p0 = axes[0].get_position()
    p_last = axes[-1].get_position()
    rendered_bottom  = p0.y0
    rendered_top = p0.y1
    rendered_left = p0.x0
    rendered_right = p_last.x1
    rendered_total_w = rendered_right - rendered_left
    # --------------

    cbar_left = rendered_right + cbar_gap
    cbar_ax = fig.add_axes([cbar_left, rendered_bottom, cbar_w, rendered_top - rendered_bottom])
    cbar = fig.colorbar(im, cax=cbar_ax, orientation='vertical')

    bar_bottom_pos = rendered_top + bar_gap
    bar_ax = fig.add_axes([rendered_left, bar_bottom_pos, rendered_total_w, bar_h])
    gradient = np.linspace(0, 1, 256).reshape(1, -1)
    bar_ax.imshow(gradient, aspect='auto', cmap='copper', origin='lower')
    bar_ax.set_yticks([])
    bar_ax.set_xticks([])
    for spine in bar_ax.spines.values():
        spine.set_linewidth(0.6)
    bar_ax.annotate('', xy=(0.97, 0.5), xytext=(0.03, 0.5), xycoords='axes fraction', textcoords='axes fraction',
        arrowprops=dict(arrowstyle='->', color='white', lw=1.8, mutation_scale=14),)

    lbl_bottom_pos = bar_bottom_pos + bar_h + lbl_gap
    lbl_h = 0.08
    lbl_ax = fig.add_axes([rendered_left, lbl_bottom_pos, rendered_total_w, lbl_h])
    lbl_ax.set_axis_off()

    p_positions = [ax.get_position() for ax in axes]

    for i, (text, pos) in enumerate(zip(labels, p_positions)):
        if i == 0:
            xpos = (pos.x0 - rendered_left) / rendered_total_w
            ha = 'left'
        elif i == n - 1:
            xpos = (pos.x1 - rendered_left) / rendered_total_w
            ha = 'right'
        else:
            xpos = ((pos.x0 + pos.x1) / 2 - rendered_left) / rendered_total_w
            ha = 'center'

        lbl_ax.text(xpos, 0.0, text,
                    transform=lbl_ax.transAxes,
                    ha=ha, va='bottom',
                    fontsize=fs, color='black')

    results_dir = Type_dir
    os.makedirs(results_dir, exist_ok=True)
    filename_pdf = f"{results_dir}/example_ModelH.pdf"
    plt.savefig(filename_pdf, dpi=150, bbox_inches='tight')
    print(f"Saved {filename_pdf}")
    return axes

def plot_Pressure(names, Type=None, scales=None):
    if scales is None:
        scales = {name: (1.0, 1.0) for name in names}

    data = []
    for name in names:
        parent = f"{Type_dir}/Data_{name}"
        subfolders = sorted([f.path for f in os.scandir(parent) if f.is_dir()])
        files = sorted(glob.glob(f"{subfolders[0]}/*.npy"))
        arr = np.load(files[-1])
        dX, dY = scales.get(name, (1.0, 1.0))
        arr = arr / 1000 # convert Pa to kPa
        data.append(arr)

        h, w = arr.shape[:2]
        dX, dY = scales.get(name, (1.0, 1.0))
        print(f"{name}: {(w-1)*dX:.1f} x {(h-1)*dY:.1f} µm")

    # Load tif images
    tif_imgs = []
    for name in names:
        parent = fr"\raw_{name}"
        subfolders = sorted([f.path for f in os.scandir(parent) if f.is_dir()])
        first_subfolder = subfolders[0]
        files = sorted(glob.glob(f"{first_subfolder}/*.tif"))
        tif_imgs.append(np.array(Image.open(files[0])))

    vmax = max(np.max(np.abs(d)) for d in data)
    print(vmax)

    fig, axes = plt.subplots(nrows=2, ncols=3, figsize=(18, 12), gridspec_kw={'wspace': 0.05, 'hspace': -0.25, 'height_ratios': [1, 1]})

    # Top row: pressure maps
    for ax, d, name in zip(axes[0], data, names):
        custom_label = make_label(name, Type)
        dX, dY = scales.get(name, (1.0, 1.0))

        h, w = d.shape
        X = np.linspace(0, w * dX, w)
        Y = np.linspace(0, h * dY, h)
        extent = [X.min() - dX/2, # left edge                  
                  X.max() + dX/2, # right edge  
                  Y.min() - dY/2, # bottom edge
                  Y.max() + dY/2 # top edge
                ]

        im = ax.imshow(d, origin='lower', vmin=-vmax, vmax=vmax,
                       cmap='RdBu_r', extent=extent)
        ax.set_title(f"{custom_label}")
        ax.set_axis_off()
        ax.contour(X, Y, d, levels=[0], colors='black', linewidths=0.5)
        add_scale_bar(ax, bar_length=len_scale, label=f'{len_scale} µm', location='lower left')

    # Bottom row: tif images 
    for ax, img in zip(axes[1], tif_imgs):
        ax.imshow(img, cmap='gray', origin='upper')
        ax.set_axis_off()
        add_scale_bar(ax, bar_length=len_scale, label=f'{len_scale} µm', location='lower left', color='white')

    # Single shared colorbar anchored to the rightmost top panel 
    fig.canvas.draw()
    p = axes[0, -1].get_position()
    cbar_ax = fig.add_axes([p.x1 + 0.01, p.y0, 0.01, p.y1 - p.y0])
    cbar = fig.colorbar(im, cax=cbar_ax, orientation='vertical')
    cbar.set_label('Pressure (kPa µm)', fontsize=14)
    cbar.ax.axhline(0, color='black', linewidth=1.5)

    filename_pdf = str(Type_dir) + f'/example_{Type}.pdf'
    plt.savefig(filename_pdf, dpi=150, bbox_inches='tight')

    return axes

def plot_visualise(data, name, Type=None):
    custom_label = make_label(name, Type)
    fig, ax = plt.subplots(figsize=(8, 6))
    
    if Type == "Kennedy":
        re, im = data[0], data[1]
        for i in range(re.shape[0]):
            ax.plot(re[i], im[i], lw=0.6)
        ax.set_title(f"Example Isolines: {custom_label}")
        ax.set_xlabel("Re(z)")
        ax.set_ylabel("Im(z)")

    elif Type == "GNS":
        for line in data:
            ax.plot(line[0], line[1], 'b-', lw=1.5)
        ax.set_title(f"Example GNS Isolines: {custom_label}")
        ax.set_axis_off()
        ax.set_aspect('equal')

    elif Type in ("AMB+", "AMB"):
        im = ax.imshow(data, origin='lower', cmap='RdBu_r')
        divider = make_axes_locatable(ax)
        cax = divider.append_axes("bottom", size="3%", pad=0.05)
        cbar = fig.colorbar(im, cax=cax, orientation='horizontal')
        cbar.ax.axvline(0, color='black', linewidth=1.5)
        cbar.ax.plot(0.5, 1.0, color='black', markersize=6, transform=cbar.ax.transAxes, clip_on=False)
        ax.set_title(f"Example {Type} field: {custom_label}")
        ax.set_axis_off()
        add_scale_bar(ax, bar_length=len_scale, label=f'{len_scale} su')

    else:
        im = ax.imshow(data, origin='lower', cmap='RdBu_r')
        fig.colorbar(im, ax=ax)
        ax.set_title(f"Example {Type} field: {custom_label}")
        ax.contour(data, levels=[0], colors='black', linewidths=0.5, origin='lower')
        ax.set_xlabel("X")
        ax.set_ylabel("Y")

    plt.grid(alpha=0.4)
    plt.tight_layout()
    return ax

def process_fields(name, Type):
    parent = f"{Type_dir}/Data_{name}"
    subfolders = sorted([f.path for f in os.scandir(parent) if f.is_dir()])

    first_subfolder = subfolders[0]
    files = sorted(glob.glob(f"{first_subfolder}/*.npy"))
    data = np.load(files[-1])

    ax = plot_visualise(data, name, Type)
    results_dir = f"{Type_dir}/Results_{name}"
    os.makedirs(results_dir, exist_ok=True)
    filename_pdf = str(results_dir) + f'/example_{name}.pdf'
    plt.savefig(filename_pdf, dpi=150, bbox_inches='tight')
    return ax

def process_isolines(name, Type):
    parent = f"{Type_dir}/Results_{name}"
    file = sorted(glob.glob(f"{parent}/SLE_Trace_{name}.npy"))
    lines = np.load(file[0])
    lines = lines[:, :5, :] 

    ax = plot_visualise(lines, name, Type)
    results_dir = parent
    filename_pdf = str(results_dir) + f'/example_{name}.pdf'
    plt.savefig(filename_pdf, dpi=150, bbox_inches='tight')
    return ax
  
def process_GNS(name, Type):
    parent = fr"\Data_{name}"
    files = sorted(glob.glob(f"{parent}/*.pkl"))
    
    with open(files[0], 'rb') as f:
        data = pickle.load(f)
    lines = data[1][0]

    ax = plot_visualise(lines, name, Type)
    results_dir = f"{Type_dir}/Results_{name}"
    os.makedirs(results_dir, exist_ok=True)
    filename_pdf = str(results_dir) + f'/example_{name}.pdf'
    plt.savefig(filename_pdf, dpi=150, bbox_inches='tight')
    return ax 

if __name__ == "__main__":
    Type = "ModelH"
    Directory = ['Passive', '-0.01', '-0.1', '-0.5']
    Type_dir = fr"\{Type}"

    len_scale = 500
    if Type == "ModelH":
        names_to_plot = Directory
        plot_ModelH(names_to_plot, Type)

    elif Type == "Pressure":
        insers_size = 50
        scales = {
            "Control_WT_15kPa": (5.1473, 5.1473),
            "WT_3kPa":          (8.0, 8.0),
            "EcadKO_15kPa":     (8.0, 8.0),
        }
        ax = plot_Pressure(Directory, Type, scales=scales)

    else:
        for name in Directory:
            if Type == "Kennedy":
                ax = process_isolines(name, Type)

            elif Type == "GNS":
                all_data = []
                ax = process_GNS(name, Type)

            else:
                ax = process_fields(name, Type)
    
    plt.show()