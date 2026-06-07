import pickle
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import glob
import math
import os
import numpy as np
import re

"""
Script to animate phase fields (.npy and .pkl files). 
"""

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

def load_pkl_datasets(base_dir, sets):
    all_data = []
    for dataset in sets:
        input_dir = os.path.join(base_dir, f'Data_{dataset}', '*.pkl')
        pkl_files = glob.glob(input_dir)

        if not pkl_files:
            print(f"No .pkl files found for {dataset}, skipping.")
            continue

        with open(pkl_files[0], 'rb') as f:
            data = pickle.load(f)

        all_x, all_y = [], []
        for timestep_lines in data[1]:
            for line in timestep_lines:
                all_x.extend(line[0])
                all_y.extend(line[1])

        x_min, x_max = min(all_x), max(all_x)
        y_min, y_max = min(all_y), max(all_y)
        x_pad = (x_max - x_min) * 0.05
        y_pad = (y_max - y_min) * 0.05

        all_data.append({
            'name':       dataset,
            'type':       'pkl',
            'time_data':  data[0],
            'lines_data': data[1],
            'file':       pkl_files[0],
            'x_lim':      (x_min - x_pad, x_max + x_pad),
            'y_lim':      (y_min - y_pad, y_max + y_pad),
        })
        print(f"Loaded {dataset}: {pkl_files[0]}")

    return all_data

def load_npy_datasets(base_dir, sets):
    all_data = []
    for dataset in sets:
        data_dir = os.path.join(base_dir, f'Data_{dataset}')

        # Find all subfolders, sorted naturally (phi1, phi2, phi10, ...)
        subfolders = sorted(
            [d for d in glob.glob(os.path.join(data_dir, '*')) if os.path.isdir(d)],
            key=lambda p: [int(t) if t.isdigit() else t for t in re.split(r'(\d+)', os.path.basename(p))]
        )

        if not subfolders:
            print(f"No subfolders found in {data_dir}, skipping.")
            continue

        all_frames = []
        segments = [] 

        for subfolder in subfolders:
            npy_files = sorted(
                glob.glob(os.path.join(subfolder, '*.npy')),
                key=lambda p: int(re.search(r'\d+', os.path.basename(p)).group())
            )
            if not npy_files:
                print(f" No .npy files in {subfolder}, skipping.")
                continue

            start = len(all_frames)
            frames = [np.load(f) for f in npy_files]
            all_frames.extend(frames)
            segments.append({
                'name':      os.path.basename(subfolder),
                'short_name': re.search(r'(phi\d+)', os.path.basename(subfolder)).group(1)
                            if re.search(r'(phi\d+)', os.path.basename(subfolder))
                            else os.path.basename(subfolder).split('_')[0],
                'start': start,
                'end':   len(all_frames) - 1,
            })
            print(f"  {dataset}/{os.path.basename(subfolder)}: {len(frames)} frames")

        if not all_frames:
            print(f"No frames loaded for {dataset}, skipping.")
            continue

        stacked = np.stack(all_frames)
        all_data.append({
            'name':     dataset,
            'type':     'npy',
            'frames':   all_frames,
            'segments': segments,
            'vmin':     float(np.nanmin(stacked)),
            'vmax':     float(np.nanmax(stacked)),
        })
        print(f"Loaded {dataset}: {len(all_frames)} total frames across {len(segments)} folders")
    return all_data

def animate_datasets(all_data, title='', Type=None, interval=50, save_path=None, fps=10, dpi=150):
    num_frames = max(
        len(d['time_data']) if d['type'] == 'pkl' else len(d['frames'])
        for d in all_data
    )
    num_sets = len(all_data)
    ncols = min(3, num_sets)
    nrows = math.ceil(num_sets / ncols)

    fig, axes = plt.subplots(nrows, ncols, figsize=(6 * ncols, 5 * nrows))
    axes = np.array(axes).flatten() if num_sets > 1 else [axes]
    for ax in axes[num_sets:]:
        ax.set_visible(False)
   
    fig.suptitle(title, fontsize=16, weight='bold', y=0.98)
    line_objects = [[] for _ in range(num_sets)]
    im_objects   = [None] * num_sets
    # Per-subplot subtitle text objects (shows current field name for npy)
    sub_titles   = [None] * num_sets
    time_texts   = [None] * num_sets

    for idx, (ax, dataset) in enumerate(zip(axes, all_data)):
        if dataset['type'] == 'pkl':
            ax.set_xlim(dataset['x_lim'])
            ax.set_ylim(dataset['y_lim'])
            ax.set_aspect('equal')
            ax.grid(True, alpha=0.3)
        else:
            im = ax.imshow(
                dataset['frames'][0],
                origin='lower',
                cmap='RdBu_r',
                vmin=dataset['vmin'],
                vmax=dataset['vmax'],
                interpolation='nearest',
            )
            fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
            im_objects[idx] = im
            # Subtitle showing current field name, updated each frame
            sub_titles[idx] = ax.set_title(
                make_label(dataset['name'], Type),
                fontsize=12, weight='bold'
            )

        if dataset['type'] == 'pkl':
            ax.set_title(dataset['name'], fontsize=12, weight='bold')
        ax.set_xlabel('X', fontsize=10)
        ax.set_ylabel('Y', fontsize=10)
        time_texts[idx] = ax.text(
            0.98, 0.02, 'Frame 1',
            transform=ax.transAxes,
            ha='right', va='bottom',
            fontsize=9, color='black',
            bbox=dict(boxstyle='round,pad=0.2', fc='white', alpha=0.6),
        )

    def _field_name_at(dataset, frame):
        """Return the field/segment name active at a given frame index."""
        for seg in dataset['segments']:
            if seg['start'] <= frame <= seg['end']:
                return seg['name']
        return dataset['segments'][-1]['name']

    def init():
        for idx, dataset in enumerate(all_data):
            if dataset['type'] == 'npy':
                im_objects[idx].set_data(dataset['frames'][0])
            else:
                for lo in line_objects[idx]:
                    lo.set_data([], [])
        return (
            [lo for ll in line_objects for lo in ll]
            + [im for im in im_objects if im]
            + [st for st in sub_titles if st]    # <-- add
            + [tt for tt in time_texts if tt]    # <-- add
        )

    def animate(frame):
        for idx, (ax, dataset) in enumerate(zip(axes, all_data)):
            if dataset['type'] == 'pkl':
                for lo in line_objects[idx]:
                    lo.remove()
                line_objects[idx].clear()
                f_idx = min(frame, len(dataset['lines_data']) - 1)
                for line in dataset['lines_data'][f_idx]:
                    lo, = ax.plot(line[0], line[1], 'b-', linewidth=1.5)
                    line_objects[idx].append(lo)
                time_texts[idx].set_text(f'Frame {f_idx + 1}/{len(dataset["lines_data"])}')
            else:
                f_idx = min(frame, len(dataset['frames']) - 1)
                im_objects[idx].set_data(dataset['frames'][f_idx])
                # Update subtitle to show current field name
                field = _field_name_at(dataset, f_idx)
                sub_titles[idx].set_text(f"{make_label(dataset['name'], Type)}  –  {field}")
                time_texts[idx].set_text(f'Frame {f_idx + 1}/{len(dataset["frames"])}')

        fig.suptitle(f'{title}  –  frame {frame + 1}/{num_frames}', fontsize=16, weight='bold', y=0.98)
        return (
            [lo for ll in line_objects for lo in ll]
            + [im for im in im_objects if im]
            + [st for st in sub_titles if st]
            + [tt for tt in time_texts if tt]
        )

    anim = animation.FuncAnimation(
        fig, animate, init_func=init,
        frames=num_frames, interval=interval,
        blit=True, repeat=True,
    )

    if save_path:
        anim.save(save_path, writer='ffmpeg', fps=fps, dpi=dpi)
        print(f"Saved animation to {save_path}")

    plt.tight_layout()
    if SHOW_plot:
        plt.show()
    else:
        plt.close()
    return anim

if __name__ == '__main__':
    Type = "ModelH"
    activity = ['Passive', '-0.01', '-0.05', '-0.1', '-0.2', '-0.5', '-1.0']

    SAVE = True
    SHOW_plot = True

    if Type == "Rainer":
        base_dir = fr'  '
        all_data = load_pkl_datasets(base_dir, activity)
    else:
        base_dir = fr'  '
        all_data = load_npy_datasets(base_dir, activity)

    if SAVE:
        save_path = os.path.join(base_dir, f'{Type}_animation.mp4')
    else:
        save_path = None

    animate_datasets(
        all_data,
        title=Type,
        Type=Type,   
        interval=50,
        save_path=save_path,
        fps=10,
    )

    if SHOW_plot:
        plt.show()
    else:
        plt.close()