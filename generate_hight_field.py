import matplotlib.pyplot as plt
import numpy as np
import scipy.spatial as spatial
import scipy.ndimage as nd
import os

"""
Script to generate call monolayer using the Voronoi cell layer model (SPV-model).
"""

def generate_voronoi_cell_layer(n, m, n_cells):
    # 1. Generate random cell centers
    points = np.column_stack([
        np.random.uniform(0, n, n_cells),
        np.random.uniform(0, m, n_cells)
    ])

    # Voronoi tessellation
    vor = spatial.Voronoi(points)

    # Pixel grid
    x = np.arange(n)
    y = np.arange(m)
    xx, yy = np.meshgrid(x, y)

    pixel_coords = np.column_stack([xx.flatten(), yy.flatten()])

    # Map each pixel to nearest seed (cell)
    tree = spatial.KDTree(points)
    _, labels = tree.query(pixel_coords)
    labels = labels.reshape((m, n))

    # 2. Heightmap from bumps per cell
    heightmap = np.zeros((m, n), dtype=np.float64)

    for i, (cx, cy) in enumerate(points):
        # Per-cell properties
        height = np.random.uniform(0.4, 1.2)
        radius = np.random.uniform(8, 15)

        mask = (labels == i)

        # Gaussian bump inside each Voronoi cell
        dist_sq = (xx - cx)**2 + (yy - cy)**2
        bump = height * np.exp(-dist_sq / (2 * radius**2))
        heightmap += bump * mask

    # 3. Slight smoothness + noise (biological realism)
    heightmap = nd.gaussian_filter(heightmap, sigma=1.2)

    noise = np.random.normal(0, 0.03, (m, n))
    heightmap += noise

    return heightmap

def generate_dataset(num_maps, n, m, n_cells, output_dir=""):
    os.makedirs(output_dir, exist_ok=True)

    for i in range(num_maps):
        hmap = generate_voronoi_cell_layer(n, m, n_cells)
        fname = os.path.join(output_dir, f"heightmap_{i:03d}.npy")
        np.save(fname, hmap)
        print(f"Saved {fname}")

if __name__ == "__main__":
    save_dir = f""
    num_maps = 500
    n = 512
    m = 512
    n_cells = 500
    # generate_dataset(num_maps=num_maps, n=n, m=m, n_cells=n_cells, output_dir=save_dir)

example = generate_voronoi_cell_layer(n, m, n_cells)
plt.figure(figsize=(8, 6))
mesh = plt.pcolormesh(example, shading='auto', cmap='viridis')
plt.colorbar(mesh, label="Height")
plt.xlabel("X")
plt.ylabel("Y")
plt.title("Height Field")
plt.show()