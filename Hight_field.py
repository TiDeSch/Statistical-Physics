import numpy as np
import matplotlib.pyplot as plt

"""
Generate real space height maps based on kappa values.
The isolines/contours will follow SLE{kappa}. 
"""

seed = 13
kappa = 2
power_law = - 8 / kappa
n_fields = 100
size = 512

def Gaussian_height_field(nx, ny, power_law, seed):
    rng = np.random.default_rng(seed)
    kx = np.fft.fftfreq(nx) * nx
    ky = np.fft.fftfreq(ny) * ny
    kxg, kyg = np.meshgrid(kx, ky, indexing="xy")
    k = np.sqrt(kxg**2 + kyg**2)
    k[0, 0] = 1
    amplitude = k ** (power_law / 2)
    phase = rng.normal(size=(ny, nx)) + 1j * rng.normal(size=(ny, nx))
    ft = amplitude * phase
    ft[0, 0] = 0
    field = np.fft.ifft2(ft).real
    field -= field.mean()
    field /= field.std()
    return field

# Field
nx = ny = size
x = np.linspace(-1, 1, nx)
y = np.linspace(-1, 1, ny)
X, Y = np.meshgrid(x, y, indexing="xy")

rng = np.random.default_rng(seed)

for i in range(n_fields):
    field = Gaussian_height_field(nx, ny, power_law, seed=int(rng.integers(10**9)))
    filename = f"height_field_{i:03d}.npz"
    # np.savez_compressed(filename, field=field, X=X, Y=Y)
    # print(f"Saved: {filename}")

# Hight map 
plt.figure(figsize=(10, 8))
filled_contours = plt.contourf(X, Y, field, levels=15, cmap='viridis', alpha=0.2)
plt.contour(X, Y, field, levels=[0], colors='red', linewidths=1)
plt.title('Height Isolines')
plt.xlabel('x', fontsize=14)
plt.ylabel('y', fontsize=14)
plt.colorbar(filled_contours, label='Height', alpha=0.7)

plt.tight_layout()
plt.show()