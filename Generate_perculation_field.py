import os
import numpy as np
import matplotlib.pyplot as plt 

"""
Script to generate a perculation field.
"""

# Generate percolation fields
L = 400
n_fields = 100
prob = 0.41

save_dir = f""
os.makedirs(save_dir, exist_ok=True)

for i in range(n_fields):
    field = np.random.rand(L, L) < prob
    file_path = os.path.join(save_dir, f"perculation_field{i}.npy")
    np.save(file_path, field)

i_field = field.astype(float) * 2 - 1

plt.figure(figsize=(8, 6))
mesh = plt.pcolormesh(i_field, 
                    shading='auto', 
                    cmap='seismic', 
                    vmin=-1, 
                    vmax=1)
plt.colorbar(mesh)
plt.xlabel('X')
plt.ylabel('Y')
plt.title(f"Perculation Field")
plt.contour(i_field, 
            levels=[0], 
            colors='white', 
            linewidths=0.5
            )
plt.show()