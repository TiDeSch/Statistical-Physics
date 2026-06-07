import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import time
import os
from numba import njit, prange
from multiprocessing import Pool, cpu_count
import pyfftw.interfaces.numpy_fft as fft
import pyfftw
pyfftw.config.NUM_THREADS = 1
import pyfftw.interfaces.cache
pyfftw.interfaces.cache.enable()
from numba import set_num_threads
# set_num_threads(1)
np.random.seed(13)

"""
Conserved active models. All parallelized.

    Active Model B+ (Cahn Hillard)
    Active Model H (Cahn Hillard) 
    Potts Model B

    + a noise model
"""

####### Functions #############
@njit(parallel=True, fastmath=True)
def laplacian(arr, dx):
    nx, ny = arr.shape
    out = np.empty_like(arr)
    for idx in prange(nx * ny):
        i = idx // ny
        j = idx % ny

        ip = (i + 1) % nx
        im = (i - 1) % nx
        jp = (j + 1) % ny
        jm = (j - 1) % ny

        out[i, j] = (
            arr[ip, j] +
            arr[im, j] +
            arr[i, jp] +
            arr[i, jm] -
            4.0 * arr[i, j]
        ) / (dx * dx)
    return out

@njit(parallel=True, fastmath=True)
def divergence(vx, vy, dx):
    nx, ny = vx.shape
    div = np.empty_like(vx)
    for idx in prange(nx * ny):
        i = idx // ny
        j = idx % ny

        ip = (i+1)%nx
        im = (i-1)%nx
        jp = (j+1)%ny
        jm = (j-1)%ny

        div[i,j] = (
            vx[ip,j] - vx[im,j] +
            vy[i,jp] - vy[i,jm]
        )/(2*dx)
    return div

@njit(parallel=True, fastmath=True)
def gradient(arr, dx):
    nx, ny = arr.shape
    gx = np.empty_like(arr)
    gy = np.empty_like(arr)
    for idx in prange(nx * ny):
        i = idx // ny
        j = idx % ny

        ip = (i+1)%nx
        im = (i-1)%nx
        jp = (j+1)%ny
        jm = (j-1)%ny

        gx[i,j] = (arr[ip,j] - arr[im,j])/(2*dx)
        gy[i,j] = (arr[i,jp] - arr[i,jm])/(2*dx)
    return gx, gy

@njit(parallel=True, fastmath=True)
def grad_squared(phi, dx):
    nx, ny = phi.shape
    out = np.empty_like(phi)
    for idx in prange(nx * ny):
        i = idx // ny
        j = idx % ny

        ip = (i+1) % nx
        im = (i-1) % nx
        jp = (j+1) % ny
        jm = (j-1) % ny

        dxp = (phi[ip, j] - phi[im, j]) / (2*dx)
        dyp = (phi[i, jp] - phi[i, jm]) / (2*dx)

        out[i, j] = dxp*dxp + dyp*dyp
    return out

@njit(parallel=True, fastmath=True)
def compute_force(phi, mu, dx):
    nx, ny = phi.shape
    fx = np.empty_like(phi)
    fy = np.empty_like(phi)
    for idx in prange(nx * ny):
        i = idx // ny
        j = idx % ny

        ip = (i+1)%nx
        im = (i-1)%nx
        jp = (j+1)%ny
        jm = (j-1)%ny

        dmx = (mu[ip,j] - mu[im,j])/(2*dx)
        dmy = (mu[i,jp] - mu[i,jm])/(2*dx)

        fx[i,j] = -phi[i,j] * dmx
        fy[i,j] = -phi[i,j] * dmy
    return fx, fy

@njit(fastmath=True)
def chemical_potential(phi, a, b, K, dx):
    mu = a * phi + b * phi * phi * phi - K * laplacian(phi, dx)
    return mu

def solve_stokes(fx, fy, kx, ky, k2, eta):
    fx_hat = fft.fft2(fx)
    fy_hat = fft.fft2(fy)
    k_dot_f = kx*fx_hat + ky*fy_hat

    vx_hat = (fx_hat - kx * k_dot_f / k2) / (eta * k2)
    vy_hat = (fy_hat - ky * k_dot_f / k2) / (eta * k2)

    vx_hat[0,0] = 0
    vy_hat[0,0] = 0

    vx = np.real(fft.ifft2(vx_hat))
    vy = np.real(fft.ifft2(vy_hat))
    return vx, vy

@njit(fastmath=True)
def compute_modelH_forces(phi, a, b, K, dx, zeta):
    mu = chemical_potential(phi, a, b, K, dx)
    fx, fy = compute_force(phi, mu, dx)

    grad_phi_x, grad_phi_y = gradient(phi, dx)

    sigma_xx = -zeta * grad_phi_x * grad_phi_x
    sigma_xy = -zeta * grad_phi_x * grad_phi_y
    sigma_yy = -zeta * grad_phi_y * grad_phi_y

    fx += divergence(sigma_xx, sigma_xy, dx)
    fy += divergence(sigma_xy, sigma_yy, dx)

    return mu, fx, fy, grad_phi_x, grad_phi_y

@njit(parallel=True)
def conserved_noise(nx, ny, strength):
    eta_x = strength * np.random.normal(size=(nx, ny))
    eta_y = strength * np.random.normal(size=(nx, ny))
    return eta_x, eta_y

def update(frame):
        for i, (im, title) in enumerate(zip(ims, titles)):
            phi_cont = history[title][frame]
            im.set_data(phi_cont)
        return ims

def worker(args):
    (seed, model_name, param_name, param_val,
     Nx, Ny, max_steps, save_every,
     base, skip_initial_frames, n_frames, dx) = args
    print(f"[PID {os.getpid()}] Saving seed {seed}, model {model_name}, {param_name}={param_val:.2f}")
    entry = MODEL_REGISTRY[model_name]

    np.random.seed(seed)
    state = np.random.choice([-1, 1], size=(Nx, Ny)) * 0.1
    state = np.ascontiguousarray(state)

    dt = entry["dt"]
    func = entry["func"]
    kwargs = entry["kwargs"]()
    kwargs[param_name] = param_val

    if model_name == "Active Model H":
        kx = 2*np.pi*np.fft.fftfreq(Nx, d=dx)
        ky = 2*np.pi*np.fft.fftfreq(Ny, d=dx)
        kx, ky = np.meshgrid(kx, ky, indexing='ij')
        k2 = kx**2 + ky**2
        k2[0,0] = 1
        params_tuple = (
            kwargs["a"], kwargs["b"], kwargs["K"], kwargs["dx"],
            kwargs["M"], kwargs["zeta"], kx, ky, k2
        )
    elif model_name == "Active Model B+":
        params_tuple = (
            kwargs["a"], kwargs["b"], kwargs["K"], kwargs["dx"],
            kwargs["M"], kwargs["lam"], kwargs["zeta"]
        )
    elif model_name == "Potts Model B" or model_name == "Noise Model":
        params_tuple = (
            kwargs["a"], kwargs["b"], kwargs["K"], kwargs["dx"],
            kwargs["M"], kwargs["kappa"]
    )  

    save_dir = os.path.join(base, model_name, f"Data_{param_val:.2f}", str(seed))
    os.makedirs(save_dir, exist_ok=True)

    frame_id = 0
    for step in range(max_steps):
        state += dt * func(state, *params_tuple)

        if step % save_every == 0:
            if frame_id >= skip_initial_frames:
                np.save(os.path.join(save_dir, f"{frame_id - skip_initial_frames}.npy"), state)
            frame_id += 1
            if frame_id >= n_frames + skip_initial_frames:
                break

####### Models #########
@njit(fastmath=True)
def Active_ModelB_plus(phi, a, b, K, dx, M, lam, zeta):
    mu = chemical_potential(phi, a, b, K, dx)
    mu += lam * grad_squared(phi, dx)
    dphidt = M * laplacian(mu, dx)

    gradx, grady = gradient(phi, dx)
    lap = laplacian(phi, dx)
    plus_x = zeta * lap * gradx
    plus_y = zeta * lap * grady

    dphidt -= divergence(plus_x, plus_y, dx)
    return dphidt

@njit(fastmath=True)
def Potts_ModelB(phi, a, b, K, dx, M, kappa):
    q = 4 * np.cos(4 * np.pi / kappa) * np.cos(4 * np.pi / kappa)
    mu = chemical_potential(phi, a, b, K, dx)
    mu += -q * np.sin(phi * q)
    dphidt = M * laplacian(mu, dx)
    return dphidt

def noise_model(phi, a, b, K, dx, M, kappa):
    mu = chemical_potential(phi, a, b, K, dx)
    dphidt = M * laplacian(mu, dx)

    eta_x = np.random.normal(size=phi.shape)
    eta_y = np.random.normal(size=phi.shape)
    noise = divergence(eta_x, eta_y, dx)

    dphidt += np.sqrt(kappa / 1) * noise # sqrt(kappa/dt)

    return dphidt

def Active_ModelH(phi, a, b, K, dx, M, zeta, kx, ky, k2):
    mu, fx, fy, grad_phi_x, grad_phi_y = compute_modelH_forces(phi, a, b, K, dx, zeta)
    vx, vy = solve_stokes(fx, fy, kx, ky, k2, eta=1)
    adv = vx * grad_phi_x + vy * grad_phi_y
    dphidt = M * laplacian(mu, dx) - adv
    return dphidt

######### Model Register #########
dx = 1
MODEL_REGISTRY = {
    "Active Model B+": dict(func=Active_ModelB_plus, init=lambda: phi.copy(), step_type="scalar",
        kwargs=lambda: dict(a=-1, b=1, K=0.75, dx=dx, M=1, lam=0.5, zeta=-0.5), dt=1e-2,        
        enabled=True,),

    "Active Model H": dict(func=Active_ModelH, init=lambda: phi.copy(), step_type="scalar",
        kwargs=lambda: dict(a=-1, b=1, K=0.75, dx=dx, M=1, zeta=-0.5), dt=1e-2,
        enabled=True,),

    "Potts Model B": dict(func=Potts_ModelB, init=lambda: phi.copy(), step_type="scalar",
        kwargs=lambda: dict(a=-1, b=1, K=1, dx=dx, M=1, kappa=6), dt=1e-2,
        enabled=True,),

    "Noise Model": dict(func=noise_model, init=lambda: phi.copy(), step_type="scalar",
        kwargs=lambda: dict(a=-1, b=1, K=1, dx=dx, M=1, kappa=6), dt=1e-2,
        enabled=True,),
}

#####################################################
if __name__ == "__main__":
    start_time_sim = time.time()
    Nx = 256
    Ny = Nx

    max_steps = 50000
    save_every = max_steps // 500

    phi = np.random.choice([-1, 1], size=(Nx,Ny)) * 0.1
    phi = np.ascontiguousarray(phi)

    kx = 2*np.pi*np.fft.fftfreq(Nx, d=dx)
    ky = 2*np.pi*np.fft.fftfreq(Ny, d=dx)
    kx, ky = np.meshgrid(kx, ky, indexing='ij')
    k2 = kx**2 + ky**2
    k2[0,0] = 1

#########
    names, funcs, states, params, dts  = [], [], [], [], []
    for name, entry in MODEL_REGISTRY.items():
        if not entry["enabled"]:
            continue
        names.append(name)
        funcs.append(entry["func"])
        states.append(entry["init"]())
        kwargs = entry["kwargs"]()
        if name == "Active Model H":
            params.append((
                kwargs["a"], kwargs["b"], kwargs["K"], kwargs["dx"],
                kwargs["M"], kwargs["zeta"], kx, ky, k2
            ))
        else:
            params.append(tuple(kwargs.values()))
        dts.append(entry["dt"])

    history = {name: [] for name in names}
    n_models = len(states)

    for n in range(max_steps):
        for i in range(n_models):
            states[i] += dts[i] * funcs[i](states[i], *params[i])

        if n % save_every == 0:
            for i, name in enumerate(names):
                history[name].append(states[i].copy())

    ncols = 3
    nrows = int(np.ceil(n_models / ncols))

    fig, axes = plt.subplots(nrows, ncols, figsize=(4*ncols, 4*nrows))
    axes = np.atleast_1d(axes).ravel()

    titles = list(history.keys())
    ims = []
    for ax in axes[n_models:]:
        ax.axis("off")
    for ax, title in zip(axes, titles):
        im = ax.imshow(history[title][0],
                    cmap='coolwarm',
                    origin='lower',
                    vmin=-1, vmax=1,
                    animated=True)
        ax.set_title(title)
        ax.axis("off")
        ims.append(im)

    cbar = fig.colorbar(ims[0], ax=axes[:n_models], fraction=0.025, pad=0.04)
    cbar.set_label(r'$\phi$')

    ani = animation.FuncAnimation(fig, update, frames=len(history[titles[0]]), interval=50, blit=False)
    end_time_sim = time.time()
    print(f"Simulation runtime: {end_time_sim - start_time_sim:.2f} seconds")
    # plt.show()

########## Save Models ################
    start_time_sim = time.time()
    SAVE_MODELS = True
    base = fr"   "

    n_seeds = 100
    n_frames = 2
    skip_initial_frames = 5

    zeta_values = [0, -0.05, -0.1, -0.5, -1]
    kappa_values = [2, 8/3, 3, 4, 6]

    if SAVE_MODELS:
        tasks = []
        for seed in range(n_seeds):
            for model_name, entry in MODEL_REGISTRY.items():
                if not entry["enabled"]:
                    continue

                if "zeta" in entry["kwargs"]():
                    param_name = "zeta"
                    param_values = zeta_values
                elif "kappa" in entry["kwargs"]():
                    param_name = "kappa"
                    param_values = kappa_values
                else:
                    continue

                for param_val in param_values:
                    tasks.append((
                        seed, model_name, param_name, param_val,
                        Nx, Ny, max_steps, save_every,
                        base, skip_initial_frames, n_frames, dx
                    ))

        with Pool(min(cpu_count(), 4)) as p:
            p.map(worker, tasks)

    end_time_sim = time.time()
    print(f"Model Saving runtime: {end_time_sim - start_time_sim:.2f} seconds")
