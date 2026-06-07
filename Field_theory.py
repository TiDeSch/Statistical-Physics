import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import time
from numba import njit
np.random.seed(13)

"""
Collection of simmulated field theory model.
All continuum models

Passive models:
    Model A (Cahn Hillard)
    Model B (Cahn Hillard)

Active models:
    Active Model A (Cahn Hillard) 
    Active Model B (Cahn Hillard) 
    Active Model B+ (Cahn Hillard)
    Active Model AB (Cahn Hillard)  

    Active Model H (Cahn Hillard) 
    Active Model J (Cahn Hillard) 

    Potts Model B
    Toner-Tu Model
    KPZ Model
"""

####### Functions #############
@njit(fastmath=True)
def laplacian(arr, dx):
    nx, ny = arr.shape
    out = np.empty_like(arr)
    for i in range(nx):
        for j in range(ny):
            out[i, j] = (
                arr[(i+1)%nx, j] +
                arr[(i-1)%nx, j] +
                arr[i, (j+1)%ny] +
                arr[i, (j-1)%ny] -
                4.0 * arr[i, j]
            ) / (dx*dx)
    return out

@njit(fastmath=True)
def divergence(vx, vy, dx):
    nx, ny = vx.shape
    div = np.empty_like(vx)
    for i in range(nx):
        ip = (i+1)%nx
        im = (i-1)%nx
        for j in range(ny):
            jp = (j+1)%ny
            jm = (j-1)%ny
            div[i,j] = (
                vx[ip,j] - vx[im,j] +
                vy[i,jp] - vy[i,jm]
            )/(2*dx)
    return div

@njit(fastmath=True)
def gradient(arr, dx):
    nx, ny = arr.shape
    gx = np.empty_like(arr)
    gy = np.empty_like(arr)
    for i in range(nx):
        ip = (i+1)%nx
        im = (i-1)%nx
        for j in range(ny):
            jp = (j+1)%ny
            jm = (j-1)%ny
            gx[i,j] = (arr[ip,j] - arr[im,j])/(2*dx)
            gy[i,j] = (arr[i,jp] - arr[i,jm])/(2*dx)
    return gx, gy

@njit(fastmath=True)
def grad_squared(phi, dx):
    nx, ny = phi.shape
    out = np.empty_like(phi)
    for i in range(nx):
        ip = (i+1) % nx
        im = (i-1) % nx
        for j in range(ny):
            jp = (j+1) % ny
            jm = (j-1) % ny
            dxp = (phi[ip, j] - phi[im, j]) / (2*dx)
            dyp = (phi[i, jp] - phi[i, jm]) / (2*dx)
            out[i, j] = dxp*dxp + dyp*dyp
    return out

def chemical_potential(phi, a, b, K, dx):
    mu = a * phi + b * phi * phi * phi - K * laplacian(phi, dx)
    return mu

def solve_stokes(fx, fy, dx, eta):
    Nx, Ny = fx.shape
    kx = 2*np.pi*np.fft.fftfreq(Nx, d=dx)
    ky = 2*np.pi*np.fft.fftfreq(Ny, d=dx)
    kx, ky = np.meshgrid(kx, ky, indexing='ij')

    k2 = kx**2 + ky**2
    k2[0,0] = 1  # avoid divide by zero

    fx_hat = np.fft.fft2(fx)
    fy_hat = np.fft.fft2(fy)

    k_dot_f = kx*fx_hat + ky*fy_hat

    vx_hat = (fx_hat - kx * k_dot_f / k2) / (eta * k2)
    vy_hat = (fy_hat - ky * k_dot_f / k2) / (eta * k2)

    vx_hat[0,0] = 0
    vy_hat[0,0] = 0

    vx = np.real(np.fft.ifft2(vx_hat))
    vy = np.real(np.fft.ifft2(vy_hat))

    return vx, vy

def noise(shape, kBT, M, dx, dt):
    ndim = len(shape)
    sigma = np.sqrt(2 * kBT * M / (dx**ndim * dt))
    return sigma * np.random.normal(size=(ndim, *shape))

def conserved_noise(shape, kBT, M, dx, dt):
    sigma = np.sqrt(2 * M * kBT / dt)
    eta_x = sigma * np.random.randn(*shape)
    eta_y = sigma * np.random.randn(*shape)
    return eta_x, eta_y

def step(phi, model, dt, **kwargs): # Euler
    return phi + dt * model(phi=phi, **kwargs)

def run_to_steady_state(phi0, model, dt, max_steps, tol=1e-6, **model_kwargs):
    phi = phi0.copy()
    for n in range(max_steps):
        dphidt = model(phi=phi, **model_kwargs)
        phi += dt * dphidt

        residual = np.max(np.abs(dphidt))
        if residual < tol:
            print(f"Converged after {n} steps")
            break
    return phi

def run_to_steady_state_AB(state0, model, dt, max_steps, tol=1e-8, **model_kwargs):
    phiA, phiB = state0
    for n in range(max_steps):
        dphiA, dphiB = model(phiA, phiB, **model_kwargs)
        phiA += dt * dphiA
        phiB += dt * dphiB

        residual = max(np.max(np.abs(dphiA)),np.max(np.abs(dphiB)))
        if residual < tol:
            print(f"Converged after {n} steps")
            break
    return phiA + phiB

####### Passive Models #########
def ModelA(phi, a, b, K, dx, M, kBT, add_noise=False):
    mu = chemical_potential(phi, a, b, K, dx)
    dphidt = -M * mu
    if add_noise:
        eta = np.sqrt(2 * kBT * M)
        dphidt += eta * np.random.randn(*phi.shape)
    return dphidt

def ModelB(phi, a, b, K, dx, M, kBT, add_noise=False):
    mu = chemical_potential(phi, a, b, K, dx)
    dphidt = M * laplacian(mu, dx)
    if add_noise:
        eta = noise(phi.shape, kBT, M, dx, dt=1)
        eta_x, eta_y = eta
        dphidt -= divergence(eta_x, eta_y, dx)
    return dphidt

####### Active CH Models #########
def Active_ModelA(phi, a, b, K, dx, M, kBT, lam, add_noise=False):
    mu = chemical_potential(phi, a, b, K, dx)
    mu += lam * grad_squared(phi, dx)
    dphidt = - M * mu
    if add_noise:
        eta = np.sqrt(2 * kBT * M)
        dphidt += eta * np.random.randn(*phi.shape)
    return dphidt

def Active_ModelB(phi, a, b, K, dx, M, kBT, lam, add_noise=False):
    mu = chemical_potential(phi, a, b, K, dx)
    mu += lam * grad_squared(phi, dx)
    dphidt = M * laplacian(mu, dx)
    if add_noise:
        eta = noise(phi.shape, kBT, M, dx, dt=1)
        eta_x, eta_y = eta[0], eta[1]
        dphidt -= divergence(eta_x, eta_y, dx)
    return dphidt

def Active_ModelB_plus(phi, a, b, K, dx, M, kBT, lam, zeta, add_noise=False):
    mu = chemical_potential(phi, a, b, K, dx)
    mu += lam * grad_squared(phi, dx)
    dphidt = M * laplacian(mu, dx)

    plus = zeta * laplacian(phi, dx) * gradient(phi, dx)
    plus_x, plus_y = plus[0], plus[1]
    dphidt -= divergence(plus_x, plus_y, dx)

    if add_noise:
        eta = noise(phi.shape, kBT, M, dx, dt=1)
        eta_x, eta_y = eta[0], eta[1]
        dphidt -= divergence(eta_x, eta_y, dx)
    return dphidt

def Active_ModelAB(phiA, phiB, a, b, K, dx, MA, MB, kBT, alpha, beta, add_noise=False):
    phi = phiA + phiB
    mu = chemical_potential(phi, a, b, K, dx)

    dphiAdt = - MA * mu + alpha * phiB
    J = MB * mu + beta * phiA
    dphiBdt = laplacian(J, dx) 

    if add_noise:
        etaA = np.sqrt(2 * kBT * MA) * np.random.randn(*phi.shape)
        dphiAdt += etaA
        etaB_x, etaB_y = noise(phi.shape, kBT, MB, dx, dt=1)
        dphiBdt -= divergence(etaB_x, etaB_y, dx)
    return dphiAdt, dphiBdt

####### Other Active Models #########
def Potts_ModelB(phi, a, b, K, dx, M, kBT, kappa, add_noise=False):
    q = 4 * np.cos(4 * np.pi / kappa) * np.cos(4 * np.pi / kappa)
    mu = chemical_potential(phi, a, b, K, dx)
    mu += np.sin(phi * q)
    dphidt = M * laplacian(mu, dx)

    if add_noise:
        eta = noise(phi.shape, kBT, M, dx, dt=1)
        eta_x, eta_y = eta[0], eta[1]
        dphidt -= divergence(eta_x, eta_y, dx)
    return dphidt

def TonerTu_Model(phix, phiy, alpha, beta, nu, dx, kBT, add_noise=False):
    v2 = phix**2 + phiy**2
    force_x = alpha*phix - beta*v2*phix + nu*laplacian(phix, dx)
    force_y = alpha*phiy - beta*v2*phiy + nu*laplacian(phiy, dx)

    if add_noise:
        eta_x, eta_y = noise(phix.shape, kBT, 1, dx, dt=1)
        force_x += eta_x
        force_y += eta_y
    return force_x, force_y

def KPZ_Model(phi, nu, lam, dx, kBT, add_noise=False):
    h = phi
    dhdt = nu * laplacian(h, dx) + 0.5 * lam * grad_squared(h, dx)
    if add_noise:
        sigma = np.sqrt(2 * kBT / (dx**2))
        dhdt += sigma * np.random.randn(*h.shape)
    return dhdt

def Active_ModelH(phi, a, b, K, dx, M, kBT, zeta, add_noise=False):
    mu = chemical_potential(phi, a, b, K, dx)
    grad_mu_x, grad_mu_y = gradient(mu, dx)
    fx = -phi * grad_mu_x
    fy = -phi * grad_mu_y

    grad_phi_x, grad_phi_y = gradient(phi, dx)
    sigma_xx = -zeta * grad_phi_x * grad_phi_x
    sigma_xy = -zeta * grad_phi_x * grad_phi_y
    sigma_yy = -zeta * grad_phi_y * grad_phi_y

    fx += divergence(sigma_xx, sigma_xy, dx)
    fy += divergence(sigma_xy, sigma_yy, dx)

    vx, vy = solve_stokes(fx, fy, dx, eta=1)
    adv = vx * grad_phi_x + vy * grad_phi_y
    dphidt = M * laplacian(mu, dx) - adv

    if add_noise:
        eta_x, eta_y = conserved_noise(phi.shape, kBT, M, dx, dt=1)
        dphidt -= divergence(eta_x, eta_y, dx)
    return dphidt

def Active_ModelJ(phi, a, b, K, dx, M, kBT, lam, zeta, add_noise=False):
    mu = chemical_potential(phi, a, b, K, dx)
    mu += lam * grad_squared(phi, dx)

    dphidt = M * laplacian(mu, dx)

    grad_phi_x, grad_phi_y = gradient(phi, dx)
    lap_phi = laplacian(phi, dx)

    # (nabla^2(phi)) nabla(phi)
    Jx = lap_phi * grad_phi_x
    Jy = lap_phi * grad_phi_y

    # - 1/2 nabla(|nabla(phi)|^2)
    grad2 = grad_squared(phi, dx)
    g2x, g2y = gradient(grad2, dx)

    Jx -= 0.5 * g2x
    Jy -= 0.5 * g2y

    dphidt -= divergence(zeta * Jx, zeta * Jy, dx)

    if add_noise:
        eta = noise(phi.shape, kBT, M, dx, dt=1)
        eta_x, eta_y = eta[0], eta[1]
        dphidt -= divergence(eta_x, eta_y, dx)
    return dphidt

######### Model Register #########
MODEL_REGISTRY = {
    # -------- Passive CH Models --------
    "Model A": dict(func=ModelA, init=lambda: phi.copy(), step_type="scalar",
        kwargs=lambda: dict(a=-1, b=1, K=2, dx=1, M=1, kBT=1, add_noise=False), dt=1e-3,
        enabled=False,),

    "Model B": dict(func=ModelB, init=lambda: phi.copy(), step_type="scalar",
        kwargs=lambda: dict(a=-1, b=1, K=0.75, dx=1, M=1, kBT=1, add_noise=False), dt=1e-3,
        enabled=False,),

    # -------- Active CH Models --------
    "Active Model A": dict(func=Active_ModelA, init=lambda: phi.copy(), step_type="scalar",
        kwargs=lambda: dict(a=-1, b=1, K=2, dx=1, M=1, kBT=1, lam=-1, add_noise=False), dt=1e-3,     
        enabled=False,),

    "Active Model B": dict(func=Active_ModelB, init=lambda: phi.copy(), step_type="scalar",
        kwargs=lambda: dict(a=-1, b=1, K=3, dx=1, M=1, kBT=1, lam=-3, add_noise=False), dt=5*1e-3,        
        enabled=False,),

    "Active Model B+": dict(func=Active_ModelB_plus, init=lambda: phi.copy(), step_type="scalar",
        kwargs=lambda: dict(a=-1, b=1, K=0.75, dx=1, M=1, kBT=1, lam=4, zeta=0.2, add_noise=False), dt=1e-3,        
        enabled=False,),

    "Active Model H": dict(func=Active_ModelH, init=lambda: phi.copy(), step_type="scalar",
        kwargs=lambda: dict(a=-1, b=1, K=0.75, dx=1, M=1, kBT=1, zeta=-0.5, add_noise=False), dt=1e-3,
        enabled=True,),

    "Active Model J": dict(func=Active_ModelJ, init=lambda: phi.copy(), step_type="scalar",
        kwargs=lambda: dict(a=-1, b=1, K=2.5, dx=1, M=1, kBT=1, lam=2.5, zeta=2, add_noise=False), dt=1e-2,
        enabled=False,),

    "Active Model AB": dict(func=Active_ModelAB, init=lambda: (phiA.copy(), phiB.copy()), step_type="vector",
        kwargs=lambda: dict(a=-1, b=1, K=5, dx=1, MA=1, MB=1, kBT=1, alpha=-1, beta=-3, add_noise=False), dt=1e-3,
        enabled=False,),

    # -------- Other scalar models --------
    "Toner-Tu Model": dict(func=TonerTu_Model, init=lambda: (phix.copy(), phiy.copy()), step_type="vector",
        kwargs=lambda: dict(alpha=1, beta=1, nu=0.5, dx=1, kBT=1, add_noise=False), dt=1e-3,
        enabled=False,),

    "Potts Model B": dict(func=Potts_ModelB, init=lambda: phi_Potts.copy(), step_type="scalar",
        kwargs=lambda: dict(a=-1, b=1, K=1, dx=1, M=1, kBT=1, kappa=6, add_noise=False), dt=1e-3,
        enabled=True,),

    "KPZ Model": dict(func=KPZ_Model, init=lambda: phi.copy()*15, step_type="scalar",
        kwargs=lambda: dict(nu=1, lam=-0.5, dx=1, kBT=10, add_noise=False), dt=1e-4,
        enabled=False,),
}

#####################################################
if __name__ == "__main__":
    start_time_sim = time.time()
    Nx = 128
    Ny = Nx

    phi = np.random.choice([-1, 1], size=(Nx,Ny)) * 0.1

    phiA = np.random.choice([-1, 1], size=(Nx,Ny)) * 0.05
    phiB = np.random.choice([-1, 1], size=(Nx,Ny)) * 0.01

    phix = phiA
    phiy = phiB
    phi_Potts = np.random.uniform(-2, 2, (Nx,Ny)) * 5

    max_steps = 50000
    save_every = max_steps/50

######################
    models = {name: dict(
            state=entry["init"](),
            func=entry["func"],
            step_type=entry["step_type"],
            kwargs=entry["kwargs"]())
        for name, entry in MODEL_REGISTRY.items()
        if entry["enabled"]}

    history = {name: [] for name in models}
    for n in range(max_steps):
        for name, model in models.items():
            dt = MODEL_REGISTRY[name]["dt"]
            if model["step_type"] == "scalar":
                model["state"] += dt * model["func"](
                    model["state"], **model["kwargs"])

            elif model["step_type"] == "vector":
                f1, f2 = model["state"]
                df1, df2 = model["func"](f1, f2, **model["kwargs"])
                f1 += dt * df1
                f2 += dt * df2
                model["state"] = (f1, f2)

        if n % save_every == 0:
            for name, model in models.items():
                if model["step_type"] == "vector":
                    f1, f2 = model["state"]
                    history[name].append((f1 + f2).copy())
                else:
                    history[name].append(model["state"].copy())

    n_models = len(history)
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

    cbar = fig.colorbar(ims[0], ax=axes[:n_models],
                        fraction=0.025, pad=0.04)
    cbar.set_label(r'$\phi$')

    def update(frame):
        for im, title in zip(ims, titles):
            im.set_data(history[title][frame])
        return ims

    ani = animation.FuncAnimation(
        fig, update,
        frames=len(history[titles[0]]),
        interval=50
    )

    end_time_sim = time.time()
    print(f"Total runtime: {end_time_sim - start_time_sim:.2f} seconds")
    plt.show()

# ########## Steady State / Dynamic Equlibrium #####################
start_time_SS = time.time()
steady_states = {}
for name, entry in MODEL_REGISTRY.items():
    if not entry["enabled"]:
        continue
    print(f"Running steady state for {name}...")

    initial_state = entry["init"]()
    model_func = entry["func"]
    kwargs = entry["kwargs"]()

    # ---------- Scalar models ----------
    if entry["step_type"] == "scalar":
        steady_states[name] = run_to_steady_state(
            initial_state,
            model_func,
            dt=entry["dt"],
            max_steps=max_steps,
            **kwargs
        )

    # ---------- Vector / Two-field models ----------
    elif entry["step_type"] == "vector":
        steady_states[name] = run_to_steady_state_AB(
            initial_state,
            model_func,
            dt=entry["dt"],
            max_steps=max_steps,
            **kwargs
        )

# ---------- Plotting ----------
n_models = len(steady_states)
if n_models == 0:
    print("No models enabled for steady state.")
else:
    ncols = 3
    nrows = int(np.ceil(n_models / ncols))

    fig, axes = plt.subplots(nrows, ncols,
                             figsize=(4*ncols, 4*nrows))
    axes = np.atleast_1d(axes).ravel()
    for ax in axes[n_models:]:
        ax.axis("off")
    for ax, (name, field) in zip(axes, steady_states.items()):
        im = ax.imshow(field,
                       cmap="coolwarm",
                       origin="lower",
                       vmin=-1, vmax=1)
        ax.set_title(name)
        ax.axis("off")
    cbar = fig.colorbar(
        im,
        ax=axes[:n_models],
        fraction=0.025,
        pad=0.04
    )
    cbar.set_label(r"$\phi$")

    end_time_SS = time.time()
    print(f"Total runtime: {end_time_SS - start_time_SS:.2f} seconds")
    plt.show()