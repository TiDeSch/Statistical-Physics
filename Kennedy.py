import numpy as np
import matplotlib.pyplot as plt 
import time
import os
from numba import njit # For just-in-time (JIT) compilation to speed up numerical functions
from multiprocessing import Pool, cpu_count
from datetime import datetime
PC_time = datetime.now()
start_time = time.time()

"""
Adaptive Kennedy algorithm to generate SLE traces in the upper half plane.
It follows the Vertical slit method.  
Based on Kennedy 2009 (https://arxiv.org/abs/0909.2438). 
All parallelised to optimise performance. 
See Kennedy_README.md for more comprehensive description.
"""

@njit(cache=True)
def complex_sqrt(z): # Ensures that the imaginary part of the result is nonnegative (just 'np.sqrt' can cause discontinuous jumps or branch flips)
    s = np.sqrt(z)
    if s.imag < 0: # Flip imaginary part if negative
        s = -s                     
    return s 

@njit(inline='always')
def slit_map(w, dt, delta):
    s = np.sqrt(w*w - 4.0*dt)
    if s.imag < 0:
        s = -s
    return delta + s

@njit(cache=True, fastmath=True)
def Vslit(times, xi):
    M = len(times) - 1
    z = np.zeros(M + 1, dtype=np.complex128) # Initialize complex array

    delta = xi[1:] - xi[:-1] # Brownian increment for driving function
    dt = times[1:] - times[:-1] # Time increment
    
    for k in range(1, M + 1):
        w = 0.0 + 0.0j # Initialize complex variable w to 0
        for j in range(k - 1, -1, -1): # Iterate backward over intervals
            arg = w * w - 4.0 * dt[j] 
            s = np.sqrt(arg)
            if s.imag < 0: # Flip imaginary part if negative
                s = -s
            w = delta[j] + s # Vertical slit
        z[k] = w
    return z

def pad_ragged(ar):
    max_length = max(a.shape[0] if hasattr(a, 'shape') else len(a) for a in ar)
    arout = np.full((len(ar), max_length), np.nan)
    for i, t in enumerate(ar):
        try:
            l = t.shape[0]
        except:
            l = len(t)
        arout[i] = np.pad(t, (0, max_length - l), constant_values=np.nan)
    return arout

def trace_pad(x_traces,y_traces):
    return [pad_ragged(x_traces), pad_ragged(y_traces)]

def worker(args):
    kappa, T, eps, initial_steps, max_points, seed, verbose, max_iters, log_path = args
    rng = np.random.default_rng(seed)

    #Initialize uniform time partition
    times = list(np.linspace(0, T, initial_steps + 1))
    dt0 = T / initial_steps # Initial time step size

    #Generate initial Brownian motion
    xi = [0.0] # Float
    for _ in range(initial_steps):
        xi.append(xi[-1] + rng.normal(scale=np.sqrt(kappa * dt0))) # Gaussian increment

    z = None
    for it in range(1, max_iters + 1):
        times_arr = np.array(times, dtype=np.float64) # Convert list to NumPy array for computation
        xi_arr = np.array(xi, dtype=np.float64)

        incs = np.abs(np.diff(z)) # Compute increments |dz|
        max_inc = incs.max() if len(incs) > 0 else 0 # Find the largest increment

        final_msg = ""
        if verbose:
            print(f"[seed={seed}] Iteration {it}: points={len(times)-1}, max|dz|={max_inc:.3f}")

        if max_inc <= eps:
            final_msg = f"[seed={seed}] Iteration {it}: points={len(times)-1}, max|dz|={max_inc:.3f} — Converged\n"
            with open(log_path, 'a') as f:
                f.write(final_msg)  
            if verbose:
                print(f"[seed={seed}] Converged at iteration {it}.")
            break

        if (len(times) - 1) >= max_points:
            final_msg = f"[seed={seed}] Iteration {it}: points={len(times)-1}, max|dz|={max_inc:.3f} — Reached max_points\n"
            with open(log_path, 'a') as f:
                f.write(final_msg)  
            if verbose:
                print(f"[seed={seed}] Reached {max_points} point — stopping.")
            break

        if it == max_iters:
            final_msg = f"[seed={seed}] Iteration {it}: points={len(times)-1}, max|dz|={max_inc:.3f} — Reached max_iters\n"
            with open(log_path, 'a') as f:
                f.write(final_msg)  
            if verbose:
                print(f"[seed={seed}] Reached {max_iters} iterations — stopping.")
            break
        
        # Brownian bridge
        # Only refine the worst % of bad intervals per iteration
        # worst_fraction = max(0.05, min(0.3, 1 - eps / max_inc))   # Start broard, taper in (Slowest, Best avg. max|dz|, always hit max iterations)
        worst_fraction = max(0.05, min(0.3, eps / (max_inc)))       # Start narrow, taper out (Fastest, possible large max|dz| on some seeds, often converge.)
        # worst_fraction = 0.1                                      # Constant % (Fast, small avg. max|dz|)

        bad_mask = incs > eps
        bad_idx = np.where(bad_mask)[0]
        if len(bad_idx) > 0:
            # Sort by increment size descending, take worst top_fraction
            sorted_bad = bad_idx[np.argsort(incs[bad_idx])[::-1]]
            n_refine = max(1, int(len(sorted_bad) * worst_fraction))
            bad_indices = set(sorted_bad[:n_refine] + 1)

        # Refine intervals by inserting midpoints
        new_times, new_xi = [], []
        insert_set = set(bad_indices)
        for k in range(len(times)):
            new_times.append(times[k])
            new_xi.append(xi[k])
            if k + 1 in insert_set:
                t_left, t_right = times[k], times[k + 1] # Left and right endpoint time
                xi_left, xi_right = xi[k], xi[k + 1] # Left and right endpoint Brownian value (driving function)
                t_mid = (t_left + t_right) / 2 # Midpoint time
                var_mid = kappa * (t_right - t_left) / 4 # Variance for midpoint increment (var = kappa * dt / 2 ())
                xi_mid = (xi_left + xi_right) / 2 + rng.normal(scale=np.sqrt(var_mid)) # Generate midpoint value (driving function)
                new_times.append(t_mid), new_xi.append(xi_mid)

        times, xi = new_times, new_xi
    return z, np.array(times), it, len(times) - 1, max_inc

def Adaptive_SLE(kappa, T, eps, initial_steps, max_points, start_seed,
                           verbose, max_iters, num_curves, log_path, bad_seeds=None):
    bad_seeds = bad_seeds or set()
    seeds = []
    candidate = start_seed
    while len(seeds) < num_curves:
        if candidate not in bad_seeds:
            seeds.append(candidate)
        candidate += 1

    args_list = [(kappa, T, eps, initial_steps, max_points, seed, verbose, max_iters, log_path)
                 for seed in seeds]
    n_workers = min(cpu_count(), num_curves)
    print(f"Using {n_workers} parallel workers for {num_curves} curves for kappa{kappa}.\n")

    results = []
    all_iters, all_points, all_inc = [], [], []
    milestones = {num_curves//8, num_curves//4, 3*num_curves//8, num_curves//2, 5*num_curves//8, 3*num_curves//4, 7*num_curves//8, num_curves}
    milestone_start = time.time()
    with Pool(processes=n_workers) as pool:
        for z, times, final_it, final_pts, final_inc in pool.imap(worker, args_list):
            results.append({'z': z, 'times': times})
            all_iters.append(final_it)
            all_points.append(final_pts)
            all_inc.append(final_inc)

            count = len(results)
            if count in milestones:
                elapsed = time.time() - milestone_start
                PC_time_elapse = datetime.now()
                print(f"Progress: {count}/{num_curves} curves saved ({100*count//num_curves}%) - {elapsed:.2f}s")
                print(PC_time_elapse.strftime("%Y-%m-%d %H:%M:%S"))

    return results, all_iters, all_points, all_inc

if __name__ == "__main__":
    print("Script started at:", PC_time.strftime("%Y-%m-%d %H:%M:%S"))
    # kappa = 2: loop-erased random walk
    # kappa = 8/3 (2.67): self-avoiding walks
    # kappa = 4: marginal between simple and self-intersecting curves
    # kappa = 6: perculation
    # kappa = 8: space filling (uniform Spanning Tree, Peano curve)
    kappa = 2

    start_seed = 1
    bad_seeds = {1}  # add seeds to skip

    T = 50000
    eps = 0.5 # Precision (adaptive refinement threshold)
    max_iters = 120 # Maximum number of iterations. Ensures no indefinite refienment.
    max_points = 50000 # Max number of points
    initial_steps = int(np.sqrt(kappa * T))  # Time steps before adaptive refinement (the first itteration/curve) (low when kappa(2) low, high when kappa(6) high)
    verbose = True # Print progress information ('True' to show, 'False' to hide)
    num_curves = 1000 # Number of SLE traces to generate  

    _ = Vslit(np.array([0.0, 0.1], dtype=np.float64), np.array([0.0, 0.1], dtype=np.float64)) # Dummy function to warmup numba
    print(f"Kennedy algorithm for kappa = {kappa}: Initial Steps = {initial_steps}, Initial Time-step = {T//initial_steps}, Total time = {T}")

    save_dir = f"Data_and_Results/Kennedy/Results_kappa{kappa}"
    os.makedirs(save_dir, exist_ok=True)
    log_path = os.path.join(save_dir, f'SLE_log_kappa{kappa}.txt')

    n_bad_seeds = len(bad_seeds) if bad_seeds is not None else 0
    with open(log_path, 'w') as f:
        f.write(f"Start - {PC_time.strftime("%Y-%m-%d %H:%M:%S")}\n")
        f.write(f"SLE {num_curves} traces — kappa={kappa}, eps={eps}\n")
        f.write(f"T={T}, max iterations={max_iters}, max point={max_points}, Skiped {n_bad_seeds} bad seeds\n")
        f.write("=" * 50 + "\n")

######### Generate Curves
    results, all_iters, all_points, all_inc = Adaptive_SLE(
        kappa=kappa, T=T, eps=eps,
        initial_steps=initial_steps, max_points=max_points,
        start_seed=start_seed, verbose=verbose,
        max_iters=max_iters, num_curves=num_curves, log_path=log_path, 
        bad_seeds=bad_seeds
    )

######### Pad with NaN
    x_traces = np.array([r['z'].real for r in results], dtype=object) # Extract real part of curves 
    y_traces = np.array([r['z'].imag for r in results], dtype=object) # Extract imaginary part of curves
    x_traces_padded, y_traces_padded = trace_pad(x_traces, y_traces)
    SLE_Trace = np.array([x_traces_padded, y_traces_padded])

    times_list = np.array([r['times'] for r in results], dtype=object)
    times_padded = pad_ragged(times_list)

######### Save SLE Traces
    save_path = os.path.join(save_dir, f'SLE_Trace_kappa{kappa}.npy')
    np.save(save_path, SLE_Trace)

    save_times_path = os.path.join(save_dir, f'SLE_times_kappa{kappa}.npy')
    np.save(save_times_path, times_padded)

######### Plot
    plt.figure(figsize=(6, 6)) 
    for i, r in enumerate(results):
        plt.plot(r['z'].real, r['z'].imag, lw=0.6)
    plt.title(f"Adaptive SLE curve, kappa={kappa:.2f}")
    plt.xlabel("Re(z)")
    plt.ylabel("Im(z)")
    plt.grid(alpha=0.6)
    filename = str(save_dir) + f'/Kennedy_SLE{kappa}.pdf'
    plt.savefig(filename, dpi=150, bbox_inches='tight')

    end_time = time.time()
    PC_end_time = datetime.now()
    with open(log_path, 'a') as f:
            f.write("=" * 25 + "\n")
            f.write(f"Avg. max|dz|={np.mean(all_inc):.2f}, Avg. interations={np.mean(all_iters):.2f}, Avg. points={np.mean(all_points):.2f}\n")
            f.write("=" * 50 + "\n")
            f.write(f"Start - {PC_end_time.strftime("%Y-%m-%d %H:%M:%S")} \n")
            f.write(f"Kennedy SLE runtime: {end_time - start_time:.2f} seconds")

    print(f"Kennedy SLE runtime: {end_time - start_time:.2f} seconds")
    plt.show()