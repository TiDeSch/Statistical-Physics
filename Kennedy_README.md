# Adaptive Kennedy SLE Simulator

Parallel implementation of the adaptive Kennedy algorithm for generating chordal Schramm–Loewner Evolution (SLE) traces in the upper half-plane.

The code follows the adaptive refinement strategy described by Tom Kennedy in *Numerical Computations for the Schramm–Loewner Evolution* and uses vertical slit maps to approximate the Loewner evolution. Adaptive Brownian-bridge refinement is used to resolve poorly sampled regions of the trace.

---

## Features

* Chordal SLE simulation in the upper half-plane
* Vertical slit Loewner discretization
* Adaptive time refinement based on spatial resolution
* Brownian bridge midpoint sampling
* Numba-accelerated conformal map compositions
* Multiprocessing support for generating many traces simultaneously

---

## Math

Chordal SLE is defined by the Loewner equation

[
\frac{\partial g_t(z)}{\partial t}
==================================

\frac{2}{g_t(z)-U_t},
]

with driving function

[
U_t = \sqrt{\kappa} B_t,
]

where (B_t) is standard Brownian motion.

The algorithm approximates the driving function on short intervals and replaces each interval by an explicitly solvable vertical slit map

[
f(w)=\delta+\sqrt{w^2-4\Delta t}.
]

Successive compositions of these maps generate an approximation of the SLE trace.

---

## Adaptive Refinement

Using uniform time steps can undersample portions of the curve, particularly for larger values of (\kappa). Following Kennedy's adaptive approach, intervals are refined whenever the spatial increment

[
|z_k-z_{k-1}|
]

exceeds a prescribed threshold `eps`. New midpoint driving values are generated using a Brownian bridge. Refinement continues until all increments satisfy

[
|z_k-z_{k-1}| \le \varepsilon.
]

---


## Parameters

| Parameter       | Description                         |
| --------------- | ----------------------------------- |
| `kappa`         | SLE diffusivity parameter           |
| `T`             | Total Loewner time                  |
| `eps`           | Adaptive refinement threshold       |
| `initial_steps` | Initial uniform time discretization |
| `max_iters`     | Maximum refinement iterations       |
| `max_points`    | Maximum number of curve points      |
| `num_curves`    | Number of independent SLE traces    |
| `start_seed`    | Initial seed                        |
| `bad_seeds`     | Seeds to skip                       |

Typical SLE values:

| κ   |                         |
| --- | ----------------------- |
| 2   | Loop-erased random walk |
| 8/3 | Self-avoiding walk      |
| 4   | Critical transition     |
| 6   | Critical percolation    |
| 8   | Space-filling curve     |

---

## Output

The script creates

```text
Data_and_Results/
    Kennedy/
        Results_kappa{kappa}/
```

containing

```text
SLE_Trace_kappa{kappa}.npy
SLE_times_kappa{kappa}.npy
SLE_log_kappa{kappa}.txt
Kennedy_SLE{kappa}.pdf
```

### SLE_Trace

Shape:

```python
(2, num_curves, max_trace_length)
```

where

```python
SLE_Trace[0]
```

contains real coordinates and

```python
SLE_Trace[1]
```

contains imaginary coordinates.

Unused entries are padded with `NaN`.

### SLE_times

Adaptive time partitions corresponding to each generated trace.

### Log File

Records:

* convergence status
* final maximum increment
* number of refinement iterations
* number of points
* runtime statistics

---

## Performance

Performance improvements include:

* Numba JIT compilation
* Cached numerical kernels
* Parallel curve generation using multiprocessing
* Selective refinement of only the worst intervals

The refinement strategy can significantly reduce computational cost compared with globally refining all intervals.

---

## References

Tom Kennedy,

> Numerical Computations for the Schramm–Loewner Evolution

arXiv:0909.2438

This implementation is based primarily on the adaptive refinement strategy and vertical slit discretization described in the paper.
