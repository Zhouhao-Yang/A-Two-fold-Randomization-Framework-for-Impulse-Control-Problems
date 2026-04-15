from scipy.optimize import fsolve
import numpy as np
from scipy.sparse import diags
from scipy.sparse.linalg import spsolve
import matplotlib.pyplot as plt
import argparse

# ──────────────────────────────────────────────
# CLI arguments
# ──────────────────────────────────────────────
parser = argparse.ArgumentParser(description="Classical QVI baseline for RIC")
parser.add_argument("--batch_size",   type=int,   default=256,   help="parallel paths per step in collect_buffer (default: 256)")
parser.add_argument("--T",            type=float, default=20.0,  help="total time horizon for buffer collection (default: 20.0)")
parser.add_argument("--dt",           type=float, default=0.02,  help="time step for buffer collection (default: 0.02)")
parser.add_argument("--outer_iters",  type=int,   default=30,    help="outer policy-iteration iterations (default: 30)")
parser.add_argument("--inner_iters",  type=int,   default=30,    help="inner Gauss-Seidel sweeps per outer iter (default: 30)")
parser.add_argument("--seed",         type=int,   default=None,  help="random seed for buffer collection (default: None)")
args = parser.parse_args()

batch_size  = args.batch_size
T           = args.T
dt          = args.dt
outer_iters = args.outer_iters
inner_iters = args.inner_iters
seed        = args.seed

n_steps = int(T / dt)
print(f"\n[Config]  batch_size={batch_size}  T={T}  dt={dt}  "
      f"n_steps={n_steps}  total_samples={n_steps * batch_size:,}  "
      f"outer_iters={outer_iters}  inner_iters={inner_iters}  seed={seed}\n")

# ──────────────────────────────────────────────
# Model parameters
# ──────────────────────────────────────────────
r     = 0.10
mu    = 0.03
sigma = 0.20
h     = 1.0
p     = 1.0
Kp    = 2.0
kp    = 0.5
Km    = 2.0
km    = 0.5

disc = mu**2 + 2*r*sigma**2
t1 = (-mu + np.sqrt(disc)) / sigma**2   # > 0
t2 = (-mu - np.sqrt(disc)) / sigma**2   # < 0
print(f"t1 = {t1:.4f},   t2 = {t2:.4f}")


# ──────────────────────────────────────────────
# Analytic solution
# ──────────────────────────────────────────────
def V_pos(x, c1, c2):
    return h*x/r + mu*h/r**2 + c1*np.exp(t1*x) + c2*np.exp(t2*x)

def V_pos_p(x, c1, c2):
    return h/r + c1*t1*np.exp(t1*x) + c2*t2*np.exp(t2*x)

def coeff_neg(c1, c2):
    A  = (h + p) / r
    C1 = c1 - A*t2 / (t1*(t1 - t2))
    C2 = c2 + A*t1 / (t2*(t1 - t2))
    return C1, C2

def V_neg(x, c1, c2):
    C1, C2 = coeff_neg(c1, c2)
    return -p*x/r - mu*p/r**2 + C1*np.exp(t1*x) + C2*np.exp(t2*x)

def V_neg_p(x, c1, c2):
    C1, C2 = coeff_neg(c1, c2)
    return -p/r + C1*t1*np.exp(t1*x) + C2*t2*np.exp(t2*x)

def F(X):
    c1, c2, d, D, u, U = X
    return [
        V_neg_p(d, c1, c2) + kp,
        V_neg_p(D, c1, c2) + kp,
        V_neg(d, c1, c2) - (V_neg(D, c1, c2) + Kp + kp*(D - d)),
        V_pos_p(u, c1, c2) - km,
        V_pos_p(U, c1, c2) - km,
        V_pos(u, c1, c2) - (V_pos(U, c1, c2) + Km + km*(u - U)),
    ]

sol = fsolve(F, [-1, 1, -1, -0.1, 1, 0.1])
c1, c2, d, D, u, U = sol
print(f"d={d:.3f}, D={D:.3f}, u={u:.3f}, U={U:.3f}, c1={c1:.3f}, c2={c2:.3f}")

def V(x):
    if x <= d:  return V_neg(d, c1, c2) + kp*(d - x)
    if x <= 0:  return V_neg(x, c1, c2)
    if x <= u:  return V_pos(x, c1, c2)
    return V_pos(u, c1, c2) + km*(x - u)


# ──────────────────────────────────────────────
# Numerical grid
# ──────────────────────────────────────────────
x_min_num, x_max_num, Nx_num = -3, 3, 3001
x_grid = np.linspace(x_min_num, x_max_num, Nx_num)
dx     = x_grid[1] - x_grid[0]


# ──────────────────────────────────────────────
# Buffer collection & parameter estimation
# ──────────────────────────────────────────────
def collect_buffer(x_min, x_max, T, dt, batch_size, seed=None):
    rng     = np.random.default_rng(seed)
    n_steps = int(T / dt)
    buffer  = []
    sqrt_dt = np.sqrt(dt)

    X = rng.uniform(low=x_min, high=x_max, size=batch_size)

    for _ in range(n_steps):
        dW = rng.normal(loc=0.0, scale=sqrt_dt, size=batch_size)
        X1 = X + mu * dt + sigma * dW
        buffer.extend(zip(X, X1))
        # Uncomment the next line to use a rolling path instead of i.i.d. samples:
        # X = X1

    return buffer


def estimate_drift_and_diffusion_from_buffer(buffer, dt):
    if len(buffer) == 0:
        raise ValueError("Buffer is empty; cannot estimate parameters.")

    dX         = np.array([(x1 - x0) for (x0, x1) in buffer])
    N          = dX.shape[0]
    b_hat      = dX.sum() / (N * dt)
    sigma2_hat = (dX**2).sum() / (N * dt)
    sigma_hat  = float(np.sqrt(sigma2_hat))

    print("\n[Model-based baseline: parameter estimation]")
    print(f"  N           = {N:,}")
    print(f"  b_hat       = {b_hat: .6f}   (true b = {mu: .6f})")
    print(f"  sigma_hat   = {sigma_hat: .6f}   (true sigma = {sigma: .6f})\n")

    return b_hat, sigma_hat


buffer          = collect_buffer(x_min_num, x_max_num, T, dt, batch_size, seed)
mu_, sigma_     = estimate_drift_and_diffusion_from_buffer(buffer, dt)


# ──────────────────────────────────────────────
# Cost functions
# ──────────────────────────────────────────────
def running_cost(x):
    x = np.asarray(x)
    return np.where(x >= 0.0, h * x, -p * x)

def l_cost(xi):
    xi   = np.asarray(xi, dtype=float)
    cost = np.zeros_like(xi, dtype=float)
    pos  = xi > 0
    neg  = xi < 0
    cost[pos] = Kp + kp * xi[pos]
    cost[neg] = Km + km * (-xi[neg])
    return cost


# ──────────────────────────────────────────────
# Nonlocal impulse operator (xi-grid with extrapolation)
# ──────────────────────────────────────────────
def _extend_psi(psi, x_grid, slope_left=None, slope_right=None):
    psi    = np.asarray(psi,    dtype=float)
    x_grid = np.asarray(x_grid, dtype=float)

    if slope_left  is None: slope_left  = (psi[1]  - psi[0])  / (x_grid[1]  - x_grid[0])
    if slope_right is None: slope_right = (psi[-1] - psi[-2]) / (x_grid[-1] - x_grid[-2])

    x_min, x_max = x_grid[0], x_grid[-1]

    def psi_ext(z):
        z   = np.asarray(z, dtype=float)
        out = np.empty_like(z, dtype=float)
        mid   = (z >= x_min) & (z <= x_max)
        left  =  z <  x_min
        right =  z >  x_max
        if np.any(mid):   out[mid]   = np.interp(z[mid],   x_grid, psi)
        if np.any(left):  out[left]  = psi[0]  + slope_left  * (z[left]  - x_min)
        if np.any(right): out[right] = psi[-1] + slope_right * (z[right] - x_max)
        return out

    return psi_ext


def classical_N(psi, x_grid, xi_grid, l_cost_func,
                slope_left=None, slope_right=None):
    psi     = np.asarray(psi,     dtype=float)
    xi_grid = np.asarray(xi_grid, dtype=float)
    psi_ext = _extend_psi(psi, x_grid, slope_left, slope_right)

    Z      = x_grid[:, None] + xi_grid[None, :]   # (Nx, Nxi)
    psi_Z  = psi_ext(Z)
    l_vals = l_cost_func(xi_grid)
    return (psi_Z + l_vals[None, :]).min(axis=1)


# ──────────────────────────────────────────────
# Finite-difference coefficients
# ──────────────────────────────────────────────
def build_fd_coeffs(mu, sigma, r, x_grid):
    N  = len(x_grid)
    dx = x_grid[1] - x_grid[0]
    a  = 0.5 * sigma**2 / dx**2
    b  = mu  / (2.0 * dx)

    main  = np.zeros(N)
    lower = np.zeros(N - 1)
    upper = np.zeros(N - 1)
    for i in range(1, N - 1):
        lower[i-1] = a - b
        main[i]    = -2.0*a - r
        upper[i]   = a + b
    return lower, main, upper, dx


def solve_uncontrolled_value(mu, sigma, r, x_grid, kp, km):
    N  = len(x_grid)
    lower, main, upper, dx = build_fd_coeffs(mu, sigma, r, x_grid)
    f_vals = running_cost(x_grid)

    main_full  = main.copy()
    lower_full = lower.copy()
    upper_full = upper.copy()

    main_full[0]   = -1.0/dx;  upper_full[0]  = 1.0/dx
    lower_full[-1] = -1.0/dx;  main_full[-1]  = 1.0/dx

    A   = diags([lower_full, main_full, upper_full], offsets=[-1, 0, 1], format="csc")
    rhs = -f_vals.copy()
    rhs[0]  = -kp
    rhs[-1] =  km
    return spsolve(A, rhs)


# ──────────────────────────────────────────────
# Projected Gauss-Seidel sweep
# ──────────────────────────────────────────────
def gs_sweep(psi, Npsi, lower, main, upper, f_vals, dx, kp, km):
    N       = len(psi)
    psi_new = psi.copy()

    for i in range(1, N - 1):
        psi_pde    = -(f_vals[i] + lower[i-1]*psi_new[i-1] + upper[i]*psi[i+1]) / main[i]
        psi_new[i] = min(psi_pde, Npsi[i])

    psi_new[0]  = min(psi_new[1]  + kp*dx, Npsi[0])
    psi_new[-1] = min(psi_new[-2] + km*dx, Npsi[-1])
    return psi_new


# ──────────────────────────────────────────────
# Policy iteration
# ──────────────────────────────────────────────
def policy_iteration_classical(mu, sigma, r, x_grid,
                               Kp, kp, Km, km,
                               outer_iters=150,
                               inner_iters=150,
                               use_xi_grid=True,
                               verbose=True):
    x_grid = np.asarray(x_grid)
    f_vals = running_cost(x_grid)
    lower, main, upper, dx = build_fd_coeffs(mu, sigma, r, x_grid)

    psi = solve_uncontrolled_value(mu, sigma, r, x_grid, kp, km)

    if use_xi_grid:
        _xi_min, _xi_max, Nxi = -6.0, 6.0, 12001
        xi_grid_pi = np.linspace(_xi_min, _xi_max, Nxi)

    for n in range(1, outer_iters + 1):
        psi_old = psi.copy()

        if use_xi_grid:
            Npsi = classical_N(psi_old, x_grid, xi_grid_pi, l_cost)
        else:
            Npsi = impulse_operator(psi_old, x_grid, Kp, kp, Km, km)

        for _ in range(inner_iters):
            psi = gs_sweep(psi, Npsi, lower, main, upper, f_vals, dx, kp, km)

        diff = np.max(np.abs(psi - psi_old))
        if verbose and (n % 20 == 0 or n == 1):
            print(f"[Classical PI] iter {n:03d}, sup-norm diff = {diff:.3e}")

    return psi


# ──────────────────────────────────────────────
# Run
# ──────────────────────────────────────────────
psi_PI = policy_iteration_classical(mu_, sigma_, r,
                                    x_grid,
                                    Kp, kp, Km, km,
                                    outer_iters=outer_iters,
                                    inner_iters=inner_iters,
                                    use_xi_grid=True,
                                    verbose=True)

np.save("psi_PI.npy", psi_PI)

V_grid  = np.array([V(x) for x in x_grid])
sup_err = np.max(np.abs(psi_PI - V_grid))
print(f"Sup-norm error vs analytic classical V: {sup_err:.3e}")
