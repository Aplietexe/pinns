import sympy as sp

import torch

# ========= DTYPE CONFIGURATION =========
# Set this to either torch.float32 or torch.float64
DTYPE = torch.float64
# =======================================

# x = sp.symbols("x")

# c = [sp.Integer(0), sp.Integer(0), sp.Integer(0), sp.Integer(1)]
# f = sp.cos(sp.pi * x)

# next_a = make_recurrence(c, f)

# a = [0.00000000e00, -3.00000000e00, 3.79735763e00]
# for _ in range(21 - 3):
#     a.append(next_a(torch.tensor(a)).item())

# print(np.array(a))

from pinn_trainer import train_power_series_pinn
from pinn_evaluate import solve_and_plot

# -------------------------------------------------------------------------
# Example: solve the third-order ODE  y'''(x) = cos(π x)
# subject to
#     y(0)   = 0,
#     y(1)   = cos(π) = -1,
#     y'(1)  = 1.
# This is exactly the problem tackled in `linear.py`.

# ODE definition  Σ c_k(x) y^{(k)}(x) = f(x)  with m = 3
x = sp.symbols("x")
c: list[sp.Expr] = [
    -x,  # coefficient of y
    sp.Integer(0),  # coefficient of y'
    sp.Integer(1),  # coefficient of y''
]
f = sp.Integer(0)


# Boundary conditions for Airy function Ai(x)
# Ai(0) ≈ 0.35502805388781723926
# Ai'(0) ≈ -0.25881940379280679840
Ai0 = 0.35502805388781723926
Aip0 = -0.25881940379280679840

bcs = [
    (0.0, 0, Ai0),  # y(0) = Ai(0)
    (0.0, 1, Aip0),  # y'(0) = Ai'(0)
]

# torch.autograd.set_detect_anomaly(True)

coeffs = train_power_series_pinn(
    c,
    f,
    bcs,
    N=20,
    recurrence_weight=0.0,
    bc_weight=100.0,
    adam_iters=10000,
    lbfgs_iters=200,
    num_collocation=5000,
    dtype=DTYPE,
    x_left=-2.0,
    x_right=2.0,
)
print(coeffs)
solve_and_plot(
    c, f, bcs, coeffs, file_prefix="airy_", dtype=DTYPE, x_left=-2.0, x_right=0
)
