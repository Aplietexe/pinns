"""
Quick baseline test for direct parameter optimization
"""

import sympy as sp
import torch
from baseline_direct import train_direct_parameters

DTYPE = torch.float64
x = sp.symbols("x")

print("QUICK BASELINE TEST")
print("=" * 30)

# Test just Airy equation with fewer iterations
print("\nAiry equation (quick test)")
print("-" * 30)

c_airy = [-x, sp.Integer(0), sp.Integer(1)]
f_airy = sp.Integer(0)
Ai0 = 0.35502805388781723926
Aip0 = -0.25881940379280679840
bcs_airy = [(0.0, 0, Ai0), (0.0, 1, Aip0)]

coeffs_airy = train_direct_parameters(
    c_airy,
    f_airy,
    bcs_airy,
    N=20,
    x_left=-0.5,
    x_right=0.5,
    num_collocation=1000,
    bc_weight=100.0,
    recurrence_weight=100.0,
    dtype=DTYPE,
    adam_iters=200,
    lbfgs_iters=5,
    progress=True,
)

print(f"✓ Airy baseline completed")
print(f"First few coefficients: {coeffs_airy[:5]}")