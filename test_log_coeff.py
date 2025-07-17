"""
Quick test of logarithmic coefficient representation on Legendre equation
"""

import sympy as sp
import torch
from log_coeff_rnn import train_log_coeff_rnn

DTYPE = torch.float64
x = sp.symbols("x")

print("QUICK TEST: LOGARITHMIC COEFFICIENT REPRESENTATION")
print("=" * 50)
print("Testing on Legendre equation only")
print("Current best: 1.38e+00 loss")
print()

# Legendre equation - most challenging
c_list = [5 * 6, -2 * x, 1 - x**2]
f_expr = sp.Integer(0)
bc_tuples = [(0.0, 0, 0.0), (1.0, 0, 1.0)]
N = 15  # Reduced for faster testing
x_left = -0.9
x_right = 0.9

print(f"Expected: First few should be [0,0,0,0,0,30,...]")
print("Testing with reduced parameters for speed...")
print()

coeffs, final_loss = train_log_coeff_rnn(
    c_list,
    f_expr,
    bc_tuples,
    N=N,
    x_left=x_left,
    x_right=x_right,
    num_collocation=1000,  # Reduced
    bc_weight=100.0,
    recurrence_weight=100.0,
    dtype=DTYPE,
    adam_iters=300,  # Reduced
    lbfgs_iters=10,  # Reduced
    hidden_size=128,  # Reduced
    num_layers=1,
    rnn_type="GRU",
    init_scale=0.1,
    dropout=0.0,
    learning_rate=1e-3,
    progress=True,
)

print(f"Coefficients: {coeffs[:8]}")
print(f"Current best: 1.38e+00")

if final_loss < 1.38e+00:
    improvement = 1.38e+00 / final_loss
    print(f"✓ IMPROVEMENT: {improvement:.2f}x better!")
else:
    print("✗ No improvement")

print()
print("Next: If promising, test on other equations")