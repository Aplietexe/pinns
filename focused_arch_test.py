"""
Experiment 10: Focused Architecture Test
Test key architectures on Legendre equation
"""

import sympy as sp
import torch
from improved_rnn_pinn import train_improved_rnn_pinn

DTYPE = torch.float64
x = sp.symbols("x")

print("EXPERIMENT 10: FOCUSED ARCHITECTURE TEST")
print("=" * 45)
print("Focus: Test key architectures on Legendre equation")
print("Current best: 1.56e+00 loss")
print("Target: ~1e-14 loss (from direct optimization)")
print()

# Legendre equation
c_list = [5 * 6, -2 * x, 1 - x**2]
f_expr = sp.Integer(0)
bc_tuples = [(0.0, 0, 0.0), (1.0, 0, 1.0)]
N = 25
x_left = -0.9
x_right = 0.9
expected = "First few should be [0,0,0,0,0,30,...]"

print(f"Expected: {expected}")
print()

# Test key architectures
architectures = [
    {"name": "Baseline GRU", "hidden_size": 128, "num_layers": 1, "rnn_type": "GRU"},
    {"name": "Large GRU", "hidden_size": 256, "num_layers": 1, "rnn_type": "GRU"},
    {"name": "Baseline LSTM", "hidden_size": 128, "num_layers": 1, "rnn_type": "LSTM"},
]

best_loss = float('inf')
best_coeffs = None
best_arch = None

for arch in architectures:
    print(f"{arch['name']} ({arch['rnn_type']}, {arch['hidden_size']}x{arch['num_layers']})")
    print("-" * 50)
    
    coeffs, final_loss = train_improved_rnn_pinn(
        c_list,
        f_expr,
        bc_tuples,
        N=N,
        x_left=x_left,
        x_right=x_right,
        num_collocation=2000,
        bc_weight=100.0,
        recurrence_weight=100.0,
        dtype=DTYPE,
        adam_iters=1000,
        lbfgs_iters=25,
        hidden_size=arch["hidden_size"],
        num_layers=arch["num_layers"],
        rnn_type=arch["rnn_type"],
        normalization_type="adaptive",
        init_scale=0.1,
        dropout=0.0,
        learning_rate=1e-3,
        progress=False,
    )
    
    print(f"Coefficients: {coeffs[:8]}")
    
    # Track best architecture
    if final_loss < best_loss:
        best_loss = final_loss
        best_coeffs = coeffs
        best_arch = arch["name"]
    
    print()

print("=" * 50)
print("RESULTS")
print("=" * 50)
print(f"Best Architecture: {best_arch}")
print(f"Best Loss: {best_loss:.2e} (vs previous 1.56e+00)")
print(f"Best Coefficients: {best_coeffs[:8]}")
print()

if best_loss < 1.56e+00:
    improvement = 1.56e+00 / best_loss
    print(f"✓ IMPROVEMENT: {improvement:.2f}x better!")
else:
    print("✗ No improvement")

print()
print("Expected pattern: [0,0,0,0,0,30,...]")
print("Analysis: Legendre has mostly zero coefficients until the 6th term")
print("This sparse pattern may be challenging for autoregressive prediction")