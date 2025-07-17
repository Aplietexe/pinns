"""
Simple focused test: Pattern-only loss for Legendre equation
"""

import sympy as sp
import torch
from improved_rnn_pinn import train_improved_rnn_pinn

DTYPE = torch.float64
x = sp.symbols("x")

print("SIMPLE TEST: PATTERN-ONLY LOSS")
print("=" * 30)
print("Focus: Legendre equation with extreme recurrence weighting")
print("Expected: [0,0,0,0,0,30,...]")
print("Current best: 1.38e+00 loss")
print()

# Quick test with extreme recurrence weighting
coeffs, final_loss = train_improved_rnn_pinn(
    [5 * 6, -2 * x, 1 - x**2],  # Legendre
    sp.Integer(0),
    [(0.0, 0, 0.0), (1.0, 0, 1.0)],
    N=10,  # Very small for speed
    x_left=-0.9,
    x_right=0.9,
    num_collocation=500,  # Minimal
    bc_weight=1.0,  # Minimal BC weight
    recurrence_weight=10000.0,  # Extreme recurrence weight
    dtype=DTYPE,
    adam_iters=200,  # Minimal iterations
    lbfgs_iters=5,
    hidden_size=64,  # Smaller network
    num_layers=1,
    rnn_type="GRU",
    normalization_type="adaptive",
    init_scale=0.1,
    dropout=0.0,
    learning_rate=1e-3,
    progress=True,
)

print(f"Coefficients: {coeffs}")
print(f"Loss: {final_loss:.2e}")

if final_loss < 1.38e+00:
    improvement = 1.38e+00 / final_loss
    print(f"✓ IMPROVEMENT: {improvement:.2f}x better!")
else:
    print("✗ No improvement")

print()
print("Pattern analysis:")
print("- Are first 5 coefficients close to zero?")
print("- Does coefficient 6 have the right sign/magnitude?")
print("- This extreme test helps identify if the approach can work at all")