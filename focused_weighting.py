"""
Experiment 8: Focused Loss Weighting
Test different loss weighting strategies on Legendre equation specifically
"""

import sympy as sp
import torch
from improved_rnn_pinn import train_improved_rnn_pinn

DTYPE = torch.float64
x = sp.symbols("x")

print("EXPERIMENT 8: FOCUSED LOSS WEIGHTING")
print("=" * 40)
print("Focus: Legendre equation with different loss weights")
print("Target: ~1e-14 loss (from direct optimization)")
print("Current best: 1.56e+00 loss")
print()

# Focus on Legendre equation
c_list = [5 * 6, -2 * x, 1 - x**2]
f_expr = sp.Integer(0)
bc_tuples = [(0.0, 0, 0.0), (1.0, 0, 1.0)]
N = 25
x_left = -0.9
x_right = 0.9
expected = "First few should be [0,0,0,0,0,30,...]"

print(f"Expected: {expected}")
print()

# Test key weighting strategies
strategies = [
    {"name": "Baseline", "bc_weight": 100.0, "recurrence_weight": 100.0},
    {"name": "High Recurrence", "bc_weight": 100.0, "recurrence_weight": 1000.0},
    {"name": "Very High Recurrence", "bc_weight": 100.0, "recurrence_weight": 10000.0},
]

best_loss = float('inf')
best_coeffs = None
best_strategy = None

for strategy in strategies:
    print(f"{strategy['name']} (BC:{strategy['bc_weight']}, Rec:{strategy['recurrence_weight']})")
    print("-" * 50)
    
    coeffs, final_loss = train_improved_rnn_pinn(
        c_list,
        f_expr,
        bc_tuples,
        N=N,
        x_left=x_left,
        x_right=x_right,
        num_collocation=2000,
        bc_weight=strategy["bc_weight"],
        recurrence_weight=strategy["recurrence_weight"],
        dtype=DTYPE,
        adam_iters=1000,
        lbfgs_iters=25,
        hidden_size=128,
        num_layers=1,
        rnn_type="GRU",
        normalization_type="adaptive",
        init_scale=0.1,
        dropout=0.0,
        learning_rate=1e-3,
        progress=False,
    )
    
    print(f"Coefficients: {coeffs[:8]}")
    
    # Track best strategy
    if final_loss < best_loss:
        best_loss = final_loss
        best_coeffs = coeffs
        best_strategy = strategy["name"]
    
    print()

print("=" * 50)
print("RESULTS SUMMARY")
print("=" * 50)
print(f"Best Strategy: {best_strategy}")
print(f"Best Loss: {best_loss:.2e} (vs previous 1.56e+00)")
print(f"Best Coefficients: {best_coeffs[:8]}")
print()

# Check if we're moving toward the expected pattern
if best_loss < 1.56e+00:
    print("✓ IMPROVEMENT achieved with loss weighting!")
else:
    print("✗ No improvement with loss weighting")

print()
print("Expected pattern: [0,0,0,0,0,30,...]")
print("The key insight: Legendre polynomial has leading zeros,")
print("then a large jump at the 6th coefficient (30)")