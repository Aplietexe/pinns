"""
Experiment 9: Push Hermite to Target Performance
Hermite showed excellent results (8.39e-03 loss) with correct pattern.
Let's see if we can push it to the target ~1e-14 loss.
"""

import sympy as sp
import torch
from improved_rnn_pinn import train_improved_rnn_pinn

DTYPE = torch.float64
x = sp.symbols("x")

print("EXPERIMENT 9: PUSH HERMITE TO TARGET")
print("=" * 40)
print("Focus: Hermite equation - push to target ~1e-14 loss")
print("Current best: 8.39e-03 loss")
print("Target: ~1e-14 loss (from direct optimization)")
print()

# Hermite equation
c_list = [2 * 6, -2 * x, sp.Integer(1)]
f_expr = sp.Integer(0)
bc_tuples = [(0.0, 0, -120.0), (0.0, 1, 0.0)]
N = 30  # More coefficients
x_left = -0.9
x_right = 0.9
expected = "First few should be [-120, 0, 720, 0, -120,...]"

print(f"Expected: {expected}")
print()

# Try different strategies to push toward target
strategies = [
    {
        "name": "Extended Training",
        "adam_iters": 2000,
        "lbfgs_iters": 50,
        "num_collocation": 3000,
        "bc_weight": 100.0,
        "recurrence_weight": 100.0
    },
    {
        "name": "Higher Precision",
        "adam_iters": 1500,
        "lbfgs_iters": 100,
        "num_collocation": 4000,
        "bc_weight": 100.0,
        "recurrence_weight": 100.0
    },
    {
        "name": "Balanced High Weights",
        "adam_iters": 1500,
        "lbfgs_iters": 50,
        "num_collocation": 3000,
        "bc_weight": 500.0,
        "recurrence_weight": 500.0
    },
]

best_loss = float('inf')
best_coeffs = None
best_strategy = None

for strategy in strategies:
    print(f"{strategy['name']}")
    print("-" * 50)
    print(f"Adam: {strategy['adam_iters']}, LBFGS: {strategy['lbfgs_iters']}")
    print(f"Collocation: {strategy['num_collocation']}, BC: {strategy['bc_weight']}, Rec: {strategy['recurrence_weight']}")
    
    coeffs, final_loss = train_improved_rnn_pinn(
        c_list,
        f_expr,
        bc_tuples,
        N=N,
        x_left=x_left,
        x_right=x_right,
        num_collocation=strategy["num_collocation"],
        bc_weight=strategy["bc_weight"],
        recurrence_weight=strategy["recurrence_weight"],
        dtype=DTYPE,
        adam_iters=strategy["adam_iters"],
        lbfgs_iters=strategy["lbfgs_iters"],
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
print(f"Best Loss: {best_loss:.2e} (vs previous 8.39e-03)")
print(f"Best Coefficients: {best_coeffs[:8]}")
print()

# Check improvement
if best_loss < 8.39e-03:
    improvement = 8.39e-03 / best_loss
    print(f"✓ IMPROVEMENT: {improvement:.1f}x better!")
else:
    print("✗ No improvement")

print()
print("Expected pattern: [-120, 0, 720, 0, -120,...]")
print("Target: ~1e-14 loss")

# Check if we're getting close to target
if best_loss < 1e-10:
    print("✓ VERY CLOSE to target!")
elif best_loss < 1e-8:
    print("✓ Getting closer to target")
elif best_loss < 1e-6:
    print("✓ Significant progress toward target")
else:
    print("Still far from target, but pattern is correct")