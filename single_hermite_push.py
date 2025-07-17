"""
Experiment 9: Single Hermite Push
Test one extended training strategy on Hermite equation
"""

import sympy as sp
import torch
from improved_rnn_pinn import train_improved_rnn_pinn

DTYPE = torch.float64
x = sp.symbols("x")

print("EXPERIMENT 9: SINGLE HERMITE PUSH")
print("=" * 35)
print("Focus: Hermite equation - extended training")
print("Current best: 8.39e-03 loss")
print("Target: ~1e-14 loss (from direct optimization)")
print()

# Hermite equation
c_list = [2 * 6, -2 * x, sp.Integer(1)]
f_expr = sp.Integer(0)
bc_tuples = [(0.0, 0, -120.0), (0.0, 1, 0.0)]
N = 30
x_left = -0.9
x_right = 0.9
expected = "First few should be [-120, 0, 720, 0, -120,...]"

print(f"Expected: {expected}")
print()

print("Extended Training Strategy:")
print("- Adam: 1500 iterations")
print("- LBFGS: 50 iterations")
print("- Collocation: 3000 points")
print("- Balanced weights: BC=500, Rec=500")
print()

coeffs, final_loss = train_improved_rnn_pinn(
    c_list,
    f_expr,
    bc_tuples,
    N=N,
    x_left=x_left,
    x_right=x_right,
    num_collocation=3000,
    bc_weight=500.0,
    recurrence_weight=500.0,
    dtype=DTYPE,
    adam_iters=1500,
    lbfgs_iters=50,
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
print()

print("=" * 40)
print("RESULTS")
print("=" * 40)
print(f"Previous best: 8.39e-03")
print(f"Current loss: {final_loss:.2e}")

if final_loss < 8.39e-03:
    improvement = 8.39e-03 / final_loss
    print(f"✓ IMPROVEMENT: {improvement:.1f}x better!")
else:
    print("✗ No improvement")

print()
print("Expected pattern: [-120, 0, 720, 0, -120,...]")
print("Target: ~1e-14 loss")

# Check if we're getting close to target
if final_loss < 1e-10:
    print("✓ VERY CLOSE to target!")
elif final_loss < 1e-8:
    print("✓ Getting closer to target")
elif final_loss < 1e-6:
    print("✓ Significant progress toward target")
else:
    print("Still far from target, but pattern is likely correct")