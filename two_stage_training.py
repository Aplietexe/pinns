"""
Experiment 15: Two-Stage Training
Stage 1: Learn coefficient pattern (zeros/non-zeros)
Stage 2: Refine coefficient magnitudes
"""

import sympy as sp
import torch
from improved_rnn_pinn import train_improved_rnn_pinn

DTYPE = torch.float64
x = sp.symbols("x")

print("EXPERIMENT 15: TWO-STAGE TRAINING")
print("=" * 40)
print("Strategy: First learn pattern, then refine magnitudes")
print("Stage 1: Train with pattern-focused loss")
print("Stage 2: Train with magnitude-focused loss")
print()

# Focus on Legendre equation
print("TESTING ON LEGENDRE EQUATION")
print("=" * 30)
c_list = [5 * 6, -2 * x, 1 - x**2]
f_expr = sp.Integer(0)
bc_tuples = [(0.0, 0, 0.0), (1.0, 0, 1.0)]
N = 15  # Reduced for faster testing
x_left = -0.9
x_right = 0.9
expected = "Should be [0,0,0,0,0,30,...]"

print(f"Expected: {expected}")
print("Current best: 1.38e+00 loss")
print()

# Stage 1: Pattern-focused training
print("STAGE 1: PATTERN-FOCUSED TRAINING")
print("=" * 35)
print("Focus: Learn which coefficients should be zero")

# Train with higher recurrence weight to emphasize pattern
coeffs_stage1, loss_stage1 = train_improved_rnn_pinn(
    c_list,
    f_expr,
    bc_tuples,
    N=N,
    x_left=x_left,
    x_right=x_right,
    num_collocation=1000,  # Reduced for speed
    bc_weight=100.0,
    recurrence_weight=1000.0,  # Higher to emphasize pattern
    dtype=DTYPE,
    adam_iters=300,  # Reduced for speed
    lbfgs_iters=10,
    hidden_size=128,
    num_layers=1,
    rnn_type="GRU",
    normalization_type="adaptive",
    init_scale=0.1,
    dropout=0.0,
    learning_rate=1e-3,
    progress=True,
)

print(f"Stage 1 coefficients: {coeffs_stage1[:8]}")
print(f"Stage 1 loss: {loss_stage1:.2e}")
print()

# Stage 2: Magnitude-focused training
print("STAGE 2: MAGNITUDE-FOCUSED TRAINING")
print("=" * 35)
print("Focus: Refine coefficient magnitudes")

# Train with balanced weights to refine magnitudes
coeffs_stage2, loss_stage2 = train_improved_rnn_pinn(
    c_list,
    f_expr,
    bc_tuples,
    N=N,
    x_left=x_left,
    x_right=x_right,
    num_collocation=1000,
    bc_weight=100.0,
    recurrence_weight=100.0,  # Balanced for magnitude refinement
    dtype=DTYPE,
    adam_iters=300,
    lbfgs_iters=10,
    hidden_size=128,
    num_layers=1,
    rnn_type="GRU",
    normalization_type="adaptive",
    init_scale=0.1,
    dropout=0.0,
    learning_rate=1e-4,  # Lower learning rate for refinement
    progress=True,
)

print(f"Stage 2 coefficients: {coeffs_stage2[:8]}")
print(f"Stage 2 loss: {loss_stage2:.2e}")
print()

# Compare results
print("TWO-STAGE TRAINING RESULTS")
print("=" * 30)
print(f"Stage 1 (pattern): {loss_stage1:.2e} loss")
print(f"Stage 2 (magnitude): {loss_stage2:.2e} loss")
print(f"Current best: 1.38e+00 loss")

best_loss = min(loss_stage1, loss_stage2)
if best_loss < 1.38e+00:
    improvement = 1.38e+00 / best_loss
    print(f"✓ IMPROVEMENT: {improvement:.2f}x better!")
else:
    print("✗ No improvement from two-stage training")

print()
print("ANALYSIS:")
print("- Two-stage training: Separate pattern learning from magnitude refinement")
print("- Stage 1: Higher recurrence weight emphasizes pattern consistency")
print("- Stage 2: Lower learning rate for fine-tuning magnitudes")
print("- Next: Try different loss formulations or architectural approaches")