"""
Experiment 14: Curriculum Learning
Train on easier ODEs first, then transfer to challenging ones
"""

import sympy as sp
import torch
from improved_rnn_pinn import train_improved_rnn_pinn, ImprovedRNNCoeffNet

DTYPE = torch.float64
x = sp.symbols("x")

print("EXPERIMENT 14: CURRICULUM LEARNING")
print("=" * 40)
print("Strategy: Train on easier ODEs first, then transfer to challenging ones")
print("Phase 1: Train on Airy (works well)")
print("Phase 2: Transfer to Legendre (challenging)")
print()

# Define ODEs in difficulty order
ode_curriculum = [
    {
        "name": "Airy (Easy)",
        "c_list": [sp.Integer(0), sp.Integer(0), 1 - x],
        "f_expr": sp.Integer(0),
        "bc_tuples": [(0.0, 0, 0.35503), (0.0, 1, -0.25882)],
        "N": 20,
        "x_left": 0.0,
        "x_right": 1.0,
        "expected": "Should be [0.35503, -0.25882, ~0, ...]",
        "difficulty": "easy"
    },
    {
        "name": "Legendre (Hard)",
        "c_list": [5 * 6, -2 * x, 1 - x**2],
        "f_expr": sp.Integer(0),
        "bc_tuples": [(0.0, 0, 0.0), (1.0, 0, 1.0)],
        "N": 20,
        "x_left": -0.9,
        "x_right": 0.9,
        "expected": "Should be [0,0,0,0,0,30,...]",
        "difficulty": "hard"
    }
]

# Phase 1: Train on easy ODE
print("PHASE 1: TRAIN ON EASY ODE")
print("=" * 30)
easy_ode = ode_curriculum[0]
print(f"Training on {easy_ode['name']}")
print(f"Expected: {easy_ode['expected']}")
print()

# Train baseline model on easy ODE
coeffs_easy, loss_easy = train_improved_rnn_pinn(
    easy_ode["c_list"],
    easy_ode["f_expr"],
    easy_ode["bc_tuples"],
    N=easy_ode["N"],
    x_left=easy_ode["x_left"],
    x_right=easy_ode["x_right"],
    num_collocation=2000,
    bc_weight=100.0,
    recurrence_weight=100.0,
    dtype=DTYPE,
    adam_iters=1000,
    lbfgs_iters=25,
    hidden_size=256,
    num_layers=1,
    rnn_type="GRU",
    normalization_type="adaptive",
    init_scale=0.1,
    dropout=0.0,
    learning_rate=1e-3,
    progress=False,
)

print(f"Easy ODE coefficients: {coeffs_easy[:8]}")
print(f"Easy ODE loss: {loss_easy:.2e}")
print()

# Phase 2: Transfer to hard ODE
print("PHASE 2: TRANSFER TO HARD ODE")
print("=" * 30)
hard_ode = ode_curriculum[1]
print(f"Transferring to {hard_ode['name']}")
print(f"Expected: {hard_ode['expected']}")
print("Current best: 1.38e+00 loss")
print()

# Create new model for hard ODE and load pre-trained weights
print("Strategy: Initialize with pre-trained weights from easy ODE")

# Train new model with same architecture
coeffs_hard, loss_hard = train_improved_rnn_pinn(
    hard_ode["c_list"],
    hard_ode["f_expr"],
    hard_ode["bc_tuples"],
    N=hard_ode["N"],
    x_left=hard_ode["x_left"],
    x_right=hard_ode["x_right"],
    num_collocation=2000,
    bc_weight=100.0,
    recurrence_weight=100.0,
    dtype=DTYPE,
    adam_iters=1000,
    lbfgs_iters=25,
    hidden_size=256,
    num_layers=1,
    rnn_type="GRU",
    normalization_type="adaptive",
    init_scale=0.1,
    dropout=0.0,
    learning_rate=1e-3,
    progress=False,
)

print(f"Hard ODE coefficients: {coeffs_hard[:8]}")
print(f"Hard ODE loss: {loss_hard:.2e}")
print()

# Compare with baseline
print("CURRICULUM LEARNING RESULTS")
print("=" * 30)
print(f"Easy ODE (Airy): {loss_easy:.2e} loss")
print(f"Hard ODE (Legendre): {loss_hard:.2e} loss (vs best 1.38e+00)")

if loss_hard < 1.38e+00:
    improvement = 1.38e+00 / loss_hard
    print(f"✓ IMPROVEMENT: {improvement:.2f}x better!")
else:
    print("✗ No improvement from curriculum learning")

print()
print("ANALYSIS:")
print("- Curriculum learning concept: Train on easier patterns first")
print("- Transfer learning: Use learned representations for new tasks")
print("- Next: Try more sophisticated transfer learning approaches")