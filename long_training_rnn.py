"""
Experiment 7: Long Training RNN - Focus on Generalization
Use the improved RNN approach with much longer training times
to see if we can push performance toward target levels.
"""

import sympy as sp
import torch
from improved_rnn_pinn import train_improved_rnn_pinn

DTYPE = torch.float64
x = sp.symbols("x")

print("EXPERIMENT 7: LONG TRAINING RNN")
print("=" * 35)
print("Focus: Generalization with extended training")
print("Target: ~1e-14 loss (from direct optimization)")
print()

# Test cases - focus on generalization
test_cases = [
    {
        "name": "Airy",
        "c_list": [-x, sp.Integer(0), sp.Integer(1)],
        "f_expr": sp.Integer(0),
        "bc_tuples": [(0.0, 0, 0.35502805388781723926), (0.0, 1, -0.25881940379280679840)],
        "N": 30,
        "x_left": -0.9,
        "x_right": 0.9,
        "expected": "First few should be [0.35503, -0.25882, ~0,...]"
    },
    {
        "name": "Hermite (n=6)",
        "c_list": [2 * 6, -2 * x, sp.Integer(1)],
        "f_expr": sp.Integer(0),
        "bc_tuples": [(0.0, 0, -120.0), (0.0, 1, 0.0)],
        "N": 30,
        "x_left": -0.9,
        "x_right": 0.9,
        "expected": "First few should be [-120, 0, 720, 0, -120,...]"
    },
    {
        "name": "Legendre (l=5)",
        "c_list": [5 * 6, -2 * x, 1 - x**2],
        "f_expr": sp.Integer(0),
        "bc_tuples": [(0.0, 0, 0.0), (1.0, 0, 1.0)],
        "N": 30,
        "x_left": -0.9,
        "x_right": 0.9,
        "expected": "First few should be [0,0,0,0,0,30,...]"
    },
    {
        "name": "Beam",
        "c_list": [sp.Integer(0), sp.Integer(0), sp.Integer(0), sp.Integer(0), 1 + sp.Rational(3, 10) * x**2],
        "f_expr": sp.sin(2 * x),
        "bc_tuples": [(0.0, 0, 0.0), (0.0, 1, 0.0), (1.0, 2, 0.0), (1.0, 3, 0.0)],
        "N": 30,
        "x_left": 0.0,
        "x_right": 1.0,
        "expected": "First few should be [0,0,0,0,0.25,...]"
    },
]

for test in test_cases:
    print(f"{test['name']}")
    print("-" * 50)
    print(f"Expected: {test['expected']}")
    
    # Use much longer training for better convergence
    coeffs = train_improved_rnn_pinn(
        test["c_list"],
        test["f_expr"], 
        test["bc_tuples"],
        N=test["N"],
        x_left=test["x_left"],
        x_right=test["x_right"],
        num_collocation=3000,  # Increased collocation points
        bc_weight=100.0,
        recurrence_weight=100.0,
        dtype=DTYPE,
        adam_iters=2000,  # Much longer Adam training
        lbfgs_iters=50,   # Much longer LBFGS training
        hidden_size=128,
        num_layers=1,
        rnn_type="GRU",
        normalization_type="adaptive",
        init_scale=0.1,
        dropout=0.0,
        learning_rate=1e-3,
        progress=False,  # Clean output
    )
    
    print(f"Actual coefficients: {coeffs[:8]}")
    print()
    
print("=" * 50)
print("ANALYSIS:")
print("All ODEs trained with:")
print("- 2000 Adam iterations (vs 300 before)")
print("- 50 LBFGS iterations (vs 10 before)")
print("- 3000 collocation points (vs 1000 before)")
print("- Adaptive normalization + proper initialization")
print("- Focus on generalization, not ODE-specific fixes")