"""
Test the small MLP architecture on all ODEs
"""

import sympy as sp
import torch
from mlp_experiment import train_mlp_pinn

DTYPE = torch.float64
x = sp.symbols("x")

print("TESTING SMALL MLP ON ALL ODEs")
print("=" * 40)

# Test cases
test_cases = [
    {
        "name": "Legendre (l=5)",
        "c_list": [5 * 6, -2 * x, 1 - x**2],
        "f_expr": sp.Integer(0),
        "bc_tuples": [(0.0, 0, 0.0), (1.0, 0, 1.0)],
        "N": 30,
        "x_left": -0.9,
        "x_right": 0.9,
    },
    {
        "name": "Airy",
        "c_list": [-x, sp.Integer(0), sp.Integer(1)],
        "f_expr": sp.Integer(0),
        "bc_tuples": [(0.0, 0, 0.35502805388781723926), (0.0, 1, -0.25881940379280679840)],
        "N": 40,
        "x_left": -0.9,
        "x_right": 0.9,
    },
    {
        "name": "Hermite (n=6)",
        "c_list": [2 * 6, -2 * x, sp.Integer(1)],
        "f_expr": sp.Integer(0),
        "bc_tuples": [(0.0, 0, -120.0), (0.0, 1, 0.0)],
        "N": 30,
        "x_left": -0.9,
        "x_right": 0.9,
    },
    {
        "name": "Beam",
        "c_list": [sp.Integer(0), sp.Integer(0), sp.Integer(0), sp.Integer(0), 1 + sp.Rational(3, 10) * x**2],
        "f_expr": sp.sin(2 * x),
        "bc_tuples": [(0.0, 0, 0.0), (0.0, 1, 0.0), (1.0, 2, 0.0), (1.0, 3, 0.0)],
        "N": 40,
        "x_left": 0.0,
        "x_right": 1.0,
    },
]

for test in test_cases:
    print(f"\n{test['name']}")
    print("-" * 40)
    
    coeffs = train_mlp_pinn(
        test["c_list"],
        test["f_expr"], 
        test["bc_tuples"],
        N=test["N"],
        x_left=test["x_left"],
        x_right=test["x_right"],
        num_collocation=5000,
        bc_weight=100.0,
        recurrence_weight=100.0,
        dtype=DTYPE,
        adam_iters=1000,
        lbfgs_iters=15,
        hidden_dims=[128, 128],  # Small MLP that worked best
        dropout=0.0,
        progress=True,
    )
    
    print(f"✓ {test['name']} completed")
    print(f"First few coefficients: {coeffs[:5]}")
    print()