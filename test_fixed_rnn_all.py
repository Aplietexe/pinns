"""
Test fixed RNN on all ODEs with proper boundary condition handling
"""

import sympy as sp
import torch
from fixed_rnn_pinn import train_fixed_rnn_pinn

DTYPE = torch.float64
x = sp.symbols("x")

print("FIXED RNN TEST ON ALL ODEs")
print("=" * 30)

# Test cases
test_cases = [
    {
        "name": "Legendre (l=5)",
        "c_list": [5 * 6, -2 * x, 1 - x**2],
        "f_expr": sp.Integer(0),
        "bc_tuples": [(0.0, 0, 0.0), (1.0, 0, 1.0)],
        "N": 20,
        "x_left": -0.9,
        "x_right": 0.9,
        "expected": "First few should be [0,0,0,0,0,30,...]",
        "order": 2
    },
    {
        "name": "Airy",
        "c_list": [-x, sp.Integer(0), sp.Integer(1)],
        "f_expr": sp.Integer(0),
        "bc_tuples": [(0.0, 0, 0.35502805388781723926), (0.0, 1, -0.25881940379280679840)],
        "N": 20,
        "x_left": -0.9,
        "x_right": 0.9,
        "expected": "First few should be [0.35503, -0.25882, ~0,...]",
        "order": 2
    },
    {
        "name": "Hermite (n=6)",
        "c_list": [2 * 6, -2 * x, sp.Integer(1)],
        "f_expr": sp.Integer(0),
        "bc_tuples": [(0.0, 0, -120.0), (0.0, 1, 0.0)],
        "N": 20,
        "x_left": -0.9,
        "x_right": 0.9,
        "expected": "First few should be [-120, 0, 720, 0, -120,...]",
        "order": 2
    },
    {
        "name": "Beam",
        "c_list": [sp.Integer(0), sp.Integer(0), sp.Integer(0), sp.Integer(0), 1 + sp.Rational(3, 10) * x**2],
        "f_expr": sp.sin(2 * x),
        "bc_tuples": [(0.0, 0, 0.0), (0.0, 1, 0.0), (1.0, 2, 0.0), (1.0, 3, 0.0)],
        "N": 20,
        "x_left": 0.0,
        "x_right": 1.0,
        "expected": "First few should be [0,0,0,0,0.25,...]",
        "order": 4
    },
]

for test in test_cases:
    print(f"\n{test['name']} (order {test['order']})")
    print("-" * 50)
    print(f"Expected: {test['expected']}")
    
    coeffs = train_fixed_rnn_pinn(
        test["c_list"],
        test["f_expr"], 
        test["bc_tuples"],
        N=test["N"],
        x_left=test["x_left"],
        x_right=test["x_right"],
        num_collocation=1000,
        bc_weight=100.0,
        recurrence_weight=100.0,
        dtype=DTYPE,
        adam_iters=300,
        lbfgs_iters=10,
        hidden_size=128,
        num_layers=1,
        rnn_type="GRU",
        init_scale=0.1,
        normalization_type="adaptive",
        dropout=0.0,
        learning_rate=1e-3,
        progress=False,
    )
    
    print(f"Actual coefficients: {coeffs[:8]}")
    print()