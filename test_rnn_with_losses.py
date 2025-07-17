"""
Test RNN approach on all ODEs with loss tracking
"""

import sympy as sp
import torch
from rnn_pinn import train_rnn_pinn

DTYPE = torch.float64
x = sp.symbols("x")

print("RNN TEST WITH LOSS TRACKING")
print("=" * 35)

# Expected target performance from direct optimization: ~1e-14
print("Target performance: ~1e-14 loss (from direct optimization)")
print()

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
        "expected_coeffs": "First few should be [0,0,0,0,0,30,...]"
    },
    {
        "name": "Airy",
        "c_list": [-x, sp.Integer(0), sp.Integer(1)],
        "f_expr": sp.Integer(0),
        "bc_tuples": [(0.0, 0, 0.35502805388781723926), (0.0, 1, -0.25881940379280679840)],
        "N": 20,
        "x_left": -0.9,
        "x_right": 0.9,
        "expected_coeffs": "First few should be [0.35503, -0.25882, ~0,...]"
    },
    {
        "name": "Hermite (n=6)",
        "c_list": [2 * 6, -2 * x, sp.Integer(1)],
        "f_expr": sp.Integer(0),
        "bc_tuples": [(0.0, 0, -120.0), (0.0, 1, 0.0)],
        "N": 20,
        "x_left": -0.9,
        "x_right": 0.9,
        "expected_coeffs": "First few should be [-120, 0, 720, 0, -120,...]"
    },
    {
        "name": "Beam",
        "c_list": [sp.Integer(0), sp.Integer(0), sp.Integer(0), sp.Integer(0), 1 + sp.Rational(3, 10) * x**2],
        "f_expr": sp.sin(2 * x),
        "bc_tuples": [(0.0, 0, 0.0), (0.0, 1, 0.0), (1.0, 2, 0.0), (1.0, 3, 0.0)],
        "N": 20,
        "x_left": 0.0,
        "x_right": 1.0,
        "expected_coeffs": "First few should be [0,0,0,0,0.25,...]"
    },
]

results = []

for test in test_cases:
    print(f"{test['name']}")
    print("-" * 40)
    print(f"Expected: {test['expected_coeffs']}")
    
    # Use GRU with more iterations for better convergence
    coeffs = train_rnn_pinn(
        test["c_list"],
        test["f_expr"], 
        test["bc_tuples"],
        N=test["N"],
        x_left=test["x_left"],
        x_right=test["x_right"],
        num_collocation=2000,
        bc_weight=100.0,
        recurrence_weight=100.0,
        dtype=DTYPE,
        adam_iters=500,  # Increased from 200
        lbfgs_iters=10,  # Increased from 5
        hidden_size=128,
        num_layers=1,
        rnn_type="GRU",
        init_scale=0.1,
        use_normalization=True,
        dropout=0.0,
        learning_rate=1e-3,
        progress=False,  # Clean output to see losses clearly
    )
    
    print(f"Actual coefficients: {coeffs[:5]}")
    print()
    
    # Store results for journal
    results.append({
        "name": test["name"],
        "coeffs": coeffs[:5].tolist(),
        "expected": test["expected_coeffs"]
    })

# Print summary
print("=" * 50)
print("SUMMARY")
print("=" * 50)
for result in results:
    print(f"{result['name']}: {result['coeffs']}")