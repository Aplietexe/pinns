"""
Experiment 8: Loss Weighting Strategies
Test different loss weighting strategies for the three most challenging ODEs
"""

import sympy as sp
import torch
from improved_rnn_pinn import train_improved_rnn_pinn

DTYPE = torch.float64
x = sp.symbols("x")

print("EXPERIMENT 8: LOSS WEIGHTING STRATEGIES")
print("=" * 45)
print("Focus: Different loss weights for challenging ODEs")
print("Target: ~1e-14 loss (from direct optimization)")
print()

# Focus on the three most challenging ODEs
test_cases = [
    {
        "name": "Legendre (l=5)",
        "c_list": [5 * 6, -2 * x, 1 - x**2],
        "f_expr": sp.Integer(0),
        "bc_tuples": [(0.0, 0, 0.0), (1.0, 0, 1.0)],
        "N": 25,
        "x_left": -0.9,
        "x_right": 0.9,
        "expected": "First few should be [0,0,0,0,0,30,...]",
        "current_loss": 1.56e+00
    },
    {
        "name": "Beam",
        "c_list": [sp.Integer(0), sp.Integer(0), sp.Integer(0), sp.Integer(0), 1 + sp.Rational(3, 10) * x**2],
        "f_expr": sp.sin(2 * x),
        "bc_tuples": [(0.0, 0, 0.0), (0.0, 1, 0.0), (1.0, 2, 0.0), (1.0, 3, 0.0)],
        "N": 25,
        "x_left": 0.0,
        "x_right": 1.0,
        "expected": "First few should be [0,0,0,0,0.25,...]",
        "current_loss": 5.84e-01
    },
    {
        "name": "Hermite (n=6)",
        "c_list": [2 * 6, -2 * x, sp.Integer(1)],
        "f_expr": sp.Integer(0),
        "bc_tuples": [(0.0, 0, -120.0), (0.0, 1, 0.0)],
        "N": 25,
        "x_left": -0.9,
        "x_right": 0.9,
        "expected": "First few should be [-120, 0, 720, 0, -120,...]",
        "current_loss": 8.39e-03
    }
]

# Different loss weighting strategies
weighting_strategies = [
    {"name": "Baseline", "bc_weight": 100.0, "recurrence_weight": 100.0},
    {"name": "High Recurrence", "bc_weight": 100.0, "recurrence_weight": 1000.0},
    {"name": "High BC", "bc_weight": 1000.0, "recurrence_weight": 100.0},
    {"name": "Balanced High", "bc_weight": 500.0, "recurrence_weight": 500.0},
    {"name": "Very High Recurrence", "bc_weight": 100.0, "recurrence_weight": 10000.0},
]

for test in test_cases:
    print(f"\n{test['name']} (Current best: {test['current_loss']:.2e})")
    print("=" * 60)
    print(f"Expected: {test['expected']}")
    
    best_loss = float('inf')
    best_coeffs = None
    best_strategy = None
    
    for strategy in weighting_strategies:
        print(f"\n{strategy['name']} (BC:{strategy['bc_weight']}, Rec:{strategy['recurrence_weight']})")
        print("-" * 40)
        
        coeffs, final_loss = train_improved_rnn_pinn(
            test["c_list"],
            test["f_expr"], 
            test["bc_tuples"],
            N=test["N"],
            x_left=test["x_left"],
            x_right=test["x_right"],
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
        
    print(f"\nBEST STRATEGY: {best_strategy}")
    print(f"Best Loss: {best_loss:.2e} (vs previous {test['current_loss']:.2e})")
    print(f"Best Coefficients: {best_coeffs[:8]}")
    print()