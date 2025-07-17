"""
Experiment 10: Architectural Variations
Test different network architectures on the challenging ODEs
"""

import sympy as sp
import torch
from improved_rnn_pinn import train_improved_rnn_pinn

DTYPE = torch.float64
x = sp.symbols("x")

print("EXPERIMENT 10: ARCHITECTURAL VARIATIONS")
print("=" * 45)
print("Focus: Test different architectures on challenging ODEs")
print("Target: ~1e-14 loss (from direct optimization)")
print()

# Test cases - focus on the two most challenging
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
        "current_best": 1.56e+00
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
        "current_best": 5.84e-01
    },
]

# Test different architectures
architectures = [
    {"name": "Baseline", "hidden_size": 128, "num_layers": 1, "rnn_type": "GRU"},
    {"name": "Larger", "hidden_size": 256, "num_layers": 1, "rnn_type": "GRU"},
    {"name": "Deeper", "hidden_size": 128, "num_layers": 2, "rnn_type": "GRU"},
    {"name": "LSTM", "hidden_size": 128, "num_layers": 1, "rnn_type": "LSTM"},
    {"name": "Large LSTM", "hidden_size": 256, "num_layers": 1, "rnn_type": "LSTM"},
]

for test in test_cases:
    print(f"\n{test['name']} (Current best: {test['current_best']:.2e})")
    print("=" * 60)
    print(f"Expected: {test['expected']}")
    
    best_loss = float('inf')
    best_coeffs = None
    best_arch = None
    
    for arch in architectures:
        print(f"\n{arch['name']} ({arch['rnn_type']}, {arch['hidden_size']}x{arch['num_layers']})")
        print("-" * 30)
        
        try:
            coeffs, final_loss = train_improved_rnn_pinn(
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
                adam_iters=1000,
                lbfgs_iters=25,
                hidden_size=arch["hidden_size"],
                num_layers=arch["num_layers"],
                rnn_type=arch["rnn_type"],
                normalization_type="adaptive",
                init_scale=0.1,
                dropout=0.0,
                learning_rate=1e-3,
                progress=False,
            )
            
            print(f"Coefficients: {coeffs[:8]}")
            
            # Track best architecture
            if final_loss < best_loss:
                best_loss = final_loss
                best_coeffs = coeffs
                best_arch = arch["name"]
                
        except Exception as e:
            print(f"Error: {e}")
            continue
    
    print(f"\nBEST ARCHITECTURE: {best_arch}")
    print(f"Best Loss: {best_loss:.2e} (vs previous {test['current_best']:.2e})")
    print(f"Best Coefficients: {best_coeffs[:8]}")
    
    if best_loss < test["current_best"]:
        improvement = test["current_best"] / best_loss
        print(f"✓ IMPROVEMENT: {improvement:.2f}x better!")
    else:
        print("✗ No improvement")
    
    print()

print("=" * 60)
print("SUMMARY")
print("=" * 60)
print("Tested architectures:")
for arch in architectures:
    print(f"- {arch['name']}: {arch['rnn_type']} {arch['hidden_size']}x{arch['num_layers']}")
print()
print("Goal: Find architectures that work better for challenging ODEs")
print("Focus: Generalization over specific problem solving")