"""
Experiment 17: Extreme Recurrence Weighting Optimization
Build on the breakthrough finding from simple_test.py
"""

import sympy as sp
import torch
from improved_rnn_pinn import train_improved_rnn_pinn

DTYPE = torch.float64
x = sp.symbols("x")

print("EXPERIMENT 17: EXTREME RECURRENCE WEIGHTING OPTIMIZATION")
print("=" * 55)
print("Building on breakthrough: Extreme recurrence weighting helps sparse patterns")
print("Strategy: Test on all challenging equations with optimized parameters")
print()

# Test cases - focus on challenging ones
test_cases = [
    {
        "name": "Legendre (l=5)",
        "c_list": [5 * 6, -2 * x, 1 - x**2],
        "f_expr": sp.Integer(0),
        "bc_tuples": [(0.0, 0, 0.0), (1.0, 0, 1.0)],
        "N": 20,
        "x_left": -0.9,
        "x_right": 0.9,
        "expected": "Should be [0,0,0,0,0,30,...]",
        "current_best": 1.38e+00,
        "simple_test": 9.93e-01
    },
    {
        "name": "Beam",
        "c_list": [sp.Integer(0), sp.Integer(0), sp.Integer(0), sp.Integer(0), 1 + sp.Rational(3, 10) * x**2],
        "f_expr": sp.sin(2 * x),
        "bc_tuples": [(0.0, 0, 0.0), (0.0, 1, 0.0), (1.0, 2, 0.0), (1.0, 3, 0.0)],
        "N": 20,
        "x_left": 0.0,
        "x_right": 1.0,
        "expected": "Should be [0,0,0,0,0.25,...]",
        "current_best": 5.84e-01,
        "simple_test": None
    }
]

print("Testing extreme recurrence weighting (10000x) with optimized parameters")
print()

for test in test_cases:
    print(f"{test['name']}")
    print("-" * 40)
    print(f"Expected: {test['expected']}")
    print(f"Current best: {test['current_best']:.2e}")
    if test['simple_test']:
        print(f"Simple test result: {test['simple_test']:.2e}")
    print()
    
    # Use optimized parameters based on simple test success
    coeffs, final_loss = train_improved_rnn_pinn(
        test["c_list"],
        test["f_expr"],
        test["bc_tuples"],
        N=test["N"],
        x_left=test["x_left"],
        x_right=test["x_right"],
        num_collocation=1000,
        bc_weight=1.0,  # Minimal BC weight
        recurrence_weight=10000.0,  # Extreme recurrence weight
        dtype=DTYPE,
        adam_iters=500,  # Longer training
        lbfgs_iters=15,
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
    print(f"Loss: {final_loss:.2e}")
    
    if final_loss < test["current_best"]:
        improvement = test["current_best"] / final_loss
        print(f"✓ IMPROVEMENT: {improvement:.2f}x better!")
    else:
        print("✗ No improvement")
    
    print()

print("EXTREME RECURRENCE WEIGHTING SUMMARY")
print("=" * 40)
print("Key insight: Extreme recurrence weighting (10000x) helps sparse patterns")
print("Strategy: Prioritize pattern learning over boundary condition fitting")
print("Next steps:")
print("1. Test with even longer training")
print("2. Try different extreme weight ratios")
print("3. Test on all 6 equations to check generalization")
print("4. Optimize for target <1e-10 loss")