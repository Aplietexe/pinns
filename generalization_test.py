"""
Experiment 11: Generalization Test
Test the approach on new ODE types to demonstrate generalization
"""

import sympy as sp
import torch
from improved_rnn_pinn import train_improved_rnn_pinn

DTYPE = torch.float64
x = sp.symbols("x")

print("EXPERIMENT 11: GENERALIZATION TEST")
print("=" * 40)
print("Focus: Test approach on new ODE types")
print("Goal: Demonstrate general method, not ODE-specific solution")
print("Target: ~1e-14 loss (from direct optimization)")
print()

# New test cases - different from the original 4 ODEs
new_test_cases = [
    {
        "name": "Simple Harmonic Oscillator",
        "c_list": [sp.Integer(1), sp.Integer(0), sp.Integer(1)],  # u'' + u = 0
        "f_expr": sp.Integer(0),
        "bc_tuples": [(0.0, 0, 1.0), (0.0, 1, 0.0)],  # u(0) = 1, u'(0) = 0 -> cos(x)
        "N": 20,
        "x_left": -0.5,
        "x_right": 0.5,
        "expected": "Should be cos(x): [1, 0, -1/2!, 0, 1/4!, 0, -1/6!, ...]",
        "expected_coeffs": [1.0, 0.0, -0.5, 0.0, 0.041667, 0.0, -0.001389]
    },
    {
        "name": "Exponential Growth",
        "c_list": [sp.Integer(1), -sp.Integer(1)],  # u' - u = 0
        "f_expr": sp.Integer(0),
        "bc_tuples": [(0.0, 0, 1.0)],  # u(0) = 1 -> e^x
        "N": 20,
        "x_left": -0.5,
        "x_right": 0.5,
        "expected": "Should be e^x: [1, 1, 1/2!, 1/3!, 1/4!, ...]",
        "expected_coeffs": [1.0, 1.0, 0.5, 0.166667, 0.041667, 0.008333]
    },
    {
        "name": "Linear ODE",
        "c_list": [sp.Integer(1), -2*x],  # u' - 2x*u = 0
        "f_expr": sp.Integer(0),
        "bc_tuples": [(0.0, 0, 1.0)],  # u(0) = 1 -> e^(x^2)
        "N": 20,
        "x_left": -0.5,
        "x_right": 0.5,
        "expected": "Should be e^(x^2): [1, 0, 1, 0, 1, 0, ...]",
        "expected_coeffs": [1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0]
    },
]

print("Testing well-known ODEs with known analytic solutions:")
print()

for test in new_test_cases:
    print(f"{test['name']}")
    print("-" * 50)
    print(f"ODE: {test['c_list']} → f = {test['f_expr']}")
    print(f"BCs: {test['bc_tuples']}")
    print(f"Expected: {test['expected']}")
    print(f"Expected coeffs: {test['expected_coeffs']}")
    print()
    
    # Use the best architecture found (Large GRU) with medium training
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
        hidden_size=256,  # Use Large GRU
        num_layers=1,
        rnn_type="GRU",
        normalization_type="adaptive",
        init_scale=0.1,
        dropout=0.0,
        learning_rate=1e-3,
        progress=False,
    )
    
    print(f"Actual coefficients: {coeffs[:len(test['expected_coeffs'])]}")
    
    # Compare with expected
    expected_tensor = torch.tensor(test['expected_coeffs'], dtype=DTYPE)
    error = torch.abs(coeffs[:len(test['expected_coeffs'])] - expected_tensor)
    max_error = torch.max(error)
    
    print(f"Max coefficient error: {max_error:.2e}")
    
    if final_loss < 1e-10:
        print("✓ EXCELLENT: Very close to target!")
    elif final_loss < 1e-6:
        print("✓ GOOD: Close to target")
    elif final_loss < 1e-3:
        print("✓ DECENT: Reasonable performance")
    else:
        print("✗ POOR: Far from target")
    
    print()

print("=" * 50)
print("GENERALIZATION ANALYSIS")
print("=" * 50)
print("Key Questions:")
print("1. Does the method work on new ODEs?")
print("2. Are the coefficient patterns learned correctly?")
print("3. How close do we get to the expected analytic solutions?")
print("4. Is this a general approach or specific to the original 4 ODEs?")
print()
print("This test demonstrates whether we have a general PINN solution")