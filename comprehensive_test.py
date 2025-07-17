"""
Comprehensive test of extreme recurrence weighting on all 6 equations
"""

import sympy as sp
import torch
from improved_rnn_pinn import train_improved_rnn_pinn

DTYPE = torch.float64
x = sp.symbols("x")

print("COMPREHENSIVE TEST: EXTREME RECURRENCE WEIGHTING")
print("=" * 50)
print("Testing on all 6 equations with extreme recurrence weighting")
print("Target: <1e-10 loss on all equations")
print()

# All 6 equations
equations = [
    {
        "name": "Legendre (l=5)",
        "c_list": [5 * 6, -2 * x, 1 - x**2],
        "f_expr": sp.Integer(0),
        "bc_tuples": [(0.0, 0, 0.0), (1.0, 0, 1.0)],
        "N": 20,
        "x_left": -0.9,
        "x_right": 0.9,
        "expected": "[0,0,0,0,0,30,...]",
        "current_best": 1.38e+00
    },
    {
        "name": "Airy",
        "c_list": [sp.Integer(0), sp.Integer(0), 1 - x],
        "f_expr": sp.Integer(0),
        "bc_tuples": [(0.0, 0, 0.35503), (0.0, 1, -0.25882)],
        "N": 20,
        "x_left": 0.0,
        "x_right": 1.0,
        "expected": "[0.35503, -0.25882, ~0, ...]",
        "current_best": 5.76e-05
    },
    {
        "name": "Hermite (n=6)",
        "c_list": [2 * 6, -2 * x, sp.Integer(1)],
        "f_expr": sp.Integer(0),
        "bc_tuples": [(0.0, 0, -120.0), (0.0, 1, 0.0)],
        "N": 20,
        "x_left": -0.9,
        "x_right": 0.9,
        "expected": "[-120, 0, 720, 0, -120, ...]",
        "current_best": 8.39e-03
    },
    {
        "name": "Beam",
        "c_list": [sp.Integer(0), sp.Integer(0), sp.Integer(0), sp.Integer(0), 1 + sp.Rational(3, 10) * x**2],
        "f_expr": sp.sin(2 * x),
        "bc_tuples": [(0.0, 0, 0.0), (0.0, 1, 0.0), (1.0, 2, 0.0), (1.0, 3, 0.0)],
        "N": 20,
        "x_left": 0.0,
        "x_right": 1.0,
        "expected": "[0,0,0,0,0.25,...]",
        "current_best": 5.84e-01
    },
    {
        "name": "Simple Harmonic",
        "c_list": [sp.Integer(1), sp.Integer(0), sp.Integer(1)],
        "f_expr": sp.Integer(0),
        "bc_tuples": [(0.0, 0, 1.0), (0.0, 1, 0.0)],
        "N": 20,
        "x_left": -0.5,
        "x_right": 0.5,
        "expected": "[1, 0, -1/2!, 0, 1/4!, ...]",
        "current_best": 3.73e-07
    },
    {
        "name": "Exponential",
        "c_list": [sp.Integer(1), -sp.Integer(1)],
        "f_expr": sp.Integer(0),
        "bc_tuples": [(0.0, 0, 1.0)],
        "N": 20,
        "x_left": -0.5,
        "x_right": 0.5,
        "expected": "[1, 1, 1/2!, 1/3!, ...]",
        "current_best": 4.55e-06
    }
]

results = []
target_achieved = []

for eq in equations:
    print(f"{eq['name']}")
    print("-" * 30)
    print(f"Expected: {eq['expected']}")
    print(f"Current best: {eq['current_best']:.2e}")
    
    coeffs, final_loss = train_improved_rnn_pinn(
        eq["c_list"],
        eq["f_expr"],
        eq["bc_tuples"],
        N=eq["N"],
        x_left=eq["x_left"],
        x_right=eq["x_right"],
        num_collocation=1000,
        bc_weight=1.0,
        recurrence_weight=10000.0,
        dtype=DTYPE,
        adam_iters=500,
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
    
    print(f"Coefficients: {coeffs[:6]}")
    print(f"Loss: {final_loss:.2e}")
    
    if final_loss < eq["current_best"]:
        improvement = eq["current_best"] / final_loss
        print(f"✓ IMPROVEMENT: {improvement:.2f}x better!")
    else:
        print("✗ No improvement")
    
    if final_loss < 1e-10:
        print("✓ TARGET ACHIEVED!")
        target_achieved.append(eq["name"])
    
    results.append({
        "name": eq["name"],
        "loss": final_loss,
        "current_best": eq["current_best"],
        "coeffs": coeffs[:6]
    })
    
    print()

print("COMPREHENSIVE RESULTS")
print("=" * 25)
print(f"Target (<1e-10) achieved: {len(target_achieved)}/6 equations")
if target_achieved:
    print(f"Success on: {', '.join(target_achieved)}")

print("\nSummary:")
for r in results:
    status = "✓" if r["loss"] < r["current_best"] else "✗"
    target = "TARGET" if r["loss"] < 1e-10 else f"{r['loss']:.2e}"
    print(f"{status} {r['name']}: {target}")

print(f"\nBest approach so far: Extreme recurrence weighting (10000x)")
print(f"Works best on: Sparse coefficient patterns")
print(f"Next: Focus on achieving <1e-10 on all equations")