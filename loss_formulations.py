"""
Experiment 16: Different Loss Function Formulations
Test various loss formulations for sparse coefficient patterns
"""

import sympy as sp
import torch
from improved_rnn_pinn import train_improved_rnn_pinn

DTYPE = torch.float64
x = sp.symbols("x")

print("EXPERIMENT 16: DIFFERENT LOSS FUNCTION FORMULATIONS")
print("=" * 50)
print("Strategy: Test different loss weighting strategies")
print("Focus: Sparse coefficient patterns (Legendre equation)")
print()

# Legendre equation - the most challenging
c_list = [5 * 6, -2 * x, 1 - x**2]
f_expr = sp.Integer(0)
bc_tuples = [(0.0, 0, 0.0), (1.0, 0, 1.0)]
N = 15  # Reduced for faster testing
x_left = -0.9
x_right = 0.9
expected = "Should be [0,0,0,0,0,30,...]"

print(f"Expected: {expected}")
print("Current best: 1.38e+00 loss")
print()

# Different loss formulations to test
loss_formulations = [
    {
        "name": "Baseline",
        "bc_weight": 100.0,
        "recurrence_weight": 100.0,
        "description": "Standard balanced weights"
    },
    {
        "name": "Recurrence-Heavy",
        "bc_weight": 10.0,
        "recurrence_weight": 1000.0,
        "description": "Emphasize pattern learning"
    },
    {
        "name": "Sparse-Friendly",
        "bc_weight": 1000.0,
        "recurrence_weight": 10.0,
        "description": "Emphasize boundary conditions"
    },
    {
        "name": "Minimal-Recurrence",
        "bc_weight": 100.0,
        "recurrence_weight": 1.0,
        "description": "Minimal recurrence influence"
    }
]

best_loss = float('inf')
best_coeffs = None
best_formulation = None

for form in loss_formulations:
    print(f"{form['name']}: {form['description']}")
    print(f"Weights: BC={form['bc_weight']}, Rec={form['recurrence_weight']}")
    print("-" * 40)
    
    coeffs, final_loss = train_improved_rnn_pinn(
        c_list,
        f_expr,
        bc_tuples,
        N=N,
        x_left=x_left,
        x_right=x_right,
        num_collocation=1000,
        bc_weight=form["bc_weight"],
        recurrence_weight=form["recurrence_weight"],
        dtype=DTYPE,
        adam_iters=400,
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
    
    if final_loss < best_loss:
        best_loss = final_loss
        best_coeffs = coeffs
        best_formulation = form["name"]
        print("✓ NEW BEST!")
    
    print()

print("LOSS FORMULATION RESULTS")
print("=" * 30)
print(f"Best formulation: {best_formulation}")
print(f"Best loss: {best_loss:.2e} (vs current best 1.38e+00)")
print(f"Best coefficients: {best_coeffs[:6]}")

if best_loss < 1.38e+00:
    improvement = 1.38e+00 / best_loss
    print(f"✓ IMPROVEMENT: {improvement:.2f}x better!")
else:
    print("✗ No improvement from loss formulation changes")

print()
print("ANALYSIS:")
print("- Different loss weights emphasize different aspects")
print("- Recurrence-heavy: Better for pattern learning")
print("- BC-heavy: Better for boundary satisfaction")
print("- Sparse patterns may need specialized loss formulations")
print("- Next: Try architectural modifications if loss changes don't help")