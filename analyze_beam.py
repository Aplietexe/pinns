"""
Analyze the Beam equation to understand why it causes model collapse
"""

import sympy as sp
import torch
import numpy as np
from single_ode import make_recurrence

# Beam equation setup
x = sp.symbols("x")
c_list = [sp.Integer(0), sp.Integer(0), sp.Integer(0), sp.Integer(0), 1 + sp.Rational(3, 10) * x**2]
f_expr = sp.sin(2 * x)
bc_tuples = [(0.0, 0, 0.0), (0.0, 1, 0.0), (1.0, 2, 0.0), (1.0, 3, 0.0)]

print("BEAM EQUATION ANALYSIS")
print("=" * 30)

print(f"c_list: {c_list}")
print(f"f_expr: {f_expr}")
print(f"bc_tuples: {bc_tuples}")
print()

# The equation is: c_0 * u + c_1 * u' + c_2 * u'' + c_3 * u''' + c_4 * u'''' = f(x)
# Which becomes: 0 * u + 0 * u' + 0 * u'' + 0 * u''' + (1 + 3x²/10) * u'''' = sin(2x)
# Or simply: (1 + 3x²/10) * u'''' = sin(2x)
# This is a 4th order ODE

print("Equation structure:")
print("(1 + 3x²/10) * u'''' = sin(2x)")
print("This is a 4th order ODE where the first 4 coefficients should be determined by boundary conditions")
print()

# Check boundary conditions
print("Boundary conditions:")
for i, (x_val, deriv_order, bc_val) in enumerate(bc_tuples):
    print(f"  u^({deriv_order})(x={x_val}) = {bc_val}")
print()

# Let's see what the recurrence relation looks like
print("Analyzing recurrence relation...")
N = 10
dtype = torch.float64
device = torch.device("cpu")

try:
    rec_next_coef = make_recurrence(c_list, f_expr, dtype=dtype, device=device, max_n=N - 4)
    print("Recurrence relation created successfully")
    
    # Test with some sample coefficients
    test_coeffs = torch.zeros(8, dtype=dtype, device=device)
    test_coeffs[4] = 0.25  # Expected 5th coefficient
    
    print(f"Test coefficients: {test_coeffs}")
    
    # Try to compute next coefficient
    for i in range(4, 8):
        if i < len(test_coeffs):
            prefix = test_coeffs[:i]
            print(f"Prefix (first {i} coeffs): {prefix}")
            try:
                next_coef = rec_next_coef(prefix)
                print(f"Next coefficient prediction: {next_coef}")
            except Exception as e:
                print(f"Error computing next coefficient: {e}")
    
except Exception as e:
    print(f"Error creating recurrence: {e}")

print()
print("HYPOTHESIS:")
print("The Beam equation may be problematic because:")
print("1. First 4 coefficients are constrained by boundary conditions, not recurrence")
print("2. Recurrence only starts from the 5th coefficient") 
print("3. Autoregressive approach struggles with this mixed constraint structure")
print("4. The sin(2x) RHS creates a complex coefficient pattern")
print()

# Let's also check what the expected solution should look like
print("Expected solution pattern:")
print("For 4th order ODE with 4 boundary conditions:")
print("- First 4 coefficients: determined by boundary conditions")
print("- Remaining coefficients: determined by recurrence relation with sin(2x)")
print("- Expected first few: [0, 0, 0, 0, 0.25, ...] (approximately)")