"""
Correct analysis of Airy equation recurrence relation
"""

import sympy as sp
import torch
import numpy as np
from single_ode import make_recurrence

DTYPE = torch.float32
DEVICE = torch.device("cpu")

def correct_airy_recurrence(prefix):
    """Correct manual implementation of Airy recurrence"""
    if len(prefix) < 3:
        raise ValueError("Need at least 3 coefficients for Airy recurrence")
    
    # Airy equation: u'' = xu
    # Power series: u(x) = Σ a_n * x^n
    # u''(x) = Σ (n+2)(n+1) * a_(n+2) * x^n  
    # xu(x) = Σ a_(n-1) * x^n
    
    # Equation: (n+2)(n+1) * a_(n+2) - a_(n-1) = 0 for n ≥ 1
    # Therefore: a_(n+2) = a_(n-1) / ((n+2)(n+1))
    
    # For the next coefficient a_k where k = len(prefix):
    # a_k = a_(k-3) / (k * (k-1))
    
    k = len(prefix)
    if k < 3:
        raise ValueError("Need at least 3 coefficients")
    
    # a_k = a_(k-3) / (k * (k-1))
    a_k = prefix[k-3] / (k * (k-1))
    return a_k

def analyze_correct_airy_recurrence():
    """Analyze the correct Airy recurrence relation"""
    print("CORRECT AIRY RECURRENCE ANALYSIS")
    print("=" * 35)
    
    print("Airy equation: u'' = xu")
    print("Power series: u(x) = Σ a_n * x^n")
    print("u''(x) = Σ (n+2)(n+1) * a_(n+2) * x^n")
    print("xu(x) = Σ a_(n-1) * x^n (for n ≥ 1)")
    print()
    print("Recurrence relation:")
    print("(n+2)(n+1) * a_(n+2) = a_(n-1) for n ≥ 1")
    print("Therefore: a_(n+2) = a_(n-1) / ((n+2)(n+1))")
    print()
    print("Or equivalently: a_(k+3) = a_k / ((k+3)(k+2))")
    print("Which means: a_k = a_(k-3) / (k * (k-1)) for k ≥ 3")
    print()
    
    # Test with standard initial conditions
    print("Testing with a_0=1, a_1=0, a_2=0 (Airy Ai function):")
    coeffs = [1.0, 0.0, 0.0]
    
    print(f"Initial: a_0={coeffs[0]}, a_1={coeffs[1]}, a_2={coeffs[2]}")
    
    # Compute next coefficients using correct recurrence
    for k in range(3, 10):
        next_coeff = correct_airy_recurrence(coeffs)
        coeffs.append(next_coeff)
        print(f"a_{k} = a_{k-3} / ({k} * {k-1}) = {coeffs[k-3]} / {k*(k-1)} = {next_coeff}")
    
    print(f"\nFinal sequence: {coeffs}")
    return coeffs

def test_make_recurrence_with_correct_coefficients():
    """Test the make_recurrence function with correct Airy coefficients"""
    print("\nTESTING make_recurrence WITH CORRECT AIRY COEFFICIENTS")
    print("=" * 55)
    
    # Correct Airy equation: u'' - xu = 0
    # Standard form: c_2(x) * u'' + c_1(x) * u' + c_0(x) * u = f(x)
    # So: c_0(x) = -x, c_1(x) = 0, c_2(x) = 1
    
    x = sp.symbols("x")
    c_list = [-x, sp.Integer(0), sp.Integer(1)]  # [c_0, c_1, c_2]
    f_expr = sp.Integer(0)
    
    print(f"c_list = {c_list}")
    print("This represents: -x * u + 0 * u' + 1 * u'' = 0")
    print("Which is: u'' - xu = 0 ✓")
    print()
    
    try:
        recurrence_fn = make_recurrence(c_list, f_expr, dtype=DTYPE, device=DEVICE, max_n=15)
        
        # Get expected coefficients from manual calculation
        expected_coeffs = analyze_correct_airy_recurrence()
        
        print("Comparing make_recurrence with manual calculation:")
        print("-" * 50)
        
        # Test with different prefix lengths
        for i in range(3, min(8, len(expected_coeffs))):
            prefix = torch.tensor(expected_coeffs[:i], dtype=DTYPE, device=DEVICE)
            try:
                result = recurrence_fn(prefix).item()
                expected = expected_coeffs[i]
                diff = abs(result - expected)
                status = "✓" if diff < 1e-6 else "✗"
                print(f"{status} Prefix length {i}: Expected {expected:.6f}, Got {result:.6f}, Diff {diff:.6f}")
            except Exception as e:
                print(f"✗ Prefix length {i}: Error - {e}")
        
    except Exception as e:
        print(f"Error creating recurrence function: {e}")

def test_other_odes():
    """Test what the correct coefficient definitions should be for other ODEs"""
    print("\nCORRECT COEFFICIENT DEFINITIONS FOR OTHER ODES")
    print("=" * 45)
    
    # Legendre equation: (1-x²)u'' - 2xu' + l(l+1)u = 0
    # Standard form: c_2(x) * u'' + c_1(x) * u' + c_0(x) * u = f(x)
    # So: c_0(x) = l(l+1), c_1(x) = -2x, c_2(x) = (1-x²)
    
    x = sp.symbols("x")
    
    print("1. Legendre equation (l=5): (1-x²)u'' - 2xu' + 30u = 0")
    legendre_c_list = [sp.Integer(30), -2*x, 1-x**2]
    print(f"   c_list = {legendre_c_list}")
    print()
    
    print("2. Hermite equation (n=6): u'' - 2xu' + 12u = 0")
    hermite_c_list = [sp.Integer(12), -2*x, sp.Integer(1)]
    print(f"   c_list = {hermite_c_list}")
    print()
    
    print("3. Beam equation: u'''' + (1 + 0.3x²)u = sin(2x)")
    print("   This is 4th order, so c_list = [1 + 0.3x², 0, 0, 0, 1]")
    beam_c_list = [1 + sp.Rational(3,10)*x**2, 0, 0, 0, 1]
    print(f"   c_list = {beam_c_list}")
    print()
    
    print("4. Simple harmonic: u'' + u = 0")
    harmonic_c_list = [sp.Integer(1), sp.Integer(0), sp.Integer(1)]
    print(f"   c_list = {harmonic_c_list}")
    print()
    
    print("5. Exponential: u' - u = 0")
    exp_c_list = [-sp.Integer(1), sp.Integer(1)]
    print(f"   c_list = {exp_c_list}")

if __name__ == "__main__":
    expected_coeffs = analyze_correct_airy_recurrence()
    test_make_recurrence_with_correct_coefficients()
    test_other_odes()