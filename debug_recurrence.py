"""
Debug the recurrence function - there's clearly a fundamental bug
"""

import sympy as sp
import torch
import numpy as np
from single_ode import make_recurrence

DTYPE = torch.float32
DEVICE = torch.device("cpu")

def manual_airy_recurrence(prefix):
    """Manual implementation of Airy recurrence"""
    if len(prefix) < 2:
        raise ValueError("Need at least 2 coefficients")
    
    # Airy equation: u'' = xu
    # This gives: a_{n+2} * (n+2)(n+1) = a_n * 1
    # So: a_{n+2} = a_n / ((n+2)(n+1))
    
    n = len(prefix)  # We want to compute a_n
    
    if n < 2:
        raise ValueError("Need at least 2 initial coefficients")
    
    # a_n = a_{n-2} / (n * (n-1))
    a_n = prefix[n-2] / (n * (n-1))
    return a_n

def test_manual_recurrence():
    """Test manual recurrence implementation"""
    print("MANUAL RECURRENCE TEST")
    print("=" * 25)
    
    # Test case: a_0 = 1, a_1 = 0 (should give Airy Ai function)
    coeffs = [1.0, 0.0]
    
    print(f"Starting with a_0 = {coeffs[0]}, a_1 = {coeffs[1]}")
    print()
    
    # Compute next few coefficients
    for i in range(2, 8):
        next_coeff = manual_airy_recurrence(coeffs)
        coeffs.append(next_coeff)
        print(f"a_{i} = a_{i-2} / ({i} * {i-1}) = {coeffs[i-2]} / {i*(i-1)} = {next_coeff}")
    
    print(f"\\nFinal sequence: {coeffs}")
    
    # Compare with what the make_recurrence function gives
    print(f"\\nTesting make_recurrence function:")
    
    x = sp.symbols("x")
    c_list = [sp.Integer(0), sp.Integer(0), 1 - x]
    f_expr = sp.Integer(0)
    
    recurrence_fn = make_recurrence(c_list, f_expr, dtype=DTYPE, device=DEVICE, max_n=10)
    
    # Test each prefix
    for i in range(2, len(coeffs)):
        prefix = torch.tensor(coeffs[:i], dtype=DTYPE, device=DEVICE)
        try:
            result = recurrence_fn(prefix).item()
            expected = coeffs[i]
            print(f"Prefix {coeffs[:i]} -> Expected: {expected:.6f}, Got: {result:.6f}, Diff: {abs(expected-result):.6f}")
        except Exception as e:
            print(f"Prefix {coeffs[:i]} -> Error: {e}")

def analyze_recurrence_formula():
    """Analyze what the recurrence function should be doing"""
    print("\\nRECURRENCE FORMULA ANALYSIS")
    print("=" * 30)
    
    # For Airy equation: u'' - xu = 0
    # In general form: c_2(x) * u'' + c_1(x) * u' + c_0(x) * u = f(x)
    # For Airy: c_2(x) = 1, c_1(x) = 0, c_0(x) = -x, f(x) = 0
    
    print("Airy equation: u'' - xu = 0")
    print("Standard form: c_2(x) * u'' + c_1(x) * u' + c_0(x) * u = f(x)")
    print("For Airy: c_2(x) = 1, c_1(x) = 0, c_0(x) = -x, f(x) = 0")
    print()
    
    # But in our code, we have:
    print("In our code c_list = [0, 0, 1-x]")
    print("This means: c_0(x) = 0, c_1(x) = 0, c_2(x) = 1-x")
    print("So the equation is: (1-x) * u'' = 0")
    print("This is NOT the Airy equation!")
    print()
    
    print("For the actual Airy equation u'' - xu = 0:")
    print("We need: c_2(x) = 1, c_1(x) = 0, c_0(x) = -x")
    print("So c_list should be [-x, 0, 1]")
    
    # Test with correct coefficients
    print("\\nTesting with correct Airy coefficients:")
    
    x = sp.symbols("x")
    c_list_correct = [-x, sp.Integer(0), sp.Integer(1)]  # Correct Airy
    f_expr = sp.Integer(0)
    
    print(f"c_list = {c_list_correct}")
    
    try:
        recurrence_fn_correct = make_recurrence(c_list_correct, f_expr, dtype=DTYPE, device=DEVICE, max_n=10)
        
        # Test with a_0 = 1, a_1 = 0
        prefix = torch.tensor([1.0, 0.0], dtype=DTYPE, device=DEVICE)
        result = recurrence_fn_correct(prefix).item()
        expected = 1.0 / (2 * 1)  # a_2 = a_0 / (2*1) = 1/2 = 0.5
        
        print(f"Prefix [1.0, 0.0] -> Expected: {expected:.6f}, Got: {result:.6f}")
        print(f"Difference: {abs(expected - result):.6f}")
        
    except Exception as e:
        print(f"Error with correct coefficients: {e}")

if __name__ == "__main__":
    test_manual_recurrence()
    analyze_recurrence_formula()