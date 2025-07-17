"""
Corrected ODE coefficient definitions for all test cases
"""

import sympy as sp
import torch
from single_ode import make_recurrence

def get_corrected_ode_definitions():
    """Get the corrected coefficient definitions for all ODEs"""
    x = sp.symbols("x")
    
    # All ODEs in standard form: c_m(x) * u^(m) + ... + c_1(x) * u' + c_0(x) * u = f(x)
    
    odes = {
        "airy": {
            "name": "Airy equation",
            "equation": "u'' = xu",
            "standard_form": "u'' - xu = 0",
            "c_list": [-x, sp.Integer(0), sp.Integer(1)],  # [c_0, c_1, c_2]
            "f_expr": sp.Integer(0),
            "bc_tuples": [(0.0, 0, 0.35503), (0.0, 1, -0.25882)],  # Airy Ai values
            "N": 20,
            "x_left": 0.0,
            "x_right": 1.0,
            "expected_pattern": "[1, 0, 0, 1/6, 0, 0, 1/180, 0, 0, ...]"
        },
        
        "legendre": {
            "name": "Legendre equation (l=5)",
            "equation": "(1-x²)u'' - 2xu' + 30u = 0",
            "standard_form": "30u - 2xu' + (1-x²)u'' = 0",
            "c_list": [sp.Integer(30), -2*x, 1-x**2],  # [c_0, c_1, c_2]
            "f_expr": sp.Integer(0),
            "bc_tuples": [(0.0, 0, 0.0), (1.0, 0, 1.0)],
            "N": 25,
            "x_left": -0.9,
            "x_right": 0.9,
            "expected_pattern": "[0, 0, 0, 0, 0, 1, ...]"  # P_5(x)
        },
        
        "hermite": {
            "name": "Hermite equation (n=6)",
            "equation": "u'' - 2xu' + 12u = 0",
            "standard_form": "12u - 2xu' + u'' = 0",
            "c_list": [sp.Integer(12), -2*x, sp.Integer(1)],  # [c_0, c_1, c_2]
            "f_expr": sp.Integer(0),
            "bc_tuples": [(0.0, 0, -120.0), (0.0, 1, 0.0)],  # H_6(0) = -120
            "N": 25,
            "x_left": -0.9,
            "x_right": 0.9,
            "expected_pattern": "[-120, 0, 720, 0, -120, 0, 8, 0, ...]"  # H_6(x)
        },
        
        "beam": {
            "name": "Beam equation",
            "equation": "u'''' + (1 + 0.3x²)u = sin(2x)",
            "standard_form": "(1 + 0.3x²)u + u'''' = sin(2x)",
            "c_list": [1 + sp.Rational(3,10)*x**2, sp.Integer(0), sp.Integer(0), sp.Integer(0), sp.Integer(1)],
            "f_expr": sp.sin(2*x),
            "bc_tuples": [(0.0, 0, 0.0), (0.0, 1, 0.0), (1.0, 2, 0.0), (1.0, 3, 0.0)],
            "N": 25,
            "x_left": 0.0,
            "x_right": 1.0,
            "expected_pattern": "[0, 0, 0, 0, a_4, a_5, ...]"  # First 4 coeffs from BCs
        },
        
        "harmonic": {
            "name": "Simple harmonic oscillator",
            "equation": "u'' + u = 0",
            "standard_form": "u + u'' = 0",
            "c_list": [sp.Integer(1), sp.Integer(0), sp.Integer(1)],  # [c_0, c_1, c_2]
            "f_expr": sp.Integer(0),
            "bc_tuples": [(0.0, 0, 1.0), (0.0, 1, 0.0)],  # cos(x): u(0)=1, u'(0)=0
            "N": 20,
            "x_left": -0.5,
            "x_right": 0.5,
            "expected_pattern": "[1, 0, -1/2, 0, 1/24, 0, -1/720, ...]"  # cos(x)
        },
        
        "exponential": {
            "name": "Exponential growth",
            "equation": "u' - u = 0",
            "standard_form": "-u + u' = 0",
            "c_list": [-sp.Integer(1), sp.Integer(1)],  # [c_0, c_1]
            "f_expr": sp.Integer(0),
            "bc_tuples": [(0.0, 0, 1.0)],  # e^x: u(0)=1
            "N": 20,
            "x_left": -0.5,
            "x_right": 0.5,
            "expected_pattern": "[1, 1, 1/2, 1/6, 1/24, 1/120, ...]"  # e^x
        }
    }
    
    return odes

def test_corrected_recurrence():
    """Test the corrected recurrence relations"""
    print("TESTING CORRECTED RECURRENCE RELATIONS")
    print("=" * 45)
    
    odes = get_corrected_ode_definitions()
    DTYPE = torch.float32
    DEVICE = torch.device("cpu")
    
    for key, ode in odes.items():
        print(f"\n{ode['name']}")
        print("-" * 30)
        print(f"Equation: {ode['equation']}")
        print(f"c_list: {ode['c_list']}")
        print(f"Expected pattern: {ode['expected_pattern']}")
        
        try:
            # Create recurrence function
            recurrence_fn = make_recurrence(
                ode['c_list'], 
                ode['f_expr'], 
                dtype=DTYPE, 
                device=DEVICE, 
                max_n=15
            )
            
            # Test with different prefix lengths
            test_prefixes = [
                torch.tensor([1.0, 0.0, 0.0], dtype=DTYPE, device=DEVICE),
                torch.tensor([0.0, 1.0, 0.0], dtype=DTYPE, device=DEVICE),
                torch.tensor([1.0, 0.0, 0.0, 0.0], dtype=DTYPE, device=DEVICE),
            ]
            
            for i, prefix in enumerate(test_prefixes):
                try:
                    result = recurrence_fn(prefix)
                    print(f"  Prefix {prefix.tolist()} -> {result.item():.6f}")
                except Exception as e:
                    print(f"  Prefix {prefix.tolist()} -> Error: {e}")
        
        except Exception as e:
            print(f"  Failed to create recurrence: {e}")

def validate_with_known_solutions():
    """Validate some ODEs with known analytical solutions"""
    print("\n\nVALIDATION WITH KNOWN SOLUTIONS")
    print("=" * 35)
    
    DTYPE = torch.float32
    DEVICE = torch.device("cpu")
    
    # Test harmonic oscillator: cos(x) = 1 - x²/2! + x⁴/4! - x⁶/6! + ...
    print("Harmonic oscillator: cos(x) series")
    print("Expected: [1, 0, -1/2, 0, 1/24, 0, -1/720, ...]")
    
    odes = get_corrected_ode_definitions()
    harmonic = odes['harmonic']
    
    try:
        recurrence_fn = make_recurrence(
            harmonic['c_list'], 
            harmonic['f_expr'], 
            dtype=DTYPE, 
            device=DEVICE, 
            max_n=10
        )
        
        # Start with cos(x) coefficients
        coeffs = [1.0, 0.0]  # cos(x) starts with [1, 0, -1/2, 0, 1/24, ...]
        
        print("Computing coefficients:")
        for i in range(2, 8):
            prefix = torch.tensor(coeffs, dtype=DTYPE, device=DEVICE)
            try:
                next_coeff = recurrence_fn(prefix).item()
                coeffs.append(next_coeff)
                
                # Compare with known cos(x) coefficients
                if i == 2:
                    expected = -1.0/2  # -1/2!
                elif i == 3:
                    expected = 0.0
                elif i == 4:
                    expected = 1.0/24  # 1/4!
                elif i == 5:
                    expected = 0.0
                elif i == 6:
                    expected = -1.0/720  # -1/6!
                else:
                    expected = "unknown"
                
                print(f"  a_{i} = {next_coeff:.6f}, expected ≈ {expected}")
                
            except Exception as e:
                print(f"  a_{i}: Error - {e}")
                break
                
    except Exception as e:
        print(f"Failed to create harmonic recurrence: {e}")

if __name__ == "__main__":
    test_corrected_recurrence()
    validate_with_known_solutions()