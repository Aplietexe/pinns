"""
Baseline direct parameter optimization for comparison.
This optimizes the coefficients directly as nn.Parameter to establish target performance.
"""

import math
import sympy as sp
import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import trange

from single_ode import make_recurrence, _poly_eval


def train_direct_parameters(
    c_list: list[sp.Expr],
    f_expr: sp.Expr,
    bc_tuples: list[tuple[float, int, float]],
    *,
    N: int = 10,
    x_left: float = 0.0,
    x_right: float = 1.0,
    num_collocation: int = 1000,
    bc_weight: float = 100.0,
    recurrence_weight: float = 1.0,
    dtype: torch.dtype = torch.float32,
    device: torch.device | None = None,
    seed: int = 1234,
    adam_iters: int = 1000,
    lbfgs_iters: int = 15,
    progress: bool = True,
) -> torch.Tensor:
    """Train direct parameters as baseline comparison."""
    
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    torch.manual_seed(seed)
    
    # Pre-compute factorials 0!, …, N!
    fact = torch.tensor(
        [math.factorial(i) for i in range(N + 1)], dtype=dtype, device=device
    )
    
    # Collocation points
    import numpy as np
    xs_coll_np = np.linspace(x_left, x_right, num_collocation)
    xs_collocation = torch.tensor(xs_coll_np, dtype=dtype, device=device)
    
    # Lambdify coefficient functions & RHS
    x_sym = sp.symbols("x")
    c_funcs = [sp.lambdify(x_sym, c) for c in c_list]
    f_func = sp.lambdify(x_sym, f_expr)
    
    c_vals = [
        torch.tensor(cf(xs_coll_np), dtype=dtype, device=device) for cf in c_funcs
    ]
    f_vals = torch.tensor(f_func(xs_coll_np), dtype=dtype, device=device)
    
    m = len(c_list) - 1
    
    # Direct parameter optimization
    coeffs = nn.Parameter(torch.randn(N + 1, dtype=dtype, device=device) * 0.1)
    
    # Build recurrence for additional loss term
    rec_next_coef = make_recurrence(
        c_list,
        f_expr,
        dtype=dtype,
        device=device,
        max_n=N - m,
    )
    
    def loss_fn() -> torch.Tensor:
        # ODE residual
        u_ks = [_poly_eval(xs_collocation, coeffs, fact, shift=k) for k in range(m + 1)]
        residual = sum(c_vals[k] * u_ks[k] for k in range(m + 1)) - f_vals
        loss_pde = (residual**2).mean()
        
        # Boundary conditions
        bc_terms: list[torch.Tensor] = []
        for x0, k, val in bc_tuples:
            x_t = torch.tensor([x0], dtype=dtype, device=device)
            u_val = _poly_eval(x_t, coeffs, fact, shift=k)
            bc_terms.append((u_val - val) ** 2)
        loss_bc = (
            torch.sum(torch.stack(bc_terms))
            if bc_terms
            else torch.tensor(0.0, device=device)
        )
        
        # Recurrence consistency loss
        rec_terms: list[torch.Tensor] = []
        if recurrence_weight != 0.0:
            for ell in range(m, N + 1):
                a_prev = coeffs[:ell]
                a_pred = rec_next_coef(a_prev)
                rec_terms.append((coeffs[ell] - a_pred) ** 2)
        loss_rec = (
            torch.stack(rec_terms).mean()
            if rec_terms
            else torch.tensor(0.0, device=device)
        )
        
        return loss_pde + bc_weight * loss_bc + recurrence_weight * loss_rec
    
    # Adam optimization
    opt = optim.AdamW([coeffs], lr=1e-3)
    if progress:
        print("Direct Parameter Optimization - Adam")
        print("-" * 40)
    
    pbar = trange(adam_iters, desc="Adam", disable=not progress)
    for _ in pbar:
        opt.zero_grad()
        loss = loss_fn()
        loss.backward()
        opt.step()
        pbar.set_postfix(loss=loss.item())
    
    # LBFGS fine-tuning
    if progress:
        print("Direct Parameter Optimization - LBFGS")
        print("-" * 40)
    
    opt_lbfgs = optim.LBFGS(
        [coeffs],
        lr=1.0,
        max_iter=40,
        tolerance_grad=1e-16,
        tolerance_change=1e-16,
        history_size=30,
        line_search_fn="strong_wolfe",
    )
    
    pbar = trange(lbfgs_iters, desc="LBFGS", disable=not progress)
    for _ in pbar:
        def closure():
            opt_lbfgs.zero_grad()
            loss_val = loss_fn()
            loss_val.backward()
            return loss_val
        
        loss_val = opt_lbfgs.step(closure)
        pbar.set_postfix(loss=loss_val.item())
    
    return coeffs.detach().cpu()


if __name__ == "__main__":
    # Test on all ODEs
    import numpy as np
    
    DTYPE = torch.float64
    x = sp.symbols("x")
    
    print("BASELINE: Direct Parameter Optimization")
    print("=" * 50)
    
    # Test cases
    test_cases = [
        {
            "name": "Legendre (l=5)",
            "c_list": [5 * 6, -2 * x, 1 - x**2],
            "f_expr": sp.Integer(0),
            "bc_tuples": [(0.0, 0, 0.0), (1.0, 0, 1.0)],
            "N": 30,
            "x_left": -0.9,
            "x_right": 0.9,
        },
        {
            "name": "Airy",
            "c_list": [-x, sp.Integer(0), sp.Integer(1)],
            "f_expr": sp.Integer(0),
            "bc_tuples": [(0.0, 0, 0.35502805388781723926), (0.0, 1, -0.25881940379280679840)],
            "N": 40,
            "x_left": -0.9,
            "x_right": 0.9,
        },
        {
            "name": "Hermite (n=6)",
            "c_list": [2 * 6, -2 * x, sp.Integer(1)],
            "f_expr": sp.Integer(0),
            "bc_tuples": [(0.0, 0, -120.0), (0.0, 1, 0.0)],
            "N": 30,
            "x_left": -0.9,
            "x_right": 0.9,
        },
        {
            "name": "Beam",
            "c_list": [sp.Integer(0), sp.Integer(0), sp.Integer(0), sp.Integer(0), 1 + sp.Rational(3, 10) * x**2],
            "f_expr": sp.sin(2 * x),
            "bc_tuples": [(0.0, 0, 0.0), (0.0, 1, 0.0), (1.0, 2, 0.0), (1.0, 3, 0.0)],
            "N": 40,
            "x_left": 0.0,
            "x_right": 1.0,
        },
    ]
    
    for test in test_cases:
        print(f"\n{test['name']}")
        print("-" * 40)
        
        coeffs = train_direct_parameters(
            test["c_list"],
            test["f_expr"], 
            test["bc_tuples"],
            N=test["N"],
            x_left=test["x_left"],
            x_right=test["x_right"],
            num_collocation=5000,
            bc_weight=100.0,
            recurrence_weight=100.0,
            dtype=DTYPE,
            adam_iters=1000,
            lbfgs_iters=15,
        )
        
        print(f"✓ {test['name']} baseline completed")