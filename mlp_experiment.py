"""
Experiment 1: Replace transformer with simple MLP
"""

import math
import sympy as sp
import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import trange
from single_ode import make_recurrence, _poly_eval


class SimpleMLP(nn.Module):
    """Simple MLP that predicts next coefficient from prefix"""
    
    def __init__(self, max_input_len: int = 50, hidden_dims: list[int] = [256, 256], dropout: float = 0.1):
        super().__init__()
        self.max_input_len = max_input_len
        
        # Build MLP layers
        layers = []
        in_dim = max_input_len
        
        for hidden_dim in hidden_dims:
            layers.extend([
                nn.Linear(in_dim, hidden_dim),
                nn.ReLU(),
                nn.Dropout(dropout) if dropout > 0 else nn.Identity()
            ])
            in_dim = hidden_dim
        
        # Output layer
        layers.append(nn.Linear(in_dim, 1))
        
        self.mlp = nn.Sequential(*layers)
        
    def forward(self, prefix: torch.Tensor) -> torch.Tensor:
        """Predict next coefficient from prefix"""
        if prefix.ndim != 1:
            raise ValueError("prefix must be a 1-D tensor")
        
        # Pad or truncate to max_input_len
        if prefix.shape[0] > self.max_input_len:
            x = prefix[-self.max_input_len:]
        else:
            # Pad with zeros at the beginning
            padding = torch.zeros(self.max_input_len - prefix.shape[0], 
                                device=prefix.device, dtype=prefix.dtype)
            x = torch.cat([padding, prefix])
        
        return self.mlp(x).squeeze(-1)


def train_mlp_pinn(
    c_list: list[sp.Expr],
    f_expr: sp.Expr,
    bc_tuples: list[tuple[float, int, float]],
    *,
    N: int = 10,
    x_left: float = 0.0,
    x_right: float = 1.0,
    num_collocation: int = 1000,
    bc_weight: float = 100.0,
    adam_iters: int = 1000,
    lbfgs_iters: int = 15,
    recurrence_weight: float = 1.0,
    dtype: torch.dtype = torch.float32,
    device: torch.device | None = None,
    seed: int = 1234,
    progress: bool = True,
    hidden_dims: list[int] = [256, 256],
    dropout: float = 0.1,
) -> torch.Tensor:
    """Train MLP-based PINN"""
    
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    torch.manual_seed(seed)
    
    # Pre-compute factorials and other setup (same as original)
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
    
    # Create MLP model
    net = SimpleMLP(
        max_input_len=N + 1,
        hidden_dims=hidden_dims,
        dropout=dropout
    ).to(device=device, dtype=dtype)
    
    # Build recurrence for additional loss term
    rec_next_coef = make_recurrence(
        c_list,
        f_expr,
        dtype=dtype,
        device=device,
        max_n=N - m,
    )
    
    def loss_fn() -> torch.Tensor:
        # Auto-regressively generate coefficients
        coeff_list: list[torch.Tensor] = []
        prefix = torch.empty(0, dtype=dtype, device=device)
        
        for _ in range(N + 1):
            next_coef = net(prefix)
            coeff_list.append(next_coef)
            prefix = torch.stack(coeff_list)
        
        coeffs = torch.stack(coeff_list)
        
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
    
    # Stage 1: Adam
    opt = optim.AdamW(net.parameters(), lr=1e-3)
    if progress:
        print("MLP Architecture - Adam")
        print("-" * 30)
    
    pbar = trange(adam_iters, desc="Adam", disable=not progress)
    for _ in pbar:
        opt.zero_grad()
        loss = loss_fn()
        loss.backward()
        opt.step()
        pbar.set_postfix(loss=loss.item())
    
    # Stage 2: LBFGS
    if progress:
        print("MLP Architecture - LBFGS")
        print("-" * 30)
    
    opt_lbfgs = optim.LBFGS(
        net.parameters(),
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
    
    # Generate final coefficients
    with torch.no_grad():
        coeff_list_final: list[torch.Tensor] = []
        prefix = torch.empty(0, dtype=dtype, device=device)
        
        for _ in range(N + 1):
            coef = net(prefix)
            coeff_list_final.append(coef)
            prefix = torch.stack(coeff_list_final)
    
    return torch.stack(coeff_list_final).detach().cpu()


if __name__ == "__main__":
    # Test on Airy equation
    DTYPE = torch.float64
    x = sp.symbols("x")
    
    print("EXPERIMENT 1: MLP vs Transformer")
    print("=" * 40)
    
    # Airy equation
    c_airy = [-x, sp.Integer(0), sp.Integer(1)]
    f_airy = sp.Integer(0)
    Ai0 = 0.35502805388781723926
    Aip0 = -0.25881940379280679840
    bcs_airy = [(0.0, 0, Ai0), (0.0, 1, Aip0)]
    
    print("\nTesting different MLP architectures:")
    
    # Test different architectures
    architectures = [
        {"name": "Small MLP", "hidden_dims": [128, 128], "dropout": 0.0},
        {"name": "Medium MLP", "hidden_dims": [256, 256], "dropout": 0.1},
        {"name": "Large MLP", "hidden_dims": [512, 512], "dropout": 0.1},
        {"name": "Deep MLP", "hidden_dims": [256, 256, 256], "dropout": 0.1},
    ]
    
    for arch in architectures:
        print(f"\n{arch['name']}")
        print("-" * 30)
        
        coeffs = train_mlp_pinn(
            c_airy,
            f_airy,
            bcs_airy,
            N=20,
            x_left=-0.5,
            x_right=0.5,
            num_collocation=1000,
            bc_weight=100.0,
            recurrence_weight=100.0,
            dtype=DTYPE,
            adam_iters=500,
            lbfgs_iters=10,
            hidden_dims=arch["hidden_dims"],
            dropout=arch["dropout"],
            progress=True,
        )
        
        print(f"✓ {arch['name']} completed")
        print(f"First few coefficients: {coeffs[:3]}")
        print()