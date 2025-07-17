"""
Experiment 3: Improved MLP with better initialization and normalization
"""

import math
import sympy as sp
import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import trange
from single_ode import make_recurrence, _poly_eval


class ImprovedMLP(nn.Module):
    """Improved MLP with better initialization and normalization"""
    
    def __init__(self, max_input_len: int = 50, hidden_dims: list[int] = [128, 128], 
                 dropout: float = 0.1, init_scale: float = 0.1, use_layernorm: bool = False):
        super().__init__()
        self.max_input_len = max_input_len
        self.init_scale = init_scale
        
        # Build MLP layers
        layers = []
        in_dim = max_input_len
        
        for i, hidden_dim in enumerate(hidden_dims):
            layer = nn.Linear(in_dim, hidden_dim)
            layers.append(layer)
            
            if use_layernorm:
                layers.append(nn.LayerNorm(hidden_dim))
                
            layers.append(nn.ReLU())
            
            if dropout > 0:
                layers.append(nn.Dropout(dropout))
                
            in_dim = hidden_dim
        
        # Output layer with special initialization
        self.output_layer = nn.Linear(in_dim, 1)
        
        # Store main layers for custom initialization
        self.main_layers = nn.Sequential(*layers)
        
        # Initialize weights
        self._init_weights()
        
    def _init_weights(self):
        """Custom weight initialization"""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                # Xavier initialization with custom scaling
                nn.init.xavier_uniform_(module.weight, gain=self.init_scale)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0.0)
        
        # Special initialization for output layer
        nn.init.xavier_uniform_(self.output_layer.weight, gain=self.init_scale * 0.1)
        nn.init.constant_(self.output_layer.bias, 0.0)
        
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
        
        # Apply input normalization
        x = self._normalize_input(x)
        
        # Forward pass
        x = self.main_layers(x)
        return self.output_layer(x).squeeze(-1)
    
    def _normalize_input(self, x: torch.Tensor) -> torch.Tensor:
        """Normalize input to help with different coefficient scales"""
        # Robust normalization: avoid division by zero
        std = torch.std(x)
        if std > 1e-8:
            x = (x - torch.mean(x)) / (std + 1e-8)
        return x


def train_improved_mlp_pinn(
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
    hidden_dims: list[int] = [128, 128],
    dropout: float = 0.1,
    init_scale: float = 0.1,
    use_layernorm: bool = False,
    learning_rate: float = 1e-3,
    adaptive_lr: bool = True,
) -> torch.Tensor:
    """Train improved MLP-based PINN"""
    
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
    
    # Create improved MLP model
    net = ImprovedMLP(
        max_input_len=N + 1,
        hidden_dims=hidden_dims,
        dropout=dropout,
        init_scale=init_scale,
        use_layernorm=use_layernorm
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
    
    # Stage 1: Adam with possibly adaptive learning rate
    opt = optim.AdamW(net.parameters(), lr=learning_rate)
    
    if adaptive_lr:
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            opt, mode='min', factor=0.5, patience=100, verbose=False
        )
    
    if progress:
        print("Improved MLP Architecture - Adam")
        print("-" * 35)
    
    pbar = trange(adam_iters, desc="Adam", disable=not progress)
    for _ in pbar:
        opt.zero_grad()
        loss = loss_fn()
        loss.backward()
        opt.step()
        
        if adaptive_lr:
            scheduler.step(loss)
            
        pbar.set_postfix(loss=loss.item())
    
    # Stage 2: LBFGS
    if progress:
        print("Improved MLP Architecture - LBFGS")
        print("-" * 35)
    
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
    # Test different initialization strategies
    DTYPE = torch.float64
    x = sp.symbols("x")
    
    print("EXPERIMENT 3: Improved MLP with Better Initialization")
    print("=" * 55)
    
    # Test on Airy equation first
    c_airy = [-x, sp.Integer(0), sp.Integer(1)]
    f_airy = sp.Integer(0)
    Ai0 = 0.35502805388781723926
    Aip0 = -0.25881940379280679840
    bcs_airy = [(0.0, 0, Ai0), (0.0, 1, Aip0)]
    
    print("\nTesting different initialization strategies on Airy:")
    
    # Test different configurations
    configs = [
        {"name": "Small init", "init_scale": 0.01, "use_layernorm": False},
        {"name": "Medium init", "init_scale": 0.1, "use_layernorm": False},
        {"name": "Large init", "init_scale": 1.0, "use_layernorm": False},
        {"name": "Medium + LayerNorm", "init_scale": 0.1, "use_layernorm": True},
        {"name": "Medium + Adaptive LR", "init_scale": 0.1, "adaptive_lr": True},
    ]
    
    for config in configs:
        print(f"\n{config['name']}")
        print("-" * 30)
        
        # Set defaults
        init_scale = config.get("init_scale", 0.1)
        use_layernorm = config.get("use_layernorm", False)
        adaptive_lr = config.get("adaptive_lr", False)
        
        coeffs = train_improved_mlp_pinn(
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
            hidden_dims=[128, 128],
            dropout=0.0,
            init_scale=init_scale,
            use_layernorm=use_layernorm,
            adaptive_lr=adaptive_lr,
            progress=True,
        )
        
        print(f"✓ {config['name']} completed")
        print(f"First few coefficients: {coeffs[:3]}")
        print()