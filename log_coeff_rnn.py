"""
Experiment 13: Logarithmic Coefficient Representation
Instead of predicting coefficient a directly, predict log(|a|) and sign(a)
This should handle extreme magnitude variations better
"""

import math
import sympy as sp
import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import trange
from single_ode import make_recurrence, _poly_eval


class LogCoeffRNN(nn.Module):
    """RNN that predicts coefficients via logarithmic representation"""
    
    def __init__(self, hidden_size: int = 128, num_layers: int = 1, 
                 rnn_type: str = "GRU", init_scale: float = 0.1,
                 dropout: float = 0.0):
        super().__init__()
        
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.rnn_type = rnn_type
        self.init_scale = init_scale
        
        # Input projection - encode coefficients as [log_magnitude, sign]
        self.input_proj = nn.Linear(2, hidden_size)
        
        # RNN layers
        if rnn_type == "LSTM":
            self.rnn = nn.LSTM(
                input_size=hidden_size,
                hidden_size=hidden_size,
                num_layers=num_layers,
                batch_first=True,
                dropout=dropout if num_layers > 1 else 0
            )
        elif rnn_type == "GRU":
            self.rnn = nn.GRU(
                input_size=hidden_size,
                hidden_size=hidden_size,
                num_layers=num_layers,
                batch_first=True,
                dropout=dropout if num_layers > 1 else 0
            )
        else:
            raise ValueError(f"Unknown RNN type: {rnn_type}")
        
        # Output heads
        self.log_magnitude_head = nn.Linear(hidden_size, 1)
        self.sign_head = nn.Linear(hidden_size, 1)  # Will use tanh activation
        
        # Initialize weights
        self._init_weights()
    
    def _init_weights(self):
        """Initialize weights with small scales"""
        for name, param in self.named_parameters():
            if 'weight' in name:
                if len(param.shape) >= 2:
                    nn.init.xavier_uniform_(param, gain=self.init_scale)
                else:
                    nn.init.constant_(param, 0.0)
            elif 'bias' in name:
                nn.init.constant_(param, 0.0)
    
    def encode_coefficient(self, coeff: torch.Tensor) -> torch.Tensor:
        """Encode coefficient as [log_magnitude, sign]"""
        device = coeff.device
        dtype = coeff.dtype
        
        # Handle zero/near-zero coefficients
        abs_coeff = torch.abs(coeff)
        is_zero = abs_coeff < 1e-15
        
        # Log magnitude (use -30 for effectively zero coefficients)
        log_magnitude = torch.where(
            is_zero,
            torch.tensor(-30.0, device=device, dtype=dtype),
            torch.log(abs_coeff + 1e-15)
        )
        
        # Sign (-1 or 1, with 0 mapped to 1)
        sign = torch.where(
            is_zero,
            torch.tensor(1.0, device=device, dtype=dtype),
            torch.sign(coeff)
        )
        
        return torch.stack([log_magnitude, sign], dim=-1)
    
    def decode_prediction(self, log_magnitude: torch.Tensor, sign: torch.Tensor) -> torch.Tensor:
        """Decode predictions back to coefficient"""
        # Apply tanh to sign to get value in [-1, 1]
        sign = torch.tanh(sign)
        
        # Handle very negative log magnitudes as zero
        is_zero = log_magnitude < -20.0
        
        # Get magnitude
        magnitude = torch.where(
            is_zero,
            torch.tensor(0.0, device=log_magnitude.device, dtype=log_magnitude.dtype),
            torch.exp(log_magnitude)
        )
        
        # Combine
        coeff = sign * magnitude
        
        return coeff
    
    def forward(self, prefix: torch.Tensor) -> torch.Tensor:
        """Predict next coefficient from variable-length prefix"""
        if prefix.ndim != 1:
            raise ValueError("prefix must be a 1-D tensor")
        
        device = prefix.device
        dtype = prefix.dtype
        
        if prefix.numel() == 0:
            # Empty prefix - use zero hidden state
            batch_size = 1
            if self.rnn_type == "LSTM":
                h0 = torch.zeros(self.num_layers, batch_size, self.hidden_size, 
                               device=device, dtype=dtype)
                c0 = torch.zeros(self.num_layers, batch_size, self.hidden_size, 
                               device=device, dtype=dtype)
                hidden = (h0, c0)
            else:  # GRU
                hidden = torch.zeros(self.num_layers, batch_size, self.hidden_size, 
                                   device=device, dtype=dtype)
            
            # Use hidden state for prediction
            if self.rnn_type == "LSTM":
                final_hidden = hidden[0][-1]
            else:
                final_hidden = hidden[-1]
        else:
            # Encode prefix coefficients
            encoded_prefix = torch.stack([self.encode_coefficient(coeff) for coeff in prefix])
            
            # Project to hidden size
            input_proj = self.input_proj(encoded_prefix)  # [seq_len, hidden_size]
            input_proj = input_proj.unsqueeze(0)  # [1, seq_len, hidden_size]
            
            # Process through RNN
            rnn_out, hidden = self.rnn(input_proj)
            final_hidden = rnn_out[0, -1, :]  # [hidden_size]
        
        # Generate predictions
        log_magnitude = self.log_magnitude_head(final_hidden).squeeze(-1)
        sign = self.sign_head(final_hidden).squeeze(-1)
        
        # Decode to coefficient
        coeff = self.decode_prediction(log_magnitude, sign)
        
        return coeff


def train_log_coeff_rnn(
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
    # RNN-specific parameters
    hidden_size: int = 128,
    num_layers: int = 1,
    rnn_type: str = "GRU",
    init_scale: float = 0.1,
    dropout: float = 0.0,
    learning_rate: float = 1e-3,
) -> tuple[torch.Tensor, float]:
    """Train logarithmic coefficient RNN PINN"""
    
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    torch.manual_seed(seed)
    
    # Standard setup
    fact = torch.tensor(
        [math.factorial(i) for i in range(N + 1)], dtype=dtype, device=device
    )
    
    import numpy as np
    xs_coll_np = np.linspace(x_left, x_right, num_collocation)
    xs_collocation = torch.tensor(xs_coll_np, dtype=dtype, device=device)
    
    x_sym = sp.symbols("x")
    c_funcs = [sp.lambdify(x_sym, c) for c in c_list]
    f_func = sp.lambdify(x_sym, f_expr)
    
    c_vals = [
        torch.tensor(cf(xs_coll_np), dtype=dtype, device=device) for cf in c_funcs
    ]
    f_vals = torch.tensor(f_func(xs_coll_np), dtype=dtype, device=device)
    
    m = len(c_list) - 1
    
    # Create log coefficient RNN model
    net = LogCoeffRNN(
        hidden_size=hidden_size,
        num_layers=num_layers,
        rnn_type=rnn_type,
        init_scale=init_scale,
        dropout=dropout
    ).to(device=device, dtype=dtype)
    
    # Build recurrence
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
        
        for i in range(N + 1):
            if i == 0:
                prefix = torch.empty(0, dtype=dtype, device=device)
            else:
                prefix = torch.cat(coeff_list)
            
            next_coef = net(prefix)
            # Ensure coefficient is properly shaped for concatenation
            if next_coef.dim() == 0:
                next_coef = next_coef.unsqueeze(0)
            coeff_list.append(next_coef)
        
        coeffs = torch.cat(coeff_list)
        
        # Standard loss computation
        u_ks = [_poly_eval(xs_collocation, coeffs, fact, shift=k) for k in range(m + 1)]
        residual = sum(c_vals[k] * u_ks[k] for k in range(m + 1)) - f_vals
        loss_pde = (residual**2).mean()
        
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
    
    # Training
    opt = optim.AdamW(net.parameters(), lr=learning_rate)
    
    if progress:
        print(f"Log-Coefficient {rnn_type} - Adam")
        print("-" * 40)
    
    pbar = trange(adam_iters, desc="Adam", disable=not progress)
    for _ in pbar:
        opt.zero_grad()
        loss = loss_fn()
        loss.backward()
        opt.step()
        pbar.set_postfix(loss=loss.item())
    
    # LBFGS
    if progress:
        print(f"Log-Coefficient {rnn_type} - LBFGS")
        print("-" * 40)
    
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
        
        for i in range(N + 1):
            if i == 0:
                prefix = torch.empty(0, dtype=dtype, device=device)
            else:
                prefix = torch.cat(coeff_list_final)
            
            coef = net(prefix)
            # Ensure coefficient is properly shaped for concatenation
            if coef.dim() == 0:
                coef = coef.unsqueeze(0)
            coeff_list_final.append(coef)
        
        # Compute final loss
        final_loss = loss_fn()
        
        # Always print final loss regardless of progress setting
        print(f"Final Loss: {final_loss.item():.2e}")
        print()
    
    return torch.cat(coeff_list_final).detach().cpu(), final_loss.item()


if __name__ == "__main__":
    # Test logarithmic coefficient approach on challenging ODEs
    DTYPE = torch.float64
    x = sp.symbols("x")
    
    print("EXPERIMENT 13: LOGARITHMIC COEFFICIENT REPRESENTATION")
    print("=" * 55)
    print("Approach: Predict log(|coeff|) and sign(coeff) separately")
    print("Goal: Handle extreme magnitude variations better")
    print()
    
    # Test on the most challenging ODEs
    test_cases = [
        {
            "name": "Legendre (l=5)",
            "c_list": [5 * 6, -2 * x, 1 - x**2],
            "f_expr": sp.Integer(0),
            "bc_tuples": [(0.0, 0, 0.0), (1.0, 0, 1.0)],
            "N": 25,
            "x_left": -0.9,
            "x_right": 0.9,
            "expected": "First few should be [0,0,0,0,0,30,...]",
            "current_best": 1.38e+00
        },
        {
            "name": "Beam",
            "c_list": [sp.Integer(0), sp.Integer(0), sp.Integer(0), sp.Integer(0), 1 + sp.Rational(3, 10) * x**2],
            "f_expr": sp.sin(2 * x),
            "bc_tuples": [(0.0, 0, 0.0), (0.0, 1, 0.0), (1.0, 2, 0.0), (1.0, 3, 0.0)],
            "N": 25,
            "x_left": 0.0,
            "x_right": 1.0,
            "expected": "First few should be [0,0,0,0,0.25,...]",
            "current_best": 5.84e-01
        },
        {
            "name": "Hermite (n=6)",
            "c_list": [2 * 6, -2 * x, sp.Integer(1)],
            "f_expr": sp.Integer(0),
            "bc_tuples": [(0.0, 0, -120.0), (0.0, 1, 0.0)],
            "N": 25,
            "x_left": -0.9,
            "x_right": 0.9,
            "expected": "First few should be [-120, 0, 720, 0, -120,...]",
            "current_best": 8.39e-03
        }
    ]
    
    for test in test_cases:
        print(f"{test['name']} (Current best: {test['current_best']:.2e})")
        print("-" * 50)
        print(f"Expected: {test['expected']}")
        
        coeffs, final_loss = train_log_coeff_rnn(
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
            adam_iters=500,
            lbfgs_iters=15,
            hidden_size=256,
            num_layers=1,
            rnn_type="GRU",
            init_scale=0.1,
            dropout=0.0,
            learning_rate=1e-3,
            progress=False,
        )
        
        print(f"Coefficients: {coeffs[:8]}")
        
        if final_loss < test["current_best"]:
            improvement = test["current_best"] / final_loss
            print(f"✓ IMPROVEMENT: {improvement:.2f}x better!")
        else:
            print("✗ No improvement")
        
        print()
        
        # Early success check
        if final_loss < 1e-10:
            print("✓ ACHIEVED TARGET LOSS!")
            break