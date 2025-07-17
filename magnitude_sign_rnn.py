"""
Experiment 12: Magnitude-Sign Decomposition
Instead of predicting coefficient a directly, predict magnitude m and exponent e
such that a = sign * m * 2^e, where sign ∈ {-1, 0, 1}
"""

import math
import sympy as sp
import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import trange
from single_ode import make_recurrence, _poly_eval


class MagnitudeSignRNN(nn.Module):
    """RNN that predicts coefficients via magnitude-sign decomposition"""
    
    def __init__(self, hidden_size: int = 128, num_layers: int = 1, 
                 rnn_type: str = "GRU", init_scale: float = 0.1,
                 dropout: float = 0.0):
        super().__init__()
        
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.rnn_type = rnn_type
        self.init_scale = init_scale
        
        # Input projection - we'll encode coefficients differently
        self.input_proj = nn.Linear(3, hidden_size)  # [sign, log_magnitude, is_zero]
        
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
        
        # Three output heads
        self.sign_head = nn.Linear(hidden_size, 3)  # [-1, 0, 1]
        self.log_magnitude_head = nn.Linear(hidden_size, 1)  # log(|a|)
        self.zero_head = nn.Linear(hidden_size, 2)  # [not_zero, is_zero]
        
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
        """Encode coefficient as [sign, log_magnitude, is_zero]"""
        device = coeff.device
        dtype = coeff.dtype
        
        # Handle zero coefficients
        is_zero = torch.abs(coeff) < 1e-12
        is_zero_float = is_zero.float()
        
        # Sign: -1, 0, 1
        sign = torch.sign(coeff)
        
        # Log magnitude (avoid log(0))
        abs_coeff = torch.abs(coeff)
        log_magnitude = torch.where(
            is_zero, 
            torch.tensor(-20.0, device=device, dtype=dtype),  # Very negative for zero
            torch.log(abs_coeff + 1e-12)
        )
        
        return torch.stack([sign, log_magnitude, is_zero_float], dim=-1)
    
    def decode_prediction(self, sign_logits: torch.Tensor, log_magnitude: torch.Tensor, zero_logits: torch.Tensor) -> torch.Tensor:
        """Decode predictions back to coefficient"""
        # Squeeze tensors to remove batch dimension
        sign_logits = sign_logits.squeeze(0)
        zero_logits = zero_logits.squeeze(0)
        log_magnitude = log_magnitude.squeeze(0)
        
        # Determine if zero
        zero_probs = torch.softmax(zero_logits, dim=0)
        is_zero = zero_probs[1] > 0.5
        
        # Get sign
        sign_probs = torch.softmax(sign_logits, dim=0)
        sign = sign_probs[0] * (-1) + sign_probs[1] * 0 + sign_probs[2] * 1
        
        # Get magnitude
        magnitude = torch.exp(log_magnitude)
        
        # Combine
        coeff = torch.where(is_zero, torch.tensor(0.0, device=sign.device, dtype=sign.dtype), sign * magnitude)
        
        # Ensure coefficient is a scalar tensor
        if coeff.dim() == 0:
            coeff = coeff.unsqueeze(0)
        
        return coeff.squeeze()
    
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
        sign_logits = self.sign_head(final_hidden)
        log_magnitude = self.log_magnitude_head(final_hidden).squeeze(-1)
        zero_logits = self.zero_head(final_hidden)
        
        # Decode to coefficient
        coeff = self.decode_prediction(sign_logits, log_magnitude, zero_logits)
        
        return coeff


def train_magnitude_sign_rnn(
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
    """Train magnitude-sign RNN PINN"""
    
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
    
    # Create magnitude-sign RNN model
    net = MagnitudeSignRNN(
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
        print(f"Magnitude-Sign {rnn_type} - Adam")
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
        print(f"Magnitude-Sign {rnn_type} - LBFGS")
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
    # Test magnitude-sign approach on challenging ODEs
    DTYPE = torch.float64
    x = sp.symbols("x")
    
    print("EXPERIMENT 12: MAGNITUDE-SIGN DECOMPOSITION")
    print("=" * 50)
    print("Approach: Predict sign, log(magnitude), and zero-flag separately")
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
        }
    ]
    
    for test in test_cases:
        print(f"{test['name']} (Current best: {test['current_best']:.2e})")
        print("-" * 50)
        print(f"Expected: {test['expected']}")
        
        coeffs, final_loss = train_magnitude_sign_rnn(
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
            adam_iters=1500,
            lbfgs_iters=30,
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