"""
Experiment 5: Improved RNN with better normalization strategies
Based on analysis that current normalization works for Airy but fails for other ODEs
"""

import math
import sympy as sp
import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import trange
from single_ode import make_recurrence, _poly_eval


class ImprovedRNNCoeffNet(nn.Module):
    """Improved RNN with better normalization strategies"""
    
    def __init__(self, hidden_size: int = 128, num_layers: int = 1, 
                 rnn_type: str = "GRU", init_scale: float = 0.1,
                 normalization_type: str = "adaptive", dropout: float = 0.0):
        super().__init__()
        
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.rnn_type = rnn_type
        self.init_scale = init_scale
        self.normalization_type = normalization_type
        
        # Input projection (coefficient -> hidden_size)
        self.input_proj = nn.Linear(1, hidden_size)
        
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
        
        # Output head - small as in MLP experiments
        self.output_head = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 2),
            nn.ReLU(),
            nn.Linear(hidden_size // 2, 1)
        )
        
        # Initialize weights
        self._init_weights()
    
    def _init_weights(self):
        """Apply initialization insights from MLP experiments"""
        for name, param in self.named_parameters():
            if 'weight' in name:
                if len(param.shape) >= 2:  # Weight matrices
                    nn.init.xavier_uniform_(param, gain=self.init_scale)
                else:  # Bias vectors
                    nn.init.constant_(param, 0.0)
            elif 'bias' in name:
                nn.init.constant_(param, 0.0)
        
        # Special initialization for output head (even smaller)
        for module in self.output_head.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight, gain=self.init_scale * 0.1)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0.0)
    
    def forward(self, prefix: torch.Tensor) -> torch.Tensor:
        """Predict next coefficient from variable-length prefix"""
        if prefix.ndim != 1:
            raise ValueError("prefix must be a 1-D tensor")
        
        device = prefix.device
        dtype = prefix.dtype
        
        if prefix.numel() == 0:
            # Empty prefix - return prediction from zero hidden state
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
            
            # Use hidden state directly for prediction
            if self.rnn_type == "LSTM":
                final_hidden = hidden[0][-1]  # Last layer hidden state
            else:
                final_hidden = hidden[-1]  # Last layer hidden state
            
            output = self.output_head(final_hidden)
            if output.ndim > 1:
                output = output.squeeze(-1)
            return output
        
        # Apply different normalization strategies
        if self.normalization_type == "none":
            normalized_prefix = prefix
        elif self.normalization_type == "standard":
            normalized_prefix = self._standard_normalize(prefix)
        elif self.normalization_type == "robust":
            normalized_prefix = self._robust_normalize(prefix)
        elif self.normalization_type == "adaptive":
            normalized_prefix = self._adaptive_normalize(prefix)
        else:
            raise ValueError(f"Unknown normalization type: {self.normalization_type}")
        
        # Project coefficients to hidden size
        # Shape: [seq_len, 1] -> [1, seq_len, hidden_size]
        input_proj = self.input_proj(normalized_prefix.unsqueeze(-1))  # [seq_len, hidden_size]
        input_proj = input_proj.unsqueeze(0)  # [1, seq_len, hidden_size]
        
        # Process through RNN
        rnn_out, hidden = self.rnn(input_proj)
        
        # Use final hidden state for prediction
        final_hidden = rnn_out[0, -1, :]  # [hidden_size]
        
        # Generate prediction - be careful with squeeze
        output = self.output_head(final_hidden)
        if output.ndim > 1:
            output = output.squeeze(-1)
        return output
    
    def _standard_normalize(self, x: torch.Tensor) -> torch.Tensor:
        """Standard normalization (current approach)"""
        if x.numel() <= 1:
            return x
        
        mean = torch.mean(x)
        std = torch.std(x)
        if std > 1e-8:
            x = (x - mean) / (std + 1e-8)
        return x
    
    def _robust_normalize(self, x: torch.Tensor) -> torch.Tensor:
        """Robust normalization using median and MAD"""
        if x.numel() <= 1:
            return x
        
        # Use median instead of mean
        median = torch.median(x)
        mad = torch.median(torch.abs(x - median))
        
        if mad > 1e-8:
            x = (x - median) / (mad + 1e-8)
        return x
    
    def _adaptive_normalize(self, x: torch.Tensor) -> torch.Tensor:
        """Adaptive normalization based on sequence characteristics"""
        if x.numel() <= 1:
            return x
        
        # Check if sequence is mostly zeros
        nonzero_count = torch.sum(torch.abs(x) > 1e-8)
        total_count = x.numel()
        
        if nonzero_count < total_count * 0.2:
            # Mostly zeros - use a different strategy
            # Scale by maximum absolute value instead of std
            max_abs = torch.max(torch.abs(x))
            if max_abs > 1e-8:
                return x / (max_abs + 1e-8)
            return x
        else:
            # Normal case - use standard normalization
            return self._standard_normalize(x)


def train_improved_rnn_pinn(
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
    normalization_type: str = "adaptive",
    dropout: float = 0.0,
    learning_rate: float = 1e-3,
) -> torch.Tensor:
    """Train improved RNN-based PINN with better normalization"""
    
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
    
    # Create improved RNN model
    net = ImprovedRNNCoeffNet(
        hidden_size=hidden_size,
        num_layers=num_layers,
        rnn_type=rnn_type,
        init_scale=init_scale,
        normalization_type=normalization_type,
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
                prefix = torch.cat(coeff_list)  # Use cat instead of stack to preserve 1D
            
            next_coef = net(prefix)
            coeff_list.append(next_coef)
        
        coeffs = torch.cat(coeff_list)  # Each coeff is [1], so cat gives [N+1]
        
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
    
    # Training with MLP insights
    opt = optim.AdamW(net.parameters(), lr=learning_rate)
    
    if progress:
        print(f"Improved {rnn_type} ({normalization_type} norm) - Adam")
        print("-" * 50)
    
    pbar = trange(adam_iters, desc="Adam", disable=not progress)
    for _ in pbar:
        opt.zero_grad()
        loss = loss_fn()
        loss.backward()
        opt.step()
        pbar.set_postfix(loss=loss.item())
    
    # LBFGS
    if progress:
        print(f"Improved {rnn_type} ({normalization_type} norm) - LBFGS")
        print("-" * 50)
    
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
    
    # Generate final coefficients and compute final loss
    with torch.no_grad():
        coeff_list_final: list[torch.Tensor] = []
        
        for i in range(N + 1):
            if i == 0:
                prefix = torch.empty(0, dtype=dtype, device=device)
            else:
                prefix = torch.cat(coeff_list_final)  # Use cat instead of stack to preserve 1D
            
            coef = net(prefix)
            coeff_list_final.append(coef)
        
        # Compute final loss
        final_loss = loss_fn()
        
        # Always print final loss regardless of progress setting
        print(f"Final Loss: {final_loss.item():.2e}")
        print()
    
    return torch.cat(coeff_list_final).detach().cpu(), final_loss.item()


if __name__ == "__main__":
    # Test different normalization strategies
    DTYPE = torch.float64
    x = sp.symbols("x")
    
    print("EXPERIMENT 5: Improved RNN with Better Normalization")
    print("=" * 55)
    
    # Test on Legendre equation first (known to fail)
    c_legendre = [5 * 6, -2 * x, 1 - x**2]
    f_legendre = sp.Integer(0)
    bcs_legendre = [(0.0, 0, 0.0), (1.0, 0, 1.0)]
    
    print("\\nTesting different normalization strategies on Legendre:")
    
    normalization_types = ["none", "standard", "robust", "adaptive"]
    
    for norm_type in normalization_types:
        print(f"\\n{norm_type.upper()} Normalization")
        print("-" * 30)
        
        coeffs = train_improved_rnn_pinn(
            c_legendre,
            f_legendre,
            bcs_legendre,
            N=20,
            x_left=-0.5,
            x_right=0.5,
            num_collocation=1000,
            bc_weight=100.0,
            recurrence_weight=100.0,
            dtype=DTYPE,
            adam_iters=300,
            lbfgs_iters=10,
            hidden_size=128,
            rnn_type="GRU",
            normalization_type=norm_type,
            init_scale=0.1,
            dropout=0.0,
            progress=False,
        )
        
        print(f"First few coefficients: {coeffs[:5]}")
        print()