"""
Utilities for training a power-series Physics-Informed Neural Network (PINN)
solver for linear ODEs of the form::

    sum_{k=0}^m c_k(x) y^{(k)}(x) = f(x),      x ∈ [x_left, x_right]

subject to generic boundary/initial conditions supplied as
```
    (x0, k, value)  →  y^{(k)}(x0) = value
```
where `k` is the derivative order.

The solution is represented as a truncated Maclaurin series

    y(x) ≈ Σ_{n=0}^N a_n x^n / n!

with the coefficients `a = (a_0, …, a_N)` optimized by a two-stage training
procedure.
"""

import math

import numpy as np
import sympy as sp
import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import trange


def make_recurrence(
    c_list: list[sp.Expr],
    f_expr: sp.Expr,
    max_n: int,
    *,
    dtype: torch.dtype = torch.float32,
    device: torch.device | None = None,
):
    """Create a differentiable recurrence *s_ℓ = Φ(s₀,…,s_{ℓ-1})*.

    Parameters
    ----------
    c_list
        ``c_list[k]`` is the SymPy expression *c_k(x)* for *k = 0 … m*.
    f_expr
        Inhomogeneous term *f(x)*.
    max_n
        Maximum truncation order for the recurrence.
    dtype, device
        Torch dtype / device for the returned callable.

    Returns
    -------
    next_coef : callable(torch.Tensor) -> torch.Tensor
        Given a 1-D tensor ``s_prev`` containing *(s₀,…,s_{ℓ-1})* with
        ``ℓ ≥ m``, returns the scalar tensor ``s_ℓ`` while preserving
        autograd through the input tensor.
    """

    x = sp.symbols("x")
    m = len(c_list) - 1

    # --- regular-point check ---------------------------------------------
    c_m0 = sp.N(c_list[m].subs(x, 0))
    if c_m0 == 0:
        raise ValueError("x = 0 is a singular point (c_m(0)=0).")
    c_m0_t = torch.tensor(float(c_m0), dtype=dtype, device=device)

    # Pre-compute factorials up to (max_n + m)
    fact_np = np.array([math.factorial(i) for i in range(max_n + m + 1)], dtype=float)

    # Pre-compute Maclaurin coefficients c_{k,j} for j ≤ max_n
    ckj_arr = np.zeros((m + 1, max_n + 1), dtype=float)
    for k in range(m + 1):
        for j in range(max_n + 1):
            ckj_arr[k, j] = float(sp.series(c_list[k], x, 0, j + 1).coeff(x, j))

    # Pre-compute b_n (RHS series coefficients)
    bn_arr = np.array(
        [float(sp.series(f_expr, x, 0, n + 1).coeff(x, n)) for n in range(max_n + 1)],
        dtype=float,
    )
    bn_t = torch.tensor(bn_arr, dtype=dtype, device=device)  # (max_n+1,)

    # Pre-compute factorial ratios and overall constants
    # We store a list of length (max_n+1); entry n is tensor of shape (m+1, n+1)
    coef_consts: list[list[torch.Tensor]] = []
    idx_tensors: list[list[torch.Tensor]] = []

    for n in range(max_n + 1):
        coef_n: list[torch.Tensor] = []
        idx_n: list[torch.Tensor] = []
        j_range = np.arange(0, n + 1)

        fact_base = fact_np[n - j_range]  # (n+1,)
        for k in range(m + 1):
            # Skip (k=m, j=0) later when summing
            fact_num = fact_np[n - j_range + k]
            ratio = fact_num / fact_base  # (n+1,)
            consts = ckj_arr[k, : n + 1] * ratio  # (n+1,)
            coef_n.append(torch.tensor(consts, dtype=dtype, device=device))

            idx = torch.tensor(n - j_range + k, dtype=torch.long, device=device)
            idx_n.append(idx)

        coef_consts.append(coef_n)
        idx_tensors.append(idx_n)

    pref_arr = np.array(
        [math.factorial(n) / math.factorial(n + m) for n in range(max_n + 1)],
        dtype=float,
    )
    pref_t = torch.tensor(pref_arr, dtype=dtype, device=device)

    # ------------------------------------------------------------------
    def next_coef(s_prev: torch.Tensor) -> torch.Tensor:  # type: ignore[override]
        """Compute *s_ℓ* from *s_prev = [s₀,…,s_{ℓ-1}]* with autograd."""

        if s_prev.ndim != 1:
            raise ValueError("s_prev must be a 1-D tensor.")
        ell = s_prev.shape[0]
        if ell < m:
            raise ValueError(f"Need at least {m} initial coefficients.")

        n = ell - m
        if n > max_n:
            raise ValueError(
                f"Requested coefficient index {ell} exceeds pre-computed max_n={max_n}."
            )

        T = torch.zeros((), dtype=dtype, device=s_prev.device)

        coef_n = coef_consts[n]
        idx_n = idx_tensors[n]

        for k in range(m + 1):
            if k == m:
                consts = coef_n[k][1:]
                idxs = idx_n[k][1:]
            else:
                consts = coef_n[k]
                idxs = idx_n[k]

            T = T + (consts * s_prev[idxs]).sum()

        num = bn_t[n] - T
        return (pref_t[n] / c_m0_t) * num

    return next_coef


def _poly_eval(
    xs: torch.Tensor, coeffs: torch.Tensor, fact: torch.Tensor, shift: int = 0
) -> torch.Tensor:
    """
    coeffs are the *series coefficients* s_n such that
        y(x) = Σ s_n x^n .
    The k-th derivative is
        y^{(k)}(x) = Σ_{n=k} s_n · n! / (n-k)! · x^{n-k}
    """

    N = coeffs.shape[0] - 1
    if shift > N:
        return torch.zeros_like(xs)

    a_range = torch.arange(
        0, N + 1 - shift, device=xs.device, dtype=xs.dtype
    )  # (0,…,N-shift)
    powers = xs.unsqueeze(1) ** a_range  # (B, N+1-shift)

    coeff_slice = coeffs[shift:]  # (N+1-shift,)

    # Factorial ratio  n! / (n-k)!  with  n = shift + i  and  i = a_range
    numer = fact[shift:]
    denom = fact[: N + 1 - shift]
    ratio = numer / denom  # (N+1-shift,)

    return (powers * coeff_slice * ratio).sum(dim=1)  # (B,)


class _SinusoidalPositionalEncoding(nn.Module):
    """Standard sinusoidal positional encoding (à la Vaswani et al.)."""

    def __init__(self, d_model: int, max_len: int = 5000):
        super().__init__()

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2, dtype=torch.float)
            * (-math.log(10000.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)

        # Register as buffer so it moves with the module but is not trainable
        self.register_buffer("pe", pe, persistent=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # type: ignore[override]
        """Add positional encodings to *x* (shape ``seq_len × batch × d_model``)."""

        seq_len = x.size(0)
        # Slicing a registered buffer confuses static checkers → ignore
        return x + self.pe[:seq_len].unsqueeze(1).type_as(x)  # type: ignore[index]


class _ARTransformerCoeffNet(nn.Module):
    """Causal Transformer that maps a prefix of coefficients to the next one."""

    def __init__(
        self,
        *,
        d_model: int = 128,
        n_layers: int = 2,
        nhead: int = 8,
        dim_feedforward: int = 512,
        dropout: float = 0.1,
        max_len: int = 128,
        dtype: torch.dtype = torch.float32,
    ):
        super().__init__()

        self.d_model = d_model

        # Learnable start token to bootstrap generation of a_0
        self.start_token = nn.Parameter(torch.zeros(d_model, dtype=dtype))

        # Embed a scalar coefficient → d_model
        self.coeff_proj = nn.Linear(1, d_model, bias=True)

        # Positional encoding
        self.pos_enc = _SinusoidalPositionalEncoding(d_model, max_len=max_len)

        enc_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=False,
        )
        self.transformer = nn.TransformerEncoder(enc_layer, num_layers=n_layers)

        # Final head → scalar coefficient
        self.head = nn.Linear(d_model, 1)

    # ------------------------------------------------------------------
    def forward(self, prefix: torch.Tensor) -> torch.Tensor:  # type: ignore[override]
        """Predict *a_k* given ``prefix = [a₀,…,a_{k-1}]`` (1-D tensor)."""

        if prefix.ndim != 1:
            raise ValueError("prefix must be a 1-D tensor of coefficients.")

        device = prefix.device if prefix.numel() else self.start_token.device
        dtype = prefix.dtype if prefix.numel() else self.start_token.dtype

        seq_len = prefix.shape[0] + 1  # start token + prefix

        # Allocate token embeddings (seq_len × batch × d_model) with batch=1
        tokens = torch.empty(seq_len, 1, self.d_model, device=device, dtype=dtype)

        # First token: learnable start symbol
        tokens[0, 0, :] = self.start_token.to(dtype=dtype)

        # Remaining tokens: embedded coefficients
        if prefix.numel() > 0:
            coeff_emb = self.coeff_proj(prefix.unsqueeze(1))  # (k × d_model)
            tokens[1:, 0, :] = coeff_emb

        # Add positional encoding
        tokens = self.pos_enc(tokens)

        # Causal mask is not required because we observe only past tokens
        h = self.transformer(tokens)  # (seq_len × 1 × d_model)

        h_last = h[-1, 0, :]  # hidden state of last observed token
        next_coeff = self.head(h_last).squeeze(0)
        return next_coeff


# -----------------------------------------------------------------------------
# Main entry point
# -----------------------------------------------------------------------------


def train_power_series_pinn(
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
    # Network configuration -------------------------------------------------
    d_model: int = 128,
    n_layers: int = 2,
    nhead: int = 8,
    dim_feedforward: int = 512,
    dropout: float = 0.0,
) -> torch.Tensor:
    """Train a power-series PINN and return the learned coefficients *a*.

    Parameters
    ----------
    c_list, f_expr
        ODE definition.  The highest derivative order is *m = len(c_list)-1*.
    bc_tuples
        Iterable of boundary/initial conditions as *(x0, k, value)*.
    N
        Truncation order of the Maclaurin series.
    x_left, x_right
        Spatial domain for collocation.
    num_collocation
        Number of interior collocation points.
    bc_weight
        Relative weight of the BC loss term.
    adam_iters, lbfgs_iters
        Optimiser iterations for the two-stage training procedure.
    recurrence_weight
        Relative weight of the recurrence consistency loss term.
    dtype, device, seed
        Standard PyTorch training knobs.
    progress
        If *True*, display progress bars.

    Returns
    -------
    torch.Tensor
        The learned coefficient vector ``a`` of shape *(N+1,)* (on CPU).
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    torch.manual_seed(seed)

    # Pre-compute factorials 0!, …, N!
    fact = torch.tensor(
        [math.factorial(i) for i in range(N + 1)], dtype=dtype, device=device
    )

    # Collocation points (uniform grid)
    xs_coll_np = np.linspace(x_left, x_right, num_collocation)
    xs_collocation = torch.tensor(xs_coll_np, dtype=dtype, device=device)

    # Lambdify coefficient functions & RHS on numpy, then convert once to torch
    x_sym = sp.symbols("x")
    c_funcs = [sp.lambdify(x_sym, c) for c in c_list]
    f_func = sp.lambdify(x_sym, f_expr)

    c_vals = [
        torch.tensor(cf(xs_coll_np), dtype=dtype, device=device) for cf in c_funcs
    ]
    f_vals = torch.tensor(f_func(xs_coll_np), dtype=dtype, device=device)

    m = len(c_list) - 1  # highest derivative order

    # ------------------------------------------------------------------
    # Neural network that *generates* the coefficients
    # ------------------------------------------------------------------

    net = _ARTransformerCoeffNet(
        d_model=d_model,
        n_layers=n_layers,
        nhead=nhead,
        dim_feedforward=dim_feedforward,
        dropout=dropout,
        max_len=N + 2,
        dtype=dtype,
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
        # --- auto-regressively generate coefficients -------------------

        coeff_list: list[torch.Tensor] = []
        prefix = torch.empty(0, dtype=dtype, device=device)
        for _ in range(N + 1):
            next_coef = net(prefix)
            coeff_list.append(next_coef)
            prefix = torch.stack(coeff_list)

        coeffs = torch.stack(coeff_list)  # (N+1,)

        # ODE residual
        u_ks = [_poly_eval(xs_collocation, coeffs, fact, shift=k) for k in range(m + 1)]
        residual = sum(c_vals[k] * u_ks[k] for k in range(m + 1)) - f_vals
        loss_pde = (residual**2).mean()

        # Boundary / initial conditions
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
        # print(loss_pde, loss_bc, loss_rec)

        return loss_pde + bc_weight * loss_bc + recurrence_weight * loss_rec

    # --- stage 1: Adam ------------------------------------------------------
    opt = optim.AdamW(net.parameters(), lr=1e-3)
    if progress:
        print("\nStage 1: Adam optimisation\n--------------------------")
    pbar = trange(adam_iters, desc="Adam", disable=not progress)
    for _ in pbar:
        opt.zero_grad(set_to_none=True)
        l = loss_fn()
        l.backward()
        opt.step()
        pbar.set_postfix(loss=l.item())

    # --- stage 2: LBFGS fine-tuning ----------------------------------------
    if progress:
        print("\nStage 2: LBFGS fine-tuning\n--------------------------")

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
            opt_lbfgs.zero_grad(set_to_none=True)
            loss_val = loss_fn()
            loss_val.backward()
            return loss_val

        loss_val = opt_lbfgs.step(closure)
        pbar.set_postfix(loss=loss_val.item())

    # -----------------------------------------------------------------------
    # Re-generate coefficients once more (no grad) with trained network
    with torch.no_grad():
        coeff_list_final: list[torch.Tensor] = []
        prefix = torch.empty(0, dtype=dtype, device=device)
        for _ in range(N + 1):
            coef = net(prefix)
            coeff_list_final.append(coef)
            prefix = torch.stack(coeff_list_final)

    coeff_learned = torch.stack(coeff_list_final).detach().cpu()
    return coeff_learned
