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

# -----------------------------------------------------------------------------
# New helpers
# -----------------------------------------------------------------------------


def factorial_tensor(
    n: int, *, dtype: torch.dtype, device: torch.device
) -> torch.Tensor:
    """Return tensor ``[0!, 1!, …, n!]`` on *device* with the requested *dtype*."""
    return torch.tensor(
        [math.factorial(i) for i in range(n + 1)], dtype=dtype, device=device
    )


# keep c_m away from zero to avoid division blow-ups in the recurrence term


def sample_batch(
    B: int,
    m: int,
    c_range: tuple[float, float],
    bc_range: tuple[float, float],
    *,
    dtype: torch.dtype,
    device: torch.device,
    cm_min_abs: float = 0.3,  # minimum |c_m| (last coefficient) allowed
) -> torch.Tensor:
    """Sample a batch of constant-coefficient ODEs + initial data.

    The returned tensor has shape ``(B, 2*m + 1)`` and layout
    ``[c_0 … c_m , y(0), …, y^(m-1)(0)]``.
    """
    c = (
        torch.rand(B, m + 1, dtype=dtype, device=device) * (c_range[1] - c_range[0])
        + c_range[0]
    )

    # Enforce |c_m| >= cm_min_abs to keep the recurrence well conditioned
    # ------------------------------------------------------------------
    cm = c[:, -1]
    too_small = cm.abs() < cm_min_abs
    if too_small.any():
        n_small = int(too_small.sum().item())

        # Sample new magnitudes in [cm_min_abs, c_range[1]]
        magnitudes = cm_min_abs + torch.rand(n_small, dtype=dtype, device=device) * (
            c_range[1] - cm_min_abs
        )
        signs = torch.where(torch.rand(n_small, device=device) < 0.5, -1.0, 1.0)
        cm[too_small] = signs * magnitudes
        c[:, -1] = cm

    bc = (
        torch.rand(B, m, dtype=dtype, device=device) * (bc_range[1] - bc_range[0])
        + bc_range[0]
    )
    return torch.cat([c, bc], dim=-1)


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

    # Support broadcasting over leading dimensions of ``coeffs``
    N = coeffs.shape[-1] - 1
    if shift > N:
        return torch.zeros(
            (*coeffs.shape[:-1], xs.shape[0]), dtype=coeffs.dtype, device=coeffs.device
        )

    exp_range = torch.arange(0, N + 1 - shift, device=xs.device, dtype=xs.dtype)

    # x^n  (shape  (P, N+1-shift) ) with P=len(xs)
    powers = xs.unsqueeze(-1) ** exp_range
    powers = powers.unsqueeze(0)  # (1, P, N+1-shift)

    coeff_slice = coeffs[..., shift:].unsqueeze(-2)  # (..., 1, N+1-shift)

    numer = fact[shift:]
    denom = fact[: N + 1 - shift]
    ratio = (numer / denom).unsqueeze(0).unsqueeze(0)  # (1, 1, N+1-shift)

    return (powers * coeff_slice * ratio).sum(dim=-1)


# -----------------------------------------------------------------------------
# IVP coefficient predictor
# -----------------------------------------------------------------------------


class IVPCoeffNet(nn.Module):
    """Predict full (N+1) Maclaurin coefficients from
    ``[c_0 … c_m , y(0), y'(0)…y^(m-1)(0)]``.
    """

    def __init__(
        self, m: int, N: int, hidden: int = 256, dtype: torch.dtype = torch.float32
    ):
        super().__init__()
        in_dim = 2 * m + 1
        out_dim = N + 1
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden, dtype=dtype),
            nn.GELU(),
            nn.Linear(hidden, hidden, dtype=dtype),
            nn.GELU(),
            nn.Linear(hidden, hidden, dtype=dtype),
            nn.GELU(),
            nn.Linear(hidden, hidden, dtype=dtype),
            nn.GELU(),
            nn.Linear(hidden, hidden, dtype=dtype),
            nn.GELU(),
            nn.Linear(hidden, out_dim, dtype=dtype),
        )
        self.m = m

    def forward(
        self,
        c_bc: torch.Tensor,
        *,
        freeze_seeds: bool,
        fact: torch.Tensor,
    ) -> torch.Tensor:
        a = self.net(c_bc)
        if freeze_seeds:
            bc = c_bc[..., self.m + 1 :]
            seeds = bc / fact[: self.m]  # a_k = y^(k)(0)/k!
            a = torch.cat([seeds.detach(), a[..., self.m :]], dim=-1)
        return a


# -----------------------------------------------------------------------------
# Main entry point
# -----------------------------------------------------------------------------


def train_power_series_pinn(
    c_list: list[sp.Expr],
    f_expr: sp.Expr,
    bc_tuples: list[tuple[float, int, float]] | None = None,
    *,
    N: int = 10,
    x_left: float = 0.0,
    x_right: float = 1.0,
    num_collocation: int = 1000,
    bc_weight: float = 100.0,
    recurrence_weight: float = 1.0,
    c_range: tuple[float, float] = (-1.0, 1.0),
    bc_range: tuple[float, float] = (-1.0, 1.0),
    freeze_seeds: bool = True,
    num_batches: int = 10_000,
    batch_size: int = 128,
    dtype: torch.dtype = torch.float32,
    device: torch.device | None = None,
    seed: int = 1234,
    progress: bool = True,
) -> "IVPCoeffNet":
    """Train an *IVP-PINN* that maps ODE/BC data to Maclaurin coefficients.

    The network is trained on randomly-sampled **constant-coefficient** IVPs
    within user-specified ranges.  When *freeze_seeds* is *True*, the first
    *m* series coefficients are forced to match the supplied initial data and
    are therefore not updated through back-prop.
    """

    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    torch.manual_seed(seed)

    # Factorials 0!, …, N!
    fact = factorial_tensor(N, dtype=dtype, device=device)

    # Collocation grid and inhomogeneous term
    xs_coll_np = np.linspace(x_left, x_right, num_collocation)
    xs_collocation = torch.tensor(xs_coll_np, dtype=dtype, device=device)

    x_sym = sp.symbols("x")
    f_func = sp.lambdify(x_sym, f_expr)
    f_vals = torch.tensor(f_func(xs_coll_np), dtype=dtype, device=device)

    # Problem order and network
    m = len(c_list) - 1
    net = IVPCoeffNet(m, N, dtype=dtype).to(device)

    # ------------------------------------------------------------------
    def compute_loss(c_bc: torch.Tensor) -> torch.Tensor:
        """Compute composite loss for a batch of IVPs."""

        a_pred = net(c_bc, freeze_seeds=freeze_seeds, fact=fact)  # (B, N+1)

        c = c_bc[:, : m + 1]  # (B, m+1)
        bc = c_bc[:, m + 1 :]  # (B, m)

        # PDE residual --------------------------------------------------
        u_ks = torch.stack(
            [_poly_eval(xs_collocation, a_pred, fact, shift=k) for k in range(m + 1)],
            dim=0,
        )  # (m+1, B, P)

        # c shape: (B, m+1), need to reshape to (m+1, B, 1) for broadcasting
        c_reshaped = c.T.unsqueeze(-1)  # (m+1, B, 1)
        residual = (c_reshaped * u_ks).sum(dim=0) - f_vals.unsqueeze(0)  # (B, P)
        loss_pde = (residual**2).mean()

        # Initial-value mismatch --------------------------------------
        if freeze_seeds:
            loss_bc = torch.tensor(0.0, device=device)
        else:
            seeds_target = bc / fact[:m]
            loss_bc = ((a_pred[:, :m] - seeds_target) ** 2).mean()

        # Recurrence consistency --------------------------------------
        if recurrence_weight != 0.0:
            rec_terms = []
            for ell in range(m, N + 1):
                n = ell - m
                ratios = fact[n : n + m] / fact[ell]  # (m,)
                num = (c[:, :m] * a_pred[:, n : n + m] * ratios).sum(dim=1)
                a_calc = -num / c[:, m]
                rec_terms.append((a_pred[:, ell] - a_calc) ** 2)
            loss_rec = torch.stack(rec_terms).mean()
        else:
            loss_rec = torch.tensor(0.0, device=device)

        return loss_pde + bc_weight * loss_bc + recurrence_weight * loss_rec

    # ------------------------------------------------------------------
    opt = optim.AdamW(net.parameters(), lr=1e-5)
    if progress:
        print("\nTraining IVP-PINN (Adam)\n-------------------------")

    pbar = trange(num_batches, desc="Adam", disable=not progress)
    for _ in pbar:
        c_bc = sample_batch(
            batch_size, m, c_range, bc_range, dtype=dtype, device=device
        )
        opt.zero_grad(set_to_none=True)
        loss_val = compute_loss(c_bc)
        loss_val.backward()
        opt.step()
        pbar.set_postfix(loss=loss_val.item())

    net_cpu = net.to("cpu").eval()
    return net_cpu


# -----------------------------------------------------------------------------
# Inference helper
# -----------------------------------------------------------------------------


def solve_ivp(
    c: torch.Tensor,
    bc: torch.Tensor,
    net: IVPCoeffNet,
    fact: torch.Tensor,
    *,
    freeze_seeds: bool = True,
) -> torch.Tensor:
    """Predict Maclaurin coefficients for a single IVP using *net*."""

    c_bc = torch.cat([c, bc]).unsqueeze(0)  # (1, 2m+1)
    with torch.no_grad():
        a = net(c_bc, freeze_seeds=freeze_seeds, fact=fact)[0]
    return a
