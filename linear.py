import os
import math
from pathlib import Path

import numpy as np
import sympy as sp
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
from tqdm import trange

# -----------------------------------------------------------------------------
# Configuration & constants
# -----------------------------------------------------------------------------

dtype = torch.float32

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

torch.manual_seed(1234)

# Domain and ODE definition ----------------------------------------------------
N: int = 10  # highest power in truncated series
x_left: float = 0.0
x_right: float = 1.0

num_collocation: int = 1000  # collocation points for the PDE term
bc_weight: float = 100.0  # weight for boundary-condition loss
num_supervised: int = 5  # number of coefficients to supervise
supervised_weight: float = 0.0  # weight for supervised coefficient loss

pi_const = math.pi

# Pre-compute factorials 0!, …, N! as torch tensor (for fast broadcasting)
fact = torch.tensor(
    [math.factorial(i) for i in range(N + 1)], dtype=dtype, device=device
)

# -----------------------------------------------------------------------------
# Analytic solution (for supervision & evaluation)
# -----------------------------------------------------------------------------

x_sym = sp.symbols("x")
analytic_expr = (
    sp.pi * x_sym * (-x_sym + (sp.pi**2) * (2 * x_sym - 3) + 1) - sp.sin(sp.pi * x_sym)
) / (sp.pi**3)

_coeff_list = []
for k in range(N + 1):
    # Differentiate k-times, evaluate at x=0 and cast to float
    _val = sp.N(sp.diff(analytic_expr, x_sym, k).subs(x_sym, 0))  # type: ignore[arg-type]
    _coeff_list.append(float(_val))  # type: ignore[arg-type]

a_true = torch.tensor(_coeff_list, dtype=dtype, device=device)
training_data = a_true[:num_supervised]

# Numpy helper for plotting ----------------------------------------------------


def analytic_solution_np(x: np.ndarray) -> np.ndarray:
    return (
        pi_const * x * (-x + (pi_const**2) * (2 * x - 3) + 1) - np.sin(pi_const * x)
    ) / (pi_const**3)


# -----------------------------------------------------------------------------
# Neural network that outputs (N+1) coefficients a₀…a_N
# -----------------------------------------------------------------------------


class CoeffNet(nn.Module):
    def __init__(self, n_coeff: int):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(1, 64), nn.Tanh(), nn.Linear(64, n_coeff))

    def forward(
        self, dummy_input: torch.Tensor
    ) -> torch.Tensor:  # (B, 1) -> (B, n_coeff)
        return self.net(dummy_input)


net = CoeffNet(N + 1).to(device)

# -----------------------------------------------------------------------------
# Helper: evaluate power-series & its derivatives given coefficients
# -----------------------------------------------------------------------------

a_range = torch.arange(0, N + 1, device=device, dtype=dtype)


def _poly_eval(xs: torch.Tensor, coeffs: torch.Tensor, shift: int = 0) -> torch.Tensor:
    """Evaluate Σ_{n=shift}^N a_n x^{n-shift} / (n-shift)! for a batch of x."""
    if shift > N:
        return torch.zeros_like(xs)
    powers = xs.unsqueeze(1) ** (a_range[: N + 1 - shift])  # (B, N+1-shift)
    coeff_slice = coeffs[shift:]
    fact_slice = fact[: N + 1 - shift]
    return (powers * (coeff_slice / fact_slice)).sum(dim=1)  # (B,)


def u(xs: torch.Tensor, coeffs: torch.Tensor) -> torch.Tensor:
    return _poly_eval(xs, coeffs, shift=0)


def du(xs: torch.Tensor, coeffs: torch.Tensor) -> torch.Tensor:
    return _poly_eval(xs, coeffs, shift=1)


def d3u(xs: torch.Tensor, coeffs: torch.Tensor) -> torch.Tensor:
    return _poly_eval(xs, coeffs, shift=3)


# -----------------------------------------------------------------------------
# Collocation points & loss function
# -----------------------------------------------------------------------------

xs_collocation = torch.linspace(
    x_left, x_right, num_collocation, device=device, dtype=dtype
)


def loss_fn() -> torch.Tensor:
    coeffs = net(torch.zeros((1, 1), device=device, dtype=dtype)).squeeze()  # (N+1,)

    # PDE residual (third derivative equality)
    pde_res = d3u(xs_collocation, coeffs) - torch.cos(pi_const * xs_collocation)
    loss_pde = (pde_res**2).mean()

    # Boundary conditions
    bc_terms = [
        (u(torch.tensor([x_left], device=device, dtype=dtype), coeffs) - 0.0) ** 2,
        (
            u(torch.tensor([x_right], device=device, dtype=dtype), coeffs)
            - math.cos(pi_const)
        )
        ** 2,
        (du(torch.tensor([x_right], device=device, dtype=dtype), coeffs) - 1.0) ** 2,
    ]
    loss_bc = torch.sum(torch.stack(bc_terms))

    # (Optional) supervised loss on first few coefficients
    loss_supervised = ((coeffs[:num_supervised] - training_data) ** 2).mean()

    return loss_pde + bc_weight * loss_bc + supervised_weight * loss_supervised


# -----------------------------------------------------------------------------
# Training Stage 1: Adam
# -----------------------------------------------------------------------------

max_iters_adam = 1000
optimizer = optim.Adam(net.parameters(), lr=1e-3)

print("\nStage 1: Adam optimisation\n--------------------------")
pbar = trange(max_iters_adam, desc="Adam")
for i in pbar:
    optimizer.zero_grad(set_to_none=True)
    loss = loss_fn()
    loss.backward()
    optimizer.step()
    pbar.set_postfix(loss=loss.item())

# -----------------------------------------------------------------------------
# Training Stage 2: LBFGS fine-tuning
# -----------------------------------------------------------------------------

print("\nStage 2: LBFGS fine-tuning\n--------------------------")

lbfgs_iters = 500
optimizer_lbfgs = optim.LBFGS(
    net.parameters(),
    lr=1.0,
    max_iter=20,
    tolerance_grad=1e-9,
    tolerance_change=1e-9,
    history_size=100,
    line_search_fn="strong_wolfe",
)

pbar = trange(lbfgs_iters, desc="LBFGS")
for i in pbar:

    def closure():
        optimizer_lbfgs.zero_grad(set_to_none=True)
        l = loss_fn()
        l.backward()
        return l

    loss_val = optimizer_lbfgs.step(closure)
    pbar.set_postfix(loss=loss_val.item())

# -----------------------------------------------------------------------------
# Evaluation & visualisation
# -----------------------------------------------------------------------------

coeff_learned = (
    net(torch.zeros((1, 1), device=device, dtype=dtype)).squeeze().detach().cpu()
)
a_learned = coeff_learned.numpy()

print("\nTraining complete. Learned coefficients:")
print(a_learned)

# Ensure output directory exists
out_dir = Path("data")
out_dir.mkdir(exist_ok=True)

# --- Plot 1: Learned solution vs analytic solution ---------------------------

x_plot = np.linspace(x_left, x_right, 101)

powers_np = x_plot[:, None] ** np.arange(0, N + 1)[None, :]
series_coef = a_learned / fact.cpu().numpy()
u_pred = powers_np @ series_coef

u_true = analytic_solution_np(x_plot)

plt.figure(figsize=(6, 4))
plt.plot(x_plot, u_pred, "b-", lw=2, label="PINN power-series")
plt.plot(x_plot, u_true, "k--", lw=2, label="Analytic")
plt.title("ODE solution comparison")
plt.xlabel("x")
plt.ylabel("u(x)")
plt.legend()
plt.tight_layout()
plt.savefig(out_dir / "solution_comparison.png", dpi=200)
plt.close()

# --- Plot 2: Absolute error ---------------------------------------------------

error = np.abs(u_true - u_pred)
error = np.maximum(error, 1e-20)  # avoid log(0)

plt.figure(figsize=(6, 4))
plt.semilogy(x_plot, error, lw=2)
plt.title("Absolute error of power-series solution")
plt.xlabel("x")
plt.ylabel("Error (log scale)")
plt.tight_layout()
plt.savefig(out_dir / "error.png", dpi=200)
plt.close()

# --- Plot 3: Coefficient error ----------------------------------------------

coeff_error = np.abs(a_true.cpu().numpy() - a_learned)
coeff_error = np.maximum(coeff_error, 1e-20)

plt.figure(figsize=(6, 4))
plt.semilogy(np.arange(N + 1), coeff_error, "o-")
plt.title("Error in learned coefficients")
plt.xlabel("Coefficient index")
plt.ylabel("Absolute error (log scale)")
plt.tight_layout()
plt.savefig(out_dir / "coefficient_error.png", dpi=200)
plt.close()

print("\nPlots saved to 'data' directory:")
print("- solution_comparison.png")
print("- error.png")
print("- coefficient_error.png")
