import sympy as sp
import torch
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import math

from pinn_trainer import train_power_series_pinn_no_batch, solve_ivp, factorial_tensor

DTYPE = torch.float64
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def train_simple_ivp_pinn():
    x = sp.symbols("x")
    c_list = [
        sp.Symbol("c0"),  # coefficient of y
        sp.Symbol("c1"),  # coefficient of y'
        sp.Integer(1),  # coefficient of y''
    ]
    f_expr = sp.Integer(0)

    N = 15
    m = len(c_list) - 1
    net = train_power_series_pinn_no_batch(
        c_list,
        f_expr,
        None,
        N=N,
        recurrence_weight=75.0,
        bc_weight=100.0,
        adam_iters=3000,
        lbfgs_iters=50,
        num_train_samples=500,
        num_collocation=200,
        dtype=DTYPE,
        device=DEVICE,
        x_left=-1.0,
        x_right=1.0,
        c_range=(-1.5, 1.5),
        bc_range=(-0.5, 1.5),
        freeze_seeds=False,
    )

    return net, N


def test_on_known_odes(net, N):
    fact = factorial_tensor(N, dtype=DTYPE, device=torch.device("cpu"))

    test_cases = [
        (1.0, 0.0, 1.0, 0.0, "cos(x)", lambda x: np.cos(x)),
        (1.0, 0.0, 0.0, 1.0, "sin(x)", lambda x: np.sin(x)),
        (-1.0, 0.0, 1.0, 0.0, "cosh(x)", lambda x: np.cosh(x)),
        (-1.0, 0.0, 0.0, 1.0, "sinh(x)", lambda x: np.sinh(x)),
        (1.0, 2.0, 1.0, 0.0, "e^(-x)", lambda x: np.exp(-x)),
    ]

    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    axes = axes.flatten()

    x_plot = np.linspace(-1, 1, 200)

    for idx, (c0, c1, y0, yp0, name, true_fn) in enumerate(test_cases):
        ax = axes[idx]

        c_tensor = torch.tensor([c0, c1, 1.0], dtype=DTYPE)
        bc_tensor = torch.tensor([y0, yp0], dtype=DTYPE)

        a_pred = solve_ivp(c_tensor, bc_tensor, net, fact, freeze_seeds=True)

        powers = x_plot[:, None] ** np.arange(0, N + 1)[None, :]
        y_pred = powers @ a_pred.numpy()

        y_true = true_fn(x_plot)

        ax.plot(x_plot, y_pred, "b-", lw=2.5, label="IVP-PINN")
        ax.plot(x_plot, y_true, "r--", lw=1.5, label="True")
        ax.set_title(f"{name}: y″ + {c1}y′ + {c0}y = 0\ny(0)={y0}, y′(0)={yp0}")
        ax.set_xlabel("x")
        ax.set_ylabel("y(x)")
        ax.legend()
        ax.grid(True, alpha=0.3)

        error = np.abs(y_pred - y_true)
        max_error = np.max(error)
        ax.text(
            0.95,
            0.05,
            f"Max error: {max_error:.2e}",
            transform=ax.transAxes,
            ha="right",
            va="bottom",
            bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.5),
        )

    plt.tight_layout()
    plt.savefig("data/ivp_pinn_demo.png", dpi=200)
    plt.close()

    print("\nDemo plot saved to data/ivp_pinn_demo.png")

    print("\n\nDetailed analysis for cos(x):")
    print("=" * 60)

    c_tensor = torch.tensor([1.0, 0.0, 1.0], dtype=DTYPE)
    bc_tensor = torch.tensor([1.0, 0.0], dtype=DTYPE)
    a_pred = solve_ivp(c_tensor, bc_tensor, net, fact, freeze_seeds=True)

    true_cos_coeffs = []
    for n in range(N + 1):
        if n % 4 == 0:
            true_cos_coeffs.append(1.0 / math.factorial(n))
        elif n % 4 == 2:
            true_cos_coeffs.append(-1.0 / math.factorial(n))
        else:
            true_cos_coeffs.append(0.0)

    print("\nTaylor coefficients comparison:")
    print("n  | True coefficient | Predicted coefficient | Error")
    print("-" * 60)
    for n in range(min(10, N + 1)):
        true_val = true_cos_coeffs[n]
        pred_val = a_pred[n].item()
        error = abs(true_val - pred_val)
        print(f"{n:2d} | {true_val:16.10f} | {pred_val:21.10f} | {error:.2e}")


if __name__ == "__main__":
    Path("data").mkdir(exist_ok=True)

    net, N = train_simple_ivp_pinn()

    test_on_known_odes(net, N)

    print("\n\nDemo complete!")
