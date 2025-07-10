"""
Demo of IVP-PINN training without batching + LBFGS fine-tuning.
"""

import sympy as sp
import torch
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import math

from pinn_trainer import train_power_series_pinn_no_batch, solve_ivp, factorial_tensor

# Configuration
DTYPE = torch.float64
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def main():
    # Define the ODE template: y'' + c1*y' + c0*y = 0
    x = sp.symbols("x")
    c_list = [
        sp.Symbol("c0"),  # coefficient of y
        sp.Symbol("c1"),  # coefficient of y'
        sp.Integer(1),  # coefficient of y''
    ]
    f_expr = sp.Integer(0)  # homogeneous

    # Training parameters
    N = 15  # truncation order
    m = len(c_list) - 1

    print("Training IVP-PINN WITHOUT batching (all samples at once)")
    print("=" * 60)

    # Train with no batching + LBFGS
    net = train_power_series_pinn_no_batch(
        c_list,
        f_expr,
        None,
        N=N,
        recurrence_weight=75.0,
        bc_weight=100.0,
        adam_iters=3000,
        lbfgs_iters=50,
        num_train_samples=500,  # All 500 ODEs evaluated at once
        num_collocation=200,  # Fewer points to save memory
        dtype=DTYPE,
        device=DEVICE,
        x_left=-1.0,
        x_right=1.0,
        c_range=(-1.5, 1.5),
        bc_range=(-0.5, 1.5),
        freeze_seeds=False,
    )

    # Test on known ODEs
    fact = factorial_tensor(N, dtype=DTYPE, device=torch.device("cpu"))

    print("\n\nTesting on known ODEs:")
    print("-" * 60)

    test_cases = [
        (1.0, 0.0, 1.0, 0.0, "cos(x)"),
        (1.0, 0.0, 0.0, 1.0, "sin(x)"),
        (4.0, 0.0, 1.0, 0.0, "cos(2x)"),
        (-1.0, 0.0, 1.0, 0.0, "cosh(x)"),
        (1.0, 2.0, 1.0, 0.0, "e^(-x)"),
    ]

    x_test = np.linspace(-1, 1, 100)

    for c0, c1, y0, yp0, name in test_cases:
        c_tensor = torch.tensor([c0, c1, 1.0], dtype=DTYPE)
        bc_tensor = torch.tensor([y0, yp0], dtype=DTYPE)

        a_pred = solve_ivp(c_tensor, bc_tensor, net, fact, freeze_seeds=True)

        # Evaluate at x=0.5
        x_val = 0.5
        powers = np.array([x_val**n for n in range(N + 1)])
        y_pred = np.dot(powers, a_pred.numpy())

        # True values
        if name == "cos(x)":
            y_true = np.cos(x_val)
        elif name == "sin(x)":
            y_true = np.sin(x_val)
        elif name == "cos(2x)":
            y_true = np.cos(2 * x_val)
        elif name == "cosh(x)":
            y_true = np.cosh(x_val)
        elif name == "e^(-x)":
            y_true = np.exp(-x_val)
        else:
            y_true = 0.0  # Should not happen

        error = abs(y_pred - y_true)
        print(
            f"{name:10s}: y(0.5) = {y_pred:10.6f} (true: {y_true:10.6f}, error: {error:.2e})"
        )

    # Create comparison plot
    print("\nCreating comparison plots...")

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # Left: cos(x) comparison
    c_tensor = torch.tensor([1.0, 0.0, 1.0], dtype=DTYPE)
    bc_tensor = torch.tensor([1.0, 0.0], dtype=DTYPE)
    a_pred = solve_ivp(c_tensor, bc_tensor, net, fact, freeze_seeds=True)

    powers = x_test[:, None] ** np.arange(0, N + 1)[None, :]
    y_pred = powers @ a_pred.numpy()

    axes[0].plot(x_test, y_pred, "b-", lw=2.5, label="IVP-PINN (no batch)")
    axes[0].plot(x_test, np.cos(x_test), "r--", lw=1.5, label="True cos(x)")
    axes[0].set_title("y″ + y = 0, y(0)=1, y′(0)=0")
    axes[0].set_xlabel("x")
    axes[0].set_ylabel("y(x)")
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    # Right: Coefficient comparison
    true_cos_coeffs = []
    for n in range(N + 1):
        if n % 4 == 0:
            true_cos_coeffs.append(1.0 / math.factorial(n))
        elif n % 4 == 2:
            true_cos_coeffs.append(-1.0 / math.factorial(n))
        else:
            true_cos_coeffs.append(0.0)

    coeff_errors = [abs(a_pred[n].item() - true_cos_coeffs[n]) for n in range(N + 1)]

    axes[1].semilogy(range(N + 1), coeff_errors, "o-")
    axes[1].set_title("Coefficient errors for cos(x)")
    axes[1].set_xlabel("Coefficient index n")
    axes[1].set_ylabel("|a_n^pred - a_n^true|")
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig("data/no_batch_comparison.png", dpi=200)
    plt.close()

    print("Plot saved to data/no_batch_comparison.png")
    print("\nDemo complete!")


if __name__ == "__main__":
    Path("data").mkdir(exist_ok=True)
    main()
