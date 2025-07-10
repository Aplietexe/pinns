import sympy as sp
import torch
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import math

from pinn_trainer import train_power_series_pinn, solve_ivp, factorial_tensor

DTYPE = torch.float64
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def train_simple_ivp_pinn():
    """Train IVP-PINN on simple second-order constant-coefficient ODEs."""

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

    print("Training IVP-PINN for y'' + c1*y' + c0*y = 0")
    print("=" * 60)

    # Train on a focused range of stable coefficients
    net = train_power_series_pinn(
        c_list,
        f_expr,
        None,  # No specific BCs needed for training
        N=N,
        recurrence_weight=75.0,  # Strong recurrence regularization
        bc_weight=100.0,
        num_batches=3000,
        batch_size=128,
        num_collocation=1000,
        dtype=DTYPE,
        device=DEVICE,
        x_left=-1.0,
        x_right=1.0,
        c_range=(-1.5, 1.5),  # Stable coefficient range
        bc_range=(-0.5, 1.5),
        freeze_seeds=False,
    )

    return net, N


def test_on_known_odes(net, N):
    """Test the trained network on ODEs with known solutions."""

    fact = factorial_tensor(N, dtype=DTYPE, device=torch.device("cpu"))

    # Test cases: (c0, c1, y0, y'0, name, true_solution)
    test_cases = [
        # y'' + y = 0, y(0)=1, y'(0)=0 → cos(x)
        (1.0, 0.0, 1.0, 0.0, "cos(x)", lambda x: np.cos(x)),
        # y'' + y = 0, y(0)=0, y'(0)=1 → sin(x)
        (1.0, 0.0, 0.0, 1.0, "sin(x)", lambda x: np.sin(x)),
        # y'' - y = 0, y(0)=1, y'(0)=0 → cosh(x)
        (-1.0, 0.0, 1.0, 0.0, "cosh(x)", lambda x: np.cosh(x)),
        # y'' - y = 0, y(0)=0, y'(0)=1 → sinh(x)
        (-1.0, 0.0, 0.0, 1.0, "sinh(x)", lambda x: np.sinh(x)),
        # y'' + 2y' + y = 0, y(0)=1, y'(0)=0 → e^(-x)
        (1.0, 2.0, 1.0, 0.0, "e^(-x)", lambda x: np.exp(-x)),
    ]

    # Create visualization
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    axes = axes.flatten()

    x_plot = np.linspace(-1, 1, 200)

    for idx, (c0, c1, y0, yp0, name, true_fn) in enumerate(test_cases):
        ax = axes[idx]

        # Prepare input
        c_tensor = torch.tensor([c0, c1, 1.0], dtype=DTYPE)
        bc_tensor = torch.tensor([y0, yp0], dtype=DTYPE)

        # Get prediction
        a_pred = solve_ivp(c_tensor, bc_tensor, net, fact, freeze_seeds=True)

        # Evaluate power series
        powers = x_plot[:, None] ** np.arange(0, N + 1)[None, :]
        y_pred = powers @ a_pred.numpy()

        # True solution
        y_true = true_fn(x_plot)

        # Plot
        ax.plot(x_plot, y_pred, "b-", lw=2.5, label="IVP-PINN")
        ax.plot(x_plot, y_true, "r--", lw=1.5, label="True")
        ax.set_title(f"{name}: y″ + {c1}y′ + {c0}y = 0\ny(0)={y0}, y′(0)={yp0}")
        ax.set_xlabel("x")
        ax.set_ylabel("y(x)")
        ax.legend()
        ax.grid(True, alpha=0.3)

        # Compute and display error
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

    # Print coefficient comparison for cos(x)
    print("\n\nDetailed analysis for cos(x):")
    print("=" * 60)

    c_tensor = torch.tensor([1.0, 0.0, 1.0], dtype=DTYPE)
    bc_tensor = torch.tensor([1.0, 0.0], dtype=DTYPE)
    a_pred = solve_ivp(c_tensor, bc_tensor, net, fact, freeze_seeds=True)

    # True Taylor coefficients of cos(x)
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


def demonstrate_generalization(net, N):
    """Show how the network generalizes to different parameter values."""

    fact = factorial_tensor(N, dtype=DTYPE, device=torch.device("cpu"))

    # Vary damping coefficient
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    x_plot = np.linspace(-1, 1, 200)

    # Left plot: varying damping with fixed frequency
    damping_values = [0.0, 0.5, 1.0, 2.0, 3.0]
    for zeta in damping_values:
        c_tensor = torch.tensor([4.0, 2 * zeta, 1.0], dtype=DTYPE)  # ω=2
        bc_tensor = torch.tensor([1.0, 0.0], dtype=DTYPE)

        a_pred = solve_ivp(c_tensor, bc_tensor, net, fact, freeze_seeds=True)
        powers = x_plot[:, None] ** np.arange(0, N + 1)[None, :]
        y_pred = powers @ a_pred.numpy()

        ax1.plot(x_plot, y_pred, label=f"ζ={zeta}")

    ax1.set_title("Damped oscillator: y″ + 2ζy′ + 4y = 0")
    ax1.set_xlabel("x")
    ax1.set_ylabel("y(x)")
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # Right plot: varying frequency with no damping
    freq_values = [0.5, 1.0, 2.0, 3.0, 4.0]
    for omega in freq_values:
        c_tensor = torch.tensor([omega**2, 0.0, 1.0], dtype=DTYPE)
        bc_tensor = torch.tensor([1.0, 0.0], dtype=DTYPE)

        a_pred = solve_ivp(c_tensor, bc_tensor, net, fact, freeze_seeds=True)
        powers = x_plot[:, None] ** np.arange(0, N + 1)[None, :]
        y_pred = powers @ a_pred.numpy()

        ax2.plot(x_plot, y_pred, label=f"ω={omega}")

    ax2.set_title("Harmonic oscillator: y″ + ω²y = 0")
    ax2.set_xlabel("x")
    ax2.set_ylabel("y(x)")
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig("data/ivp_pinn_generalization.png", dpi=200)
    plt.close()

    print("\nGeneralization plot saved to data/ivp_pinn_generalization.png")


if __name__ == "__main__":
    # Ensure output directory exists
    Path("data").mkdir(exist_ok=True)

    # Train the network
    net, N = train_simple_ivp_pinn()

    # Test on known ODEs
    print("\n\nTesting on known ODEs...")
    test_on_known_odes(net, N)

    # Demonstrate generalization
    print("\n\nDemonstrating generalization...")
    demonstrate_generalization(net, N)

    print("\n\nDemo complete!")
