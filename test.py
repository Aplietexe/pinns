import sympy as sp
import torch
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

# ========= DTYPE CONFIGURATION =========
# Set this to either torch.float32 or torch.float64
DTYPE = torch.float64
# =======================================

from pinn_trainer import train_power_series_pinn, solve_ivp, factorial_tensor
from pinn_evaluate import solve_and_plot

# -------------------------------------------------------------------------
# Example: Airy equation  y''(x) - x*y(x) = 0
# with initial conditions:
#     y(0)  = Ai(0) ≈ 0.35502805388781723926
#     y'(0) = Ai'(0) ≈ -0.25881940379280679840

# ODE definition  Σ c_k(x) y^{(k)}(x) = f(x)  with m = 2
x = sp.symbols("x")
c_list: list[sp.Expr] = [
    -x,  # coefficient of y
    sp.Integer(0),  # coefficient of y'
    sp.Integer(1),  # coefficient of y''
]
f_expr = sp.Integer(0)

# Boundary conditions for Airy function Ai(x)
Ai0 = 0.35502805388781723926
Aip0 = -0.25881940379280679840

bcs = [
    (0.0, 0, Ai0),  # y(0) = Ai(0)
    (0.0, 1, Aip0),  # y'(0) = Ai'(0)
]

# -------------------------------------------------------------------------
# Train the IVP-PINN network
print("Training IVP-PINN for constant-coefficient ODEs...")
print("=" * 60)

m = len(c_list) - 1  # highest derivative order
N = 7  # truncation order

# Train the network on random constant-coefficient ODEs
net = train_power_series_pinn(
    c_list,
    f_expr,
    bcs,
    N=N,
    recurrence_weight=1.0,
    bc_weight=100.0,
    num_batches=5000,
    batch_size=128,
    num_collocation=1000,
    dtype=DTYPE,
    x_left=-2.0,
    x_right=2.0,
    c_range=(-2.0, 2.0),
    bc_range=(-2.0, 2.0),
    freeze_seeds=True,
)

# -------------------------------------------------------------------------
# Inference: solve the specific Airy IVP
print("\n\nInference: Solving Airy equation with trained network...")
print("=" * 60)

# For the Airy equation: c = [-x, 0, 1] evaluated at x=0 gives [0, 0, 1]
c_airy = torch.tensor([0.0, 0.0, 1.0], dtype=DTYPE)
bc_airy = torch.tensor([Ai0, Aip0], dtype=DTYPE)

# Get factorials
fact = factorial_tensor(N, dtype=DTYPE, device=torch.device("cpu"))

# Solve using the trained network
coeffs_learned = solve_ivp(c_airy, bc_airy, net, fact, freeze_seeds=True)

print(f"\nLearned coefficients (first 10):")
print(coeffs_learned[:10])


# -------------------------------------------------------------------------
# Evaluation and visualization
def evaluate_ivp_network(
    net,
    c_list,
    f_expr,
    test_cases,
    N,
    *,
    x_range=(-2, 2),
    num_points=200,
    dtype=torch.float64,
    out_dir="data",
    file_prefix="ivp_",
):
    """Evaluate the IVP network on multiple test cases and create visualizations."""

    out_dir = Path(out_dir)
    out_dir.mkdir(exist_ok=True)

    m = len(c_list) - 1
    fact = factorial_tensor(N, dtype=dtype, device=torch.device("cpu"))

    x_vals = np.linspace(x_range[0], x_range[1], num_points)

    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    axes = axes.flatten()

    for idx, (c_vals, bc_vals, label, true_sol) in enumerate(test_cases[:4]):
        c_tensor = torch.tensor(c_vals, dtype=dtype)
        bc_tensor = torch.tensor(bc_vals, dtype=dtype)

        # Get predicted coefficients
        a_pred = solve_ivp(c_tensor, bc_tensor, net, fact, freeze_seeds=True)

        # Evaluate power series
        powers = x_vals[:, None] ** np.arange(0, N + 1)[None, :]
        y_pred = powers @ a_pred.numpy()

        # Plot
        ax = axes[idx]
        ax.plot(x_vals, y_pred, "b-", lw=2, label="PINN prediction")
        if true_sol is not None:
            y_true = true_sol(x_vals)
            ax.plot(x_vals, y_true, "r--", lw=1.5, label="True solution")

        ax.set_title(f"{label}\nc={c_vals}, bc={bc_vals}")
        ax.set_xlabel("x")
        ax.set_ylabel("y(x)")
        ax.legend()
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(out_dir / f"{file_prefix}test_cases.png", dpi=200)
    plt.close()

    print(f"\nTest cases plot saved to {out_dir / f'{file_prefix}test_cases.png'}")


# Create test cases
def airy_ai(x):
    """Approximate Airy Ai function using its series expansion."""
    # This is a truncated series - for demo purposes only
    c0 = 0.35502805388781723926
    c1 = -0.25881940379280679840
    return c0 + c1 * x - c0 * x**3 / 6 - c1 * x**4 / 12 + c0 * x**6 / 180


test_cases = [
    # (c_values, bc_values, label, true_solution)
    ([0.0, 0.0, 1.0], [Ai0, Aip0], "Airy equation", airy_ai),
    ([1.0, 0.0, 1.0], [1.0, 0.0], "y'' + y = 0, y(0)=1, y'(0)=0", lambda x: np.cos(x)),
    ([1.0, 0.0, 1.0], [0.0, 1.0], "y'' + y = 0, y(0)=0, y'(0)=1", lambda x: np.sin(x)),
    (
        [-1.0, 0.0, 1.0],
        [1.0, 0.0],
        "y'' - y = 0, y(0)=1, y'(0)=0",
        lambda x: np.cosh(x),
    ),
]

# Evaluate the network
evaluate_ivp_network(
    net,
    c_list,
    f_expr,
    test_cases,
    N,
    x_range=(-2, 2),
    dtype=DTYPE,
    file_prefix="airy_ivp_",
)

# -------------------------------------------------------------------------
# Also create standard plots for the Airy equation specifically
print("\nCreating detailed plots for Airy equation...")
solve_and_plot(
    c_list,
    f_expr,
    bcs,
    coeffs_learned,
    file_prefix="airy_",
    dtype=DTYPE,
    x_left=-2.0,
    x_right=0,
    analytic_expr=None,  # Let SymPy solve it
    vectorize=True,  # Needed for special functions
)

# -------------------------------------------------------------------------
# Demonstrate network's ability to generalize
print("\n\nDemonstrating generalization to different coefficient values...")
print("=" * 60)

# Create a grid of different c0 values (coefficient of y)
c0_values = np.linspace(-2, 2, 5)
plt.figure(figsize=(10, 6))

for c0 in c0_values:
    c_test = torch.tensor([c0, 0.0, 1.0], dtype=DTYPE)
    bc_test = torch.tensor([1.0, 0.0], dtype=DTYPE)  # y(0)=1, y'(0)=0

    a_test = solve_ivp(c_test, bc_test, net, fact, freeze_seeds=True)

    x_plot = np.linspace(-1, 1, 100)
    powers = x_plot[:, None] ** np.arange(0, N + 1)[None, :]
    y_plot = powers @ a_test.numpy()

    plt.plot(x_plot, y_plot, label=f"c₀={c0:.1f}")

plt.xlabel("x")
plt.ylabel("y(x)")
plt.title("Solutions for y″ + c₀·y = 0 with y(0)=1, y′(0)=0")
plt.legend()
plt.grid(True, alpha=0.3)
plt.savefig("data/generalization_demo.png", dpi=200)
plt.close()

print("Generalization demo saved to data/generalization_demo.png")

# -------------------------------------------------------------------------
# Performance summary
print("\n\nPerformance Summary")
print("=" * 60)

# Compare learned coefficients with true Airy series coefficients
# True Airy Ai(x) series coefficients (first few terms)
true_airy_coeffs = np.array(
    [
        0.35502805388781723926,  # a0 = Ai(0)
        -0.25881940379280679840,  # a1 = Ai'(0)
        0.0,  # a2 = 0
        0.05917138191420125313,  # a3
        0.0,  # a4 = 0
        -0.00863802098643969553,  # a5
    ]
)

print("\nAiry equation coefficient comparison (first 6 terms):")
print("Index | True Value      | Learned Value   | Absolute Error")
print("-" * 60)
for i in range(min(6, len(coeffs_learned))):
    if i < len(true_airy_coeffs):
        true_val = true_airy_coeffs[i]
        learned_val = coeffs_learned[i].item()
        error = abs(true_val - learned_val)
        print(f"{i:5d} | {true_val:15.10f} | {learned_val:15.10f} | {error:.2e}")
    else:
        print(f"{i:5d} | N/A             | {coeffs_learned[i].item():15.10f} | N/A")

# Evaluate accuracy at specific points
test_points = np.array([-1.0, -0.5, 0.0, 0.5, 1.0])
powers_test = test_points[:, None] ** np.arange(0, N + 1)[None, :]
y_test = powers_test @ coeffs_learned.numpy()

# Approximate true values (using truncated series)
y_true_approx = airy_ai(test_points)

print("\n\nPoint-wise evaluation:")
print("x     | PINN Value      | Approx True     | Relative Error")
print("-" * 60)
for i, x in enumerate(test_points):
    pinn_val = y_test[i]
    true_val = y_true_approx[i]
    rel_error = (
        abs((pinn_val - true_val) / true_val) if true_val != 0 else abs(pinn_val)
    )
    print(f"{x:5.1f} | {pinn_val:15.10f} | {true_val:15.10f} | {rel_error:.2e}")

print("\n\nAll evaluations complete!")
