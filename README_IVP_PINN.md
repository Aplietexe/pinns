# IVP-PINN: Initial Value Problem Physics-Informed Neural Network

This extension to the PINN trainer allows a neural network to learn the mapping from ODE coefficients and initial conditions to Maclaurin series solutions.

## Overview

The IVP-PINN (Initial Value Problem PINN) extends the original power-series PINN approach to handle families of constant-coefficient ODEs. Instead of solving a single ODE, the network learns to solve any ODE of the form:

```
Σ c_k y^(k)(x) = f(x)
```

with initial conditions `y^(k)(0) = b_k` for `k = 0, ..., m-1`.

## Architecture

### Network Structure

The `IVPCoeffNet` is a feedforward neural network that:

- **Input**: Concatenated ODE coefficients and initial values `[c_0, ..., c_m, y(0), y'(0), ..., y^(m-1)(0)]`
- **Hidden layers**: Two layers with 128 neurons each and GELU activation
- **Output**: N+1 Maclaurin series coefficients

### Key Features

1. **Seed Freezing**: When `freeze_seeds=True`, the first m coefficients are analytically computed from initial conditions:

   ```
   a_k = y^(k)(0) / k!
   ```

   This ensures exact satisfaction of initial conditions.

2. **Batched Training**: The network trains on randomly sampled constant-coefficient ODEs within specified ranges.

3. **Recurrence Regularization**: Enforces the recurrence relation between series coefficients for improved accuracy.

## Usage

### Training

```python
from pinn_trainer import train_power_series_pinn

# Define ODE template (using SymPy symbols for coefficients)
x = sp.symbols("x")
c_list = [
    sp.Symbol('c0'),  # coefficient of y
    sp.Symbol('c1'),  # coefficient of y'
    sp.Integer(1),    # coefficient of y''
]
f_expr = sp.Integer(0)  # homogeneous

# Train the network
net = train_power_series_pinn(
    c_list,
    f_expr,
    None,  # No specific BCs for training
    N=15,  # Truncation order
    num_batches=3000,
    batch_size=64,
    c_range=(-2.0, 2.0),    # Range for ODE coefficients
    bc_range=(-2.0, 2.0),   # Range for initial conditions
    freeze_seeds=True,      # Enforce initial conditions exactly
    dtype=torch.float64,
)
```

### Inference

```python
from pinn_trainer import solve_ivp, factorial_tensor

# Solve specific IVP: y'' + y = 0, y(0)=1, y'(0)=0
c = torch.tensor([1.0, 0.0, 1.0], dtype=torch.float64)  # [c0, c1, c2]
bc = torch.tensor([1.0, 0.0], dtype=torch.float64)      # [y(0), y'(0)]

fact = factorial_tensor(N, dtype=torch.float64, device=torch.device("cpu"))
coeffs = solve_ivp(c, bc, net, fact, freeze_seeds=True)

# Evaluate solution
x_vals = np.linspace(-1, 1, 100)
powers = x_vals[:, None] ** np.arange(0, N + 1)[None, :]
y_vals = powers @ coeffs.numpy()
```

## Examples

### 1. Simple Harmonic Oscillator

```python
# y'' + ω²y = 0, y(0)=1, y'(0)=0
omega = 2.0
c = torch.tensor([omega**2, 0.0, 1.0], dtype=torch.float64)
bc = torch.tensor([1.0, 0.0], dtype=torch.float64)
coeffs = solve_ivp(c, bc, net, fact)
# Solution approximates cos(ωx)
```

### 2. Damped Oscillator

```python
# y'' + 2ζy' + y = 0, y(0)=1, y'(0)=0
zeta = 0.5  # damping ratio
c = torch.tensor([1.0, 2*zeta, 1.0], dtype=torch.float64)
bc = torch.tensor([1.0, 0.0], dtype=torch.float64)
coeffs = solve_ivp(c, bc, net, fact)
```

### 3. Exponential Growth/Decay

```python
# y'' + 2y' + y = 0, y(0)=1, y'(0)=0
c = torch.tensor([1.0, 2.0, 1.0], dtype=torch.float64)
bc = torch.tensor([1.0, 0.0], dtype=torch.float64)
coeffs = solve_ivp(c, bc, net, fact)
# Solution approximates e^(-x)
```

## Performance Considerations

1. **Training Range**: Choose `c_range` and `bc_range` to cover the parameter space of interest while maintaining numerical stability.

2. **Truncation Order**: Higher N provides better accuracy but requires more training and may suffer from numerical issues.

3. **Regularization**: The `recurrence_weight` parameter is crucial for enforcing mathematical consistency.

4. **Precision**: Use `torch.float64` for better numerical accuracy, especially for higher-order terms.

## Limitations

1. Currently limited to constant-coefficient ODEs
2. Performance degrades for coefficients outside the training range
3. Higher-order derivatives (large m) may require careful tuning

## Demo Scripts

- `test.py`: Demonstrates IVP-PINN on the Airy equation
- `demo_ivp_pinn.py`: Comprehensive demo with multiple test cases and visualizations

## Visualizations

The demo scripts generate several plots:

- Solution comparisons between PINN predictions and true solutions
- Coefficient error analysis
- Generalization demonstrations showing network performance across parameter ranges
