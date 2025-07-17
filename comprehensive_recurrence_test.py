"""
Comprehensive test of ALL architectures on ALL 6 equations' recurrence relations
"""

import torch
import torch.nn as nn
import torch.optim as optim
import time
import sympy as sp
from single_ode import make_recurrence
from corrected_ode_definitions import get_corrected_ode_definitions

DTYPE = torch.float32
DEVICE = torch.device("cpu")


class SimpleRNN(nn.Module):
    """Simple RNN (GRU) architecture"""

    def __init__(self, hidden_size=128):
        super().__init__()
        self.hidden_size = hidden_size
        self.rnn = nn.GRU(1, hidden_size, batch_first=True)
        self.output = nn.Linear(hidden_size, 1)

    def forward(self, x):
        if x.numel() == 0:
            h = torch.zeros(1, self.hidden_size, device=x.device, dtype=x.dtype)
        else:
            x_shaped = x.unsqueeze(0).unsqueeze(-1)
            _, h = self.rnn(x_shaped)
            h = h.squeeze(0)
        return self.output(h).squeeze()


class SimpleLSTM(nn.Module):
    """Simple LSTM architecture"""

    def __init__(self, hidden_size=128):
        super().__init__()
        self.hidden_size = hidden_size
        self.lstm = nn.LSTM(1, hidden_size, batch_first=True)
        self.output = nn.Linear(hidden_size, 1)

    def forward(self, x):
        if x.numel() == 0:
            h = torch.zeros(1, self.hidden_size, device=x.device, dtype=x.dtype)
        else:
            x_shaped = x.unsqueeze(0).unsqueeze(-1)
            _, (h, _) = self.lstm(x_shaped)
            h = h.squeeze(0)
        return self.output(h).squeeze()


class SimpleTransformer(nn.Module):
    """Simple Transformer architecture"""

    def __init__(self, d_model=128, nhead=8, num_layers=4):
        super().__init__()
        self.d_model = d_model
        self.input_proj = nn.Linear(1, d_model)
        self.pos_encoding = nn.Parameter(torch.randn(50, d_model))  # max length 50

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=nhead, dim_feedforward=d_model * 2, batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.output = nn.Linear(d_model, 1)

    def forward(self, x):
        if x.numel() == 0:
            # Empty sequence - use a learned embedding
            h = torch.zeros(1, self.d_model, device=x.device, dtype=x.dtype)
        else:
            seq_len = len(x)
            x_shaped = x.unsqueeze(0).unsqueeze(-1)  # [1, seq_len, 1]

            # Project to d_model
            x_proj = self.input_proj(x_shaped)  # [1, seq_len, d_model]

            # Add positional encoding
            x_proj = x_proj + self.pos_encoding[:seq_len].unsqueeze(0)

            # Apply transformer
            output = self.transformer(x_proj)  # [1, seq_len, d_model]

            # Use last token
            h = output[0, -1, :]  # [d_model]

        return self.output(h).squeeze()


class SimpleMLP(nn.Module):
    """Simple MLP with fixed input size"""

    def __init__(self, input_size=10, hidden_size=128):
        super().__init__()
        self.input_size = input_size
        self.net = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, 1),
        )

    def forward(self, x):
        # Pad or truncate to fixed size
        if x.numel() == 0:
            processed = torch.zeros(self.input_size, device=x.device, dtype=x.dtype)
        else:
            if len(x) > self.input_size:
                processed = x[-self.input_size :]
            else:
                processed = torch.cat(
                    [
                        torch.zeros(
                            self.input_size - len(x), device=x.device, dtype=x.dtype
                        ),
                        x,
                    ]
                )
        return self.net(processed).squeeze()


def create_training_dataset(recurrence_fn, num_samples=500, max_len=15):
    """Create training dataset for recurrence learning"""
    dataset = []

    for _ in range(num_samples):
        # Random prefix length
        prefix_len = torch.randint(2, max_len, (1,)).item()

        # Random coefficients with reasonable magnitudes
        prefix = torch.randn(prefix_len, dtype=DTYPE, device=DEVICE) * 0.1

        try:
            target = recurrence_fn(prefix)
            if torch.isfinite(target):
                dataset.append((prefix.clone(), target.clone()))
        except:
            continue

    return dataset


def estimate_training_time(model, dataset, timeout_seconds=30):
    """Estimate full training time using a timeout"""
    if not dataset:
        return float("inf"), 0

    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    criterion = nn.MSELoss()

    start_time = time.time()
    epoch = 0

    model.train()

    # Run for timeout_seconds to estimate speed
    while time.time() - start_time < timeout_seconds:
        total_loss = 0.0

        for prefix, target in dataset:
            optimizer.zero_grad()
            pred = model(prefix)
            loss = criterion(pred, target)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        epoch += 1
        avg_loss = total_loss / len(dataset)

        # Early stopping if converged
        if avg_loss < 1e-8:
            break

    elapsed = time.time() - start_time
    epochs_per_second = epoch / elapsed

    # Estimate time for 1000 epochs
    estimated_full_time = 1000 / epochs_per_second

    return estimated_full_time, avg_loss


def train_model(model, dataset, max_epochs=1000, time_limit=1800):  # 30 min limit
    """Train model with time limit"""
    if not dataset:
        return float("inf"), 0

    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    criterion = nn.MSELoss()

    start_time = time.time()
    model.train()

    for epoch in range(max_epochs):
        total_loss = 0.0

        for prefix, target in dataset:
            optimizer.zero_grad()
            pred = model(prefix)
            loss = criterion(pred, target)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        avg_loss = total_loss / len(dataset)
        print(f"Epoch {epoch} loss: {avg_loss:.2e}")

        # Early stopping conditions
        if avg_loss < 1e-8:
            break

        # Time limit
        if time.time() - start_time > time_limit:
            break

    return avg_loss, epoch + 1


def test_architecture_on_equation(arch_name, model_class, ode_name, ode_info):
    """Test a specific architecture on a specific equation"""
    print(f"\n  Testing {arch_name} on {ode_name}...")

    # Create recurrence function
    try:
        recurrence_fn = make_recurrence(
            ode_info["c_list"], ode_info["f_expr"], dtype=DTYPE, device=DEVICE, max_n=20
        )
    except Exception as e:
        return {
            "status": "failed",
            "error": f"Failed to create recurrence: {e}",
            "loss": float("inf"),
            "epochs": 0,
            "time": 0,
        }

    # Create dataset
    dataset = create_training_dataset(recurrence_fn, num_samples=300)

    if not dataset:
        return {
            "status": "failed",
            "error": "No training data created",
            "loss": float("inf"),
            "epochs": 0,
            "time": 0,
        }

    # Create model
    model = model_class().to(device=DEVICE, dtype=DTYPE)

    # Estimate training time
    estimated_time, initial_loss = estimate_training_time(
        model, dataset[:50]
    )  # Use subset for estimate

    print(f"    Dataset size: {len(dataset)}")
    print(f"    Estimated full training time: {estimated_time:.1f} seconds")
    print(f"    Initial loss: {initial_loss:.2e}")

    # Decide on training strategy
    if estimated_time > 1800:  # 30 minutes
        print(f"    Training would take too long, using reduced parameters")
        # Create smaller model
        if arch_name == "Transformer":
            model = SimpleTransformer(d_model=64, nhead=4, num_layers=2).to(
                device=DEVICE, dtype=DTYPE
            )
        elif "RNN" in arch_name or "LSTM" in arch_name:
            model = model_class(hidden_size=64).to(device=DEVICE, dtype=DTYPE)
        elif "MLP" in arch_name:
            model = SimpleMLP(input_size=8, hidden_size=64).to(
                device=DEVICE, dtype=DTYPE
            )
        max_epochs = 500
    else:
        max_epochs = 1000

    # Train
    start_time = time.time()
    final_loss, epochs_used = train_model(model, dataset, max_epochs=max_epochs)
    training_time = time.time() - start_time

    # Test on validation data
    val_dataset = create_training_dataset(recurrence_fn, num_samples=50)
    val_error = 0.0

    if val_dataset:
        model.eval()
        with torch.no_grad():
            errors = []
            for prefix, target in val_dataset:
                try:
                    pred = model(prefix)
                    error = abs(pred - target).item()
                    errors.append(error)
                except:
                    continue

            if errors:
                val_error = sum(errors) / len(errors)

    result = {
        "status": "completed",
        "loss": final_loss,
        "val_error": val_error,
        "epochs": epochs_used,
        "time": training_time,
        "dataset_size": len(dataset),
    }

    print(f"    Final loss: {final_loss:.2e}")
    print(f"    Val error: {val_error:.2e}")
    print(f"    Epochs: {epochs_used}")
    print(f"    Time: {training_time:.1f}s")

    return result


def main():
    print("COMPREHENSIVE RECURRENCE LEARNING TEST")
    print("=" * 50)
    print("Testing ALL architectures on ALL 6 equations")
    print("Using 30-second timeout to estimate training time")
    print("Scaling down if full training would take >30 minutes")
    print()

    # Get all ODEs
    odes = get_corrected_ode_definitions()

    # Define architectures to test
    architectures = [
        ("Transformer", SimpleTransformer),
        ("RNN-GRU", SimpleRNN),
        ("RNN-LSTM", SimpleLSTM),
        ("MLP", SimpleMLP),
    ]

    # Results matrix
    results = {}

    # Test each architecture on each equation
    for arch_name, model_class in architectures:
        print(f"\n{arch_name} Architecture:")
        print("-" * 40)

        results[arch_name] = {}

        for ode_name, ode_info in odes.items():
            result = test_architecture_on_equation(
                arch_name, model_class, ode_name, ode_info
            )
            results[arch_name][ode_name] = result

    # Print summary
    print("\n" + "=" * 80)
    print("COMPREHENSIVE RESULTS SUMMARY")
    print("=" * 80)

    print(
        f"{'Architecture':<15} {'Equation':<12} {'Status':<10} {'Loss':<12} {'Val Error':<12} {'Epochs':<8} {'Time(s)':<8}"
    )
    print("-" * 80)

    for arch_name in results:
        for ode_name in results[arch_name]:
            result = results[arch_name][ode_name]
            status = result["status"]
            loss = result["loss"] if result["loss"] != float("inf") else "INF"
            val_error = result.get("val_error", 0)
            epochs = result["epochs"]
            time_taken = result["time"]

            print(
                f"{arch_name:<15} {ode_name:<12} {status:<10} {loss:<12} {val_error:<12.2e} {epochs:<8} {time_taken:<8.1f}"
            )

    # Analysis
    print("\n" + "=" * 80)
    print("ANALYSIS")
    print("=" * 80)

    # Count successes (loss < 1e-6)
    success_threshold = 1e-6

    for arch_name in results:
        successes = 0
        total = len(results[arch_name])

        for ode_name in results[arch_name]:
            if results[arch_name][ode_name]["loss"] < success_threshold:
                successes += 1

        print(
            f"{arch_name}: {successes}/{total} equations with loss < {success_threshold}"
        )

    # Best architecture for each equation
    print("\nBest architecture for each equation:")
    for ode_name in odes:
        best_arch = None
        best_loss = float("inf")

        for arch_name in results:
            if results[arch_name][ode_name]["loss"] < best_loss:
                best_loss = results[arch_name][ode_name]["loss"]
                best_arch = arch_name

        print(f"  {ode_name}: {best_arch} (loss: {best_loss:.2e})")

    print("\n" + "=" * 80)
    print("CONCLUSION")
    print("=" * 80)
    print("This systematic test reveals which architectures can learn which")
    print("recurrence relations. Architectures that succeed here should work")
    print("in the full PINN pipeline - if they don't, the issue is integration.")


if __name__ == "__main__":
    main()
