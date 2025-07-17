"""
Scientific depth scaling investigation with parameter matching and optimization improvements
"""

import torch
import torch.nn as nn
import torch.optim as optim
import time
import numpy as np
import matplotlib.pyplot as plt
from single_ode import make_recurrence
from corrected_ode_definitions import get_corrected_ode_definitions

DTYPE = torch.float32
DEVICE = torch.device("cpu")


class ImprovedTransformer(nn.Module):
    """Improved Transformer with SiLU activation and configurable parameters"""

    def __init__(self, d_model=64, nhead=4, num_layers=2):
        super().__init__()
        self.d_model = d_model
        self.input_proj = nn.Linear(1, d_model)
        self.pos_encoding = nn.Parameter(torch.randn(50, d_model))

        # Custom transformer encoder layer with SiLU activation
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=d_model * 2,
            batch_first=True,
            activation=nn.SiLU(),  # Use SiLU instead of ReLU
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.output = nn.Linear(d_model, 1)

        # Initialize parameters properly
        self.apply(self._init_weights)

    def _init_weights(self, module):
        """Improved weight initialization"""
        if isinstance(module, nn.Linear):
            torch.nn.init.xavier_uniform_(module.weight)
            if module.bias is not None:
                torch.nn.init.constant_(module.bias, 0)
        elif isinstance(module, nn.Parameter):
            torch.nn.init.normal_(module, std=0.02)

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


def count_parameters(model):
    """Count trainable parameters"""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def create_training_dataset(recurrence_fn, num_samples=200):
    """Create training dataset"""
    dataset = []

    for _ in range(num_samples):
        prefix_len = torch.randint(2, 10, (1,)).item()
        prefix = torch.randn(prefix_len, dtype=DTYPE, device=DEVICE) * 0.1

        try:
            target = recurrence_fn(prefix)
            if torch.isfinite(target):
                dataset.append((prefix.clone(), target.clone()))
        except:
            continue

    return dataset


class TrainingAnalyzer:
    """Analyze training dynamics and gradients"""

    def __init__(self, model):
        self.model = model
        self.loss_history = []
        self.gradient_norms = []
        self.layer_gradients = []
        self.learning_rates = []
        self.validation_losses = []

    def log_training_step(self, loss, optimizer, val_loss=None):
        """Log training metrics"""
        self.loss_history.append(loss)

        if val_loss is not None:
            self.validation_losses.append(val_loss)

        # Calculate gradient norms
        total_grad_norm = 0.0
        layer_grads = {}

        for name, param in self.model.named_parameters():
            if param.grad is not None:
                grad_norm = param.grad.norm().item()
                total_grad_norm += grad_norm**2

                # Group by layer type
                if "transformer.layers" in name:
                    layer_idx = name.split(".")[2]
                    if layer_idx not in layer_grads:
                        layer_grads[layer_idx] = 0.0
                    layer_grads[layer_idx] += grad_norm**2

        self.gradient_norms.append(total_grad_norm**0.5)
        self.layer_gradients.append(layer_grads)

        # Log learning rate
        self.learning_rates.append(optimizer.param_groups[0]["lr"])

    def analyze_training_dynamics(self):
        """Analyze training patterns"""
        analysis = {}

        # Loss convergence
        analysis["final_loss"] = (
            self.loss_history[-1] if self.loss_history else float("inf")
        )
        analysis["min_loss"] = (
            min(self.loss_history) if self.loss_history else float("inf")
        )
        analysis["loss_plateau"] = len(self.loss_history) > 100 and np.std(
            self.loss_history[-50:]
        ) < 0.01 * np.mean(self.loss_history[-50:])

        # Gradient flow
        analysis["avg_grad_norm"] = (
            np.mean(self.gradient_norms) if self.gradient_norms else 0.0
        )
        analysis["grad_norm_std"] = (
            np.std(self.gradient_norms) if self.gradient_norms else 0.0
        )
        analysis["vanishing_gradients"] = analysis["avg_grad_norm"] < 1e-5
        analysis["exploding_gradients"] = analysis["avg_grad_norm"] > 100.0

        # Overfitting (if validation data available)
        if self.validation_losses:
            train_loss = np.mean(self.loss_history[-10:])
            val_loss = np.mean(self.validation_losses[-10:])
            analysis["overfitting"] = val_loss > train_loss * 1.5
            analysis["train_val_gap"] = val_loss - train_loss

        # Learning rate effectiveness
        analysis["lr_stable"] = len(set(self.learning_rates)) == 1

        return analysis


def train_with_analysis(
    model, dataset, val_dataset=None, max_epochs=2000, time_limit=1200
):  # 20 min
    """Train model with comprehensive analysis"""
    if not dataset:
        return float("inf"), 0, None

    # Use AdamW optimizer as required
    optimizer = optim.AdamW(model.parameters(), lr=1e-3, weight_decay=0.01)
    criterion = nn.MSELoss()

    # Learning rate scheduler
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=100
    )

    analyzer = TrainingAnalyzer(model)

    start_time = time.time()
    model.train()

    best_loss = float("inf")
    convergence_milestones = {}

    print(f"    Starting training: {len(dataset)} samples, max {max_epochs} epochs")
    print(f"    Using AdamW optimizer with weight_decay=0.01")
    print(f"    Using SiLU activation function")

    for epoch in range(max_epochs):
        total_loss = 0.0

        for prefix, target in dataset:
            optimizer.zero_grad()
            pred = model(prefix)
            loss = criterion(pred, target)
            loss.backward()

            # Gradient clipping to prevent exploding gradients
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

            optimizer.step()
            total_loss += loss.item()

        avg_loss = total_loss / len(dataset)

        # Validation loss
        val_loss = None
        if val_dataset:
            model.eval()
            with torch.no_grad():
                val_total = 0.0
                for prefix, target in val_dataset:
                    pred = model(prefix)
                    val_total += criterion(pred, target).item()
                val_loss = val_total / len(val_dataset)
            model.train()

        # Log training metrics
        analyzer.log_training_step(avg_loss, optimizer, val_loss)

        if avg_loss < best_loss:
            best_loss = avg_loss

        # Update learning rate
        scheduler.step(avg_loss)

        # Track convergence milestones
        milestones = [1e-3, 1e-4, 1e-5, 1e-6]
        for milestone in milestones:
            if avg_loss < milestone and milestone not in convergence_milestones:
                convergence_milestones[milestone] = epoch
                print(f"    *** MILESTONE: {milestone:.0e} at epoch {epoch} ***")

        # Progress reporting
        if epoch % 200 == 0:
            lr = optimizer.param_groups[0]["lr"]
            val_str = f", val: {val_loss:.2e}" if val_loss else ""
            print(f"    Epoch {epoch}: {avg_loss:.2e} (lr: {lr:.2e}{val_str})")

        # Early stopping if excellent
        if avg_loss < 1e-8:
            print(f"    Early stopping at epoch {epoch}: {avg_loss:.2e}")
            break

        # Time limit
        if time.time() - start_time > time_limit:
            print(f"    Time limit reached at epoch {epoch}")
            break

    training_time = time.time() - start_time
    print(f"    Training completed: {training_time:.1f}s, {epoch+1} epochs")

    return best_loss, epoch + 1, analyzer, convergence_milestones, training_time


def test_parameter_matched_depth(ode_name, ode_info):
    """Test parameter-matched depth scaling"""
    print(f"\n{'='*80}")
    print(f"PARAMETER-MATCHED DEPTH SCALING TEST - {ode_name.upper()}")
    print(f"{'='*80}")

    # Create recurrence function
    try:
        recurrence_fn = make_recurrence(
            ode_info["c_list"], ode_info["f_expr"], dtype=DTYPE, device=DEVICE, max_n=20
        )
    except Exception as e:
        print(f"❌ Failed to create recurrence: {e}")
        return None

    # Create datasets
    train_dataset = create_training_dataset(recurrence_fn, num_samples=200)
    val_dataset = create_training_dataset(recurrence_fn, num_samples=30)

    if not train_dataset:
        print(f"❌ No training data created")
        return None

    print(f"Dataset sizes: train={len(train_dataset)}, val={len(val_dataset)}")

    # Test configurations
    configs = [
        ("WIDTH_BASELINE", {"d_model": 128, "nhead": 8, "num_layers": 2}),
        ("DEPTH_MATCHED", {"d_model": 64, "nhead": 4, "num_layers": 8}),
    ]

    results = {}

    for config_name, config in configs:
        print(f"\n{'-'*60}")
        print(f"Testing {config_name}: {config}")
        print(f"{'-'*60}")

        # Create model
        model = ImprovedTransformer(**config).to(device=DEVICE, dtype=DTYPE)
        model = torch.compile(model, mode="max-autotune")
        param_count = count_parameters(model)
        print(f"Parameters: {param_count:,}")

        # Train with analysis
        best_loss, epochs, analyzer, milestones, training_time = train_with_analysis(
            model, train_dataset, val_dataset
        )

        # Analyze training dynamics
        analysis = analyzer.analyze_training_dynamics()

        # Results
        results[config_name] = {
            "params": param_count,
            "loss": best_loss,
            "epochs": epochs,
            "time": training_time,
            "milestones": milestones,
            "analysis": analysis,
            "analyzer": analyzer,
        }

        print(f"Final loss: {best_loss:.2e}")
        print(f"Training analysis: {analysis}")

    return results


def compare_configurations(results):
    """Compare width vs depth configurations"""
    print(f"\n{'='*80}")
    print("COMPARATIVE ANALYSIS")
    print(f"{'='*80}")

    width_result = results["WIDTH_BASELINE"]
    depth_result = results["DEPTH_MATCHED"]

    # Parameter comparison
    print(f"Parameter counts:")
    print(f"  Width baseline: {width_result['params']:,}")
    print(f"  Depth matched: {depth_result['params']:,}")
    print(f"  Ratio: {depth_result['params'] / width_result['params']:.2f}x")

    # Performance comparison
    print(f"\nPerformance comparison:")
    print(f"  Width baseline: {width_result['loss']:.2e}")
    print(f"  Depth matched: {depth_result['loss']:.2e}")
    improvement = width_result["loss"] / depth_result["loss"]
    print(
        f"  Depth improvement: {improvement:.2f}x ({'SUCCESS' if improvement > 1 else 'FAILURE'})"
    )

    # Training dynamics comparison
    print(f"\nTraining dynamics:")
    width_analysis = width_result["analysis"]
    depth_analysis = depth_result["analysis"]

    print(f"  Gradient flow:")
    print(f"    Width avg grad norm: {width_analysis['avg_grad_norm']:.2e}")
    print(f"    Depth avg grad norm: {depth_analysis['avg_grad_norm']:.2e}")
    print(f"    Width vanishing: {width_analysis['vanishing_gradients']}")
    print(f"    Depth vanishing: {depth_analysis['vanishing_gradients']}")

    print(f"  Overfitting:")
    print(f"    Width overfitting: {width_analysis.get('overfitting', 'N/A')}")
    print(f"    Depth overfitting: {depth_analysis.get('overfitting', 'N/A')}")

    print(f"  Convergence:")
    print(f"    Width milestones: {width_result['milestones']}")
    print(f"    Depth milestones: {depth_result['milestones']}")

    # Failure mode analysis
    print(f"\nFailure mode analysis:")
    if improvement < 1:
        print(f"  Depth scaling failed despite parameter matching")
        print(f"  Potential causes:")

        if depth_analysis["vanishing_gradients"]:
            print(f"    ✓ Vanishing gradients detected")
        if depth_analysis["exploding_gradients"]:
            print(f"    ✓ Exploding gradients detected")
        if depth_analysis.get("overfitting", False):
            print(f"    ✓ Overfitting detected")
        if depth_analysis["loss_plateau"]:
            print(f"    ✓ Loss plateau detected")

        print(
            f"  Hypothesis: Depth scaling fails due to training dynamics, not capacity"
        )
    else:
        print(f"  Depth scaling succeeded with parameter matching")
        print(f"  Previous failure was due to insufficient capacity")

    return improvement > 1


def main():
    print("SCIENTIFIC DEPTH SCALING INVESTIGATION")
    print("=" * 60)
    print("Testing parameter-matched depth vs width scaling")
    print("Improvements: AdamW optimizer, SiLU activation, 20-min limit")
    print()

    # Get ODEs
    odes = get_corrected_ode_definitions()

    # Test on Hermite equation (where width scaling succeeded)
    results = test_parameter_matched_depth("hermite", odes["hermite"])

    if results:
        # Compare configurations
        depth_success = compare_configurations(results)

        print(f"\n{'='*80}")
        print("FINAL CONCLUSIONS")
        print(f"{'='*80}")

        if depth_success:
            print("✓ DEPTH SCALING SUCCESS")
            print("  - Parameter matching resolved depth scaling issues")
            print("  - Both width and depth can scale effectively")
            print("  - Previous failures were due to insufficient capacity")
        else:
            print("❌ DEPTH SCALING FAILURE CONFIRMED")
            print("  - Even with parameter matching, depth scaling fails")
            print("  - Fundamental training dynamics issue")
            print("  - Width scaling remains superior approach")

        print(f"\nRecommendations:")
        if depth_success:
            print("  - Both width and depth scaling viable")
            print("  - GPU experiments can explore both dimensions")
            print("  - Combined scaling may be effective")
        else:
            print("  - Focus on width-only scaling")
            print("  - Avoid depth scaling due to training issues")
            print("  - Investigate depth-specific optimizations")


if __name__ == "__main__":
    main()
