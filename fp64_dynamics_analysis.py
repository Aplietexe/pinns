"""
FP64 width-scaled training with comprehensive per-layer dynamics analysis
"""

import torch
import torch.nn as nn
import torch.optim as optim
import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import json
from pathlib import Path
from single_ode import make_recurrence
from corrected_ode_definitions import get_corrected_ode_definitions

# Set FP64 precision globally
torch.set_default_dtype(torch.float64)
DTYPE = torch.float64
DEVICE = torch.device("cpu")


class ImprovedTransformerFP64(nn.Module):
    """Improved Transformer with FP64 precision and SiLU activation"""
    
    def __init__(self, d_model=128, nhead=8, num_layers=2):
        super().__init__()
        self.d_model = d_model
        self.input_proj = nn.Linear(1, d_model, dtype=DTYPE)
        self.pos_encoding = nn.Parameter(torch.randn(50, d_model, dtype=DTYPE))
        
        # Custom transformer encoder layer with SiLU activation
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=d_model * 2,
            batch_first=True,
            activation=nn.SiLU(),
            dtype=DTYPE
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.output = nn.Linear(d_model, 1, dtype=DTYPE)
        
        # Initialize parameters properly
        self.apply(self._init_weights)
        
    def _init_weights(self, module):
        """Improved weight initialization for FP64"""
        if isinstance(module, nn.Linear):
            torch.nn.init.xavier_uniform_(module.weight)
            if module.bias is not None:
                torch.nn.init.constant_(module.bias, 0)
        elif isinstance(module, nn.Parameter):
            torch.nn.init.normal_(module, std=0.02)
    
    def forward(self, x):
        x = x.to(dtype=DTYPE, device=DEVICE)
        
        if x.numel() == 0:
            # Empty sequence - use a learned embedding
            h = torch.zeros(1, self.d_model, device=DEVICE, dtype=DTYPE)
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


class ComprehensiveTrainingLogger:
    """Comprehensive per-layer training dynamics logger"""
    
    def __init__(self, model):
        self.model = model
        self.data = {
            'training_log': [],
            'parameter_dynamics': [],
            'gradient_dynamics': [],
            'update_ratios': []
        }
        
        # Previous parameter states for update calculations
        self.prev_params = {}
        self.store_initial_params()
        
        # Layer names for tracking
        self.layer_names = self.get_layer_names()
        
    def get_layer_names(self):
        """Get all layer names for tracking"""
        layer_names = []
        for name, _ in self.model.named_parameters():
            layer_names.append(name)
        layer_names.append('AGGREGATE')  # Add aggregate statistics
        return layer_names
    
    def store_initial_params(self):
        """Store initial parameter states"""
        for name, param in self.model.named_parameters():
            self.prev_params[name] = param.data.clone()
    
    def calculate_layer_stats(self, tensors_dict):
        """Calculate statistics for each layer plus aggregate"""
        stats = {}
        
        # Per-layer statistics
        for name, tensor in tensors_dict.items():
            if tensor is not None:
                tensor_flat = tensor.flatten()
                stats[name] = {
                    'l2_norm': tensor_flat.norm().item(),
                    'mean': tensor_flat.mean().item(),
                    'std': tensor_flat.std().item(),
                    'min': tensor_flat.min().item(),
                    'max': tensor_flat.max().item()
                }
        
        # Aggregate statistics
        all_tensors = [tensor.flatten() for tensor in tensors_dict.values() if tensor is not None]
        if all_tensors:
            all_flat = torch.cat(all_tensors)
            stats['AGGREGATE'] = {
                'l2_norm': all_flat.norm().item(),
                'mean': all_flat.mean().item(),
                'std': all_flat.std().item(),
                'min': all_flat.min().item(),
                'max': all_flat.max().item()
            }
        
        return stats
    
    def log_training_step(self, epoch, train_loss, val_loss, optimizer, training_time):
        """Log comprehensive training dynamics"""
        
        # Core training metrics
        lr = optimizer.param_groups[0]['lr']
        training_data = {
            'epoch': epoch,
            'train_loss': train_loss,
            'val_loss': val_loss if val_loss is not None else np.nan,
            'learning_rate': lr,
            'training_time': training_time
        }
        self.data['training_log'].append(training_data)
        
        # Parameter analysis
        param_tensors = {}
        update_tensors = {}
        
        for name, param in self.model.named_parameters():
            param_tensors[name] = param.data
            
            # Calculate parameter updates
            if name in self.prev_params:
                update = param.data - self.prev_params[name]
                update_tensors[name] = update
                self.prev_params[name] = param.data.clone()
        
        # Parameter statistics
        param_stats = self.calculate_layer_stats(param_tensors)
        for layer_name, stats in param_stats.items():
            param_data = {'epoch': epoch, 'layer': layer_name}
            param_data.update(stats)
            self.data['parameter_dynamics'].append(param_data)
        
        # Gradient analysis
        grad_tensors = {}
        for name, param in self.model.named_parameters():
            if param.grad is not None:
                grad_tensors[name] = param.grad.data
        
        grad_stats = self.calculate_layer_stats(grad_tensors)
        for layer_name, stats in grad_stats.items():
            grad_data = {'epoch': epoch, 'layer': layer_name}
            grad_data.update(stats)
            self.data['gradient_dynamics'].append(grad_data)
        
        # Update-to-weight ratios
        update_ratios = {}
        for name in param_tensors:
            if name in update_tensors:
                param_norm = param_tensors[name].norm().item()
                update_norm = update_tensors[name].norm().item()
                if param_norm > 0:
                    update_ratios[name] = update_norm / param_norm
                else:
                    update_ratios[name] = np.nan
        
        # Calculate aggregate update ratio
        if update_ratios:
            all_param_norm = torch.cat([param_tensors[name].flatten() for name in param_tensors]).norm().item()
            all_update_norm = torch.cat([update_tensors[name].flatten() for name in update_tensors if name in update_tensors]).norm().item()
            if all_param_norm > 0:
                update_ratios['AGGREGATE'] = all_update_norm / all_param_norm
            else:
                update_ratios['AGGREGATE'] = np.nan
        
        for layer_name, ratio in update_ratios.items():
            ratio_data = {
                'epoch': epoch,
                'layer': layer_name,
                'update_to_weight_ratio': ratio
            }
            self.data['update_ratios'].append(ratio_data)
    
    def save_data(self, output_dir="training_dynamics"):
        """Save all collected data to CSV files"""
        output_path = Path(output_dir)
        output_path.mkdir(exist_ok=True)
        
        # Save each dataset
        for data_type, data_list in self.data.items():
            if data_list:
                df = pd.DataFrame(data_list)
                df.to_csv(output_path / f"{data_type}.csv", index=False)
        
        # Save model configuration
        config = {
            'model_type': 'ImprovedTransformerFP64',
            'parameters': {
                'd_model': self.model.d_model,
                'nhead': 8,  # Assuming from our configuration
                'num_layers': len(self.model.transformer.layers)
            },
            'dtype': 'float64',
            'total_parameters': sum(p.numel() for p in self.model.parameters()),
            'layer_names': self.layer_names
        }
        
        with open(output_path / "model_config.json", 'w') as f:
            json.dump(config, f, indent=2)
        
        print(f"Training dynamics data saved to {output_path}")


class DynamicsVisualizer:
    """Create comprehensive visualization of training dynamics"""
    
    def __init__(self, output_dir="training_dynamics"):
        self.output_dir = Path(output_dir)
        self.load_data()
    
    def load_data(self):
        """Load training dynamics data from CSV files"""
        self.training_df = pd.read_csv(self.output_dir / "training_log.csv")
        self.param_df = pd.read_csv(self.output_dir / "parameter_dynamics.csv")
        self.grad_df = pd.read_csv(self.output_dir / "gradient_dynamics.csv")
        self.update_df = pd.read_csv(self.output_dir / "update_ratios.csv")
        
        with open(self.output_dir / "model_config.json", 'r') as f:
            self.config = json.load(f)
    
    def create_subplot_visualization(self, df, metric_col, title, ylabel, log_scale=True):
        """Create subplot visualization for a specific metric"""
        layers = df['layer'].unique()
        n_layers = len(layers)
        
        # Calculate subplot grid
        n_cols = 3
        n_rows = (n_layers + n_cols - 1) // n_cols
        
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(15, 4*n_rows))
        fig.suptitle(title, fontsize=16)
        
        if n_rows == 1:
            axes = axes.reshape(1, -1)
        
        for i, layer in enumerate(layers):
            row = i // n_cols
            col = i % n_cols
            ax = axes[row, col]
            
            layer_data = df[df['layer'] == layer]
            
            ax.plot(layer_data['epoch'], layer_data[metric_col], linewidth=2, alpha=0.8)
            ax.set_title(f"{layer}", fontsize=12)
            ax.set_xlabel("Epoch")
            ax.set_ylabel(ylabel)
            ax.grid(True, alpha=0.3)
            
            if log_scale and (layer_data[metric_col] > 0).any():
                ax.set_yscale('log')
        
        # Hide empty subplots
        for i in range(n_layers, n_rows * n_cols):
            row = i // n_cols
            col = i % n_cols
            axes[row, col].set_visible(False)
        
        plt.tight_layout()
        return fig
    
    def create_training_overview(self):
        """Create training overview plot"""
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        fig.suptitle("Training Overview", fontsize=16)
        
        # Training and validation loss
        axes[0, 0].plot(self.training_df['epoch'], self.training_df['train_loss'], 
                       label='Training Loss', linewidth=2)
        if not self.training_df['val_loss'].isna().all():
            axes[0, 0].plot(self.training_df['epoch'], self.training_df['val_loss'], 
                           label='Validation Loss', linewidth=2)
        axes[0, 0].set_yscale('log')
        axes[0, 0].set_xlabel("Epoch")
        axes[0, 0].set_ylabel("Loss")
        axes[0, 0].set_title("Training Progress")
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)
        
        # Learning rate
        axes[0, 1].plot(self.training_df['epoch'], self.training_df['learning_rate'], 
                       linewidth=2, color='orange')
        axes[0, 1].set_xlabel("Epoch")
        axes[0, 1].set_ylabel("Learning Rate")
        axes[0, 1].set_title("Learning Rate Schedule")
        axes[0, 1].grid(True, alpha=0.3)
        
        # Training time per epoch
        axes[1, 0].plot(self.training_df['epoch'], self.training_df['training_time'], 
                       linewidth=2, color='green')
        axes[1, 0].set_xlabel("Epoch")
        axes[1, 0].set_ylabel("Time (seconds)")
        axes[1, 0].set_title("Training Time per Epoch")
        axes[1, 0].grid(True, alpha=0.3)
        
        # Cumulative training time
        cumulative_time = self.training_df['training_time'].cumsum()
        axes[1, 1].plot(self.training_df['epoch'], cumulative_time, 
                       linewidth=2, color='red')
        axes[1, 1].set_xlabel("Epoch")
        axes[1, 1].set_ylabel("Cumulative Time (seconds)")
        axes[1, 1].set_title("Cumulative Training Time")
        axes[1, 1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        return fig
    
    def create_all_visualizations(self):
        """Create all visualization plots"""
        print("Creating comprehensive training dynamics visualizations...")
        
        # 1. Training overview
        fig1 = self.create_training_overview()
        fig1.savefig(self.output_dir / "training_overview.png", dpi=300, bbox_inches='tight')
        
        # 2. Parameter magnitudes
        fig2 = self.create_subplot_visualization(
            self.param_df, 'l2_norm', 
            "Parameter Magnitudes (L2 Norm)", 
            "Parameter L2 Norm", 
            log_scale=True
        )
        fig2.savefig(self.output_dir / "parameter_magnitudes.png", dpi=300, bbox_inches='tight')
        
        # 3. Gradient norms
        fig3 = self.create_subplot_visualization(
            self.grad_df, 'l2_norm',
            "Gradient Norms (L2 Norm)",
            "Gradient L2 Norm",
            log_scale=True
        )
        fig3.savefig(self.output_dir / "gradient_norms.png", dpi=300, bbox_inches='tight')
        
        # 4. Update-to-weight ratios
        fig4 = self.create_subplot_visualization(
            self.update_df, 'update_to_weight_ratio',
            "Update-to-Weight Ratios",
            "Update/Weight Ratio",
            log_scale=True
        )
        fig4.savefig(self.output_dir / "update_weight_ratios.png", dpi=300, bbox_inches='tight')
        
        # 5. Parameter statistics
        fig5 = self.create_subplot_visualization(
            self.param_df, 'std',
            "Parameter Standard Deviations",
            "Parameter Std Dev",
            log_scale=True
        )
        fig5.savefig(self.output_dir / "parameter_std.png", dpi=300, bbox_inches='tight')
        
        plt.close('all')  # Close all figures to free memory
        
        print(f"All visualizations saved to {self.output_dir}")


def create_training_dataset(recurrence_fn, num_samples=200):
    """Create training dataset with FP64 precision"""
    dataset = []
    
    for _ in range(num_samples):
        prefix_len = torch.randint(2, 10, (1,)).item()
        prefix = torch.randn(prefix_len, dtype=DTYPE, device=DEVICE) * 0.1
        
        try:
            target = recurrence_fn(prefix).to(dtype=DTYPE, device=DEVICE)
            if torch.isfinite(target):
                dataset.append((prefix.clone(), target.clone()))
        except:
            continue
    
    return dataset


def train_with_comprehensive_logging(model, train_dataset, val_dataset=None, 
                                   max_epochs=5000, time_limit=1200):
    """Train model with comprehensive per-layer logging"""
    if not train_dataset:
        return float('inf'), 0, None
    
    # Use AdamW optimizer with FP64
    optimizer = optim.AdamW(model.parameters(), lr=1e-3, weight_decay=0.01)
    criterion = nn.MSELoss()
    
    # Learning rate scheduler
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=100
    )
    
    # Initialize comprehensive logger
    logger = ComprehensiveTrainingLogger(model)
    
    start_time = time.time()
    model.train()
    
    best_loss = float('inf')
    convergence_milestones = {}
    
    print(f"Starting FP64 training with comprehensive logging:")
    print(f"  Dataset: {len(train_dataset)} training, {len(val_dataset) if val_dataset else 0} validation")
    print(f"  Max epochs: {max_epochs}, Time limit: {time_limit/60:.1f} minutes")
    print(f"  Precision: {DTYPE}")
    print(f"  Device: {DEVICE}")
    
    for epoch in range(max_epochs):
        epoch_start = time.time()
        total_loss = 0.0
        
        # Training step
        for prefix, target in train_dataset:
            optimizer.zero_grad()
            pred = model(prefix)
            loss = criterion(pred, target)
            loss.backward()
            
            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            
            optimizer.step()
            total_loss += loss.item()
        
        avg_loss = total_loss / len(train_dataset)
        
        # Validation step
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
        
        # Log comprehensive training dynamics
        epoch_time = time.time() - epoch_start
        logger.log_training_step(epoch, avg_loss, val_loss, optimizer, epoch_time)
        
        if avg_loss < best_loss:
            best_loss = avg_loss
        
        # Update learning rate
        scheduler.step(avg_loss)
        
        # Track convergence milestones
        milestones = [1e-3, 1e-4, 1e-5, 1e-6, 1e-7, 1e-8]
        for milestone in milestones:
            if avg_loss < milestone and milestone not in convergence_milestones:
                convergence_milestones[milestone] = epoch
                print(f"    *** MILESTONE: {milestone:.0e} at epoch {epoch} ***")
        
        # Progress reporting
        if epoch % 100 == 0:
            lr = optimizer.param_groups[0]['lr']
            val_str = f", val: {val_loss:.2e}" if val_loss else ""
            print(f"    Epoch {epoch}: {avg_loss:.2e} (lr: {lr:.2e}{val_str})")
        
        # Early stopping
        if avg_loss < 1e-10:
            print(f"    Early stopping at epoch {epoch}: {avg_loss:.2e}")
            break
        
        # Time limit
        if time.time() - start_time > time_limit:
            print(f"    Time limit reached at epoch {epoch}")
            break
    
    total_time = time.time() - start_time
    print(f"Training completed: {total_time:.1f}s, {epoch+1} epochs")
    print(f"Best loss: {best_loss:.2e}")
    print(f"Convergence milestones: {convergence_milestones}")
    
    return best_loss, epoch + 1, logger, convergence_milestones, total_time


def main():
    print("FP64 WIDTH-SCALED TRAINING WITH COMPREHENSIVE DYNAMICS ANALYSIS")
    print("=" * 70)
    print(f"Using precision: {DTYPE}")
    print(f"Machine epsilon: {torch.finfo(DTYPE).eps}")
    print()
    
    # Get ODE
    odes = get_corrected_ode_definitions()
    ode_name = 'hermite'
    ode_info = odes[ode_name]
    
    # Create recurrence function with FP64 support
    try:
        recurrence_fn = make_recurrence(
            ode_info['c_list'], 
            ode_info['f_expr'], 
            dtype=DTYPE, 
            device=DEVICE, 
            max_n=20
        )
    except Exception as e:
        print(f"❌ Failed to create recurrence: {e}")
        return
    
    # Create datasets
    train_dataset = create_training_dataset(recurrence_fn, num_samples=200)
    val_dataset = create_training_dataset(recurrence_fn, num_samples=30)
    
    if not train_dataset:
        print("❌ No training data created")
        return
    
    print(f"Created datasets: {len(train_dataset)} train, {len(val_dataset)} val")
    
    # Create model
    model = ImprovedTransformerFP64(d_model=128, nhead=8, num_layers=2)
    model = model.to(device=DEVICE, dtype=DTYPE)
    
    # Compile model for speedup
    model = torch.compile(model, mode="max-autotune")
    
    param_count = sum(p.numel() for p in model.parameters())
    print(f"Model parameters: {param_count:,}")
    
    # Train with comprehensive logging
    best_loss, epochs, logger, milestones, total_time = train_with_comprehensive_logging(
        model, train_dataset, val_dataset, time_limit=1200  # 20 minutes
    )
    
    # Save all dynamics data
    logger.save_data("training_dynamics")
    
    # Create comprehensive visualizations
    visualizer = DynamicsVisualizer("training_dynamics")
    visualizer.create_all_visualizations()
    
    print(f"\n{'='*70}")
    print("FINAL RESULTS")
    print(f"{'='*70}")
    print(f"Best loss: {best_loss:.2e}")
    print(f"Total epochs: {epochs}")
    print(f"Total time: {total_time:.1f}s")
    print(f"Convergence milestones: {milestones}")
    
    # Compare with FP32 machine epsilon
    fp32_eps = torch.finfo(torch.float32).eps
    fp64_eps = torch.finfo(torch.float64).eps
    
    print(f"\nNumerical precision analysis:")
    print(f"FP32 machine epsilon: {fp32_eps:.2e}")
    print(f"FP64 machine epsilon: {fp64_eps:.2e}")
    print(f"Precision improvement: {fp32_eps / fp64_eps:.0f}x")
    
    if best_loss < 1e-7:
        print("✓ Achieved sub-1e-7 loss (beyond FP32 machine epsilon)")
    if best_loss < 1e-8:
        print("✓ Achieved sub-1e-8 loss (target performance)")
    
    print(f"\nAll training dynamics data and visualizations saved to 'training_dynamics/'")


if __name__ == "__main__":
    main()