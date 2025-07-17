"""
Transformer scaling investigation - test if transformers scale better than RNNs
"""

import torch
import torch.nn as nn
import torch.optim as optim
import time
from single_ode import make_recurrence
from corrected_ode_definitions import get_corrected_ode_definitions

DTYPE = torch.float32
DEVICE = torch.device("cpu")


class ScalableTransformer(nn.Module):
    """Scalable Transformer architecture with configurable parameters"""
    def __init__(self, d_model=64, nhead=4, num_layers=2):
        super().__init__()
        self.d_model = d_model
        self.input_proj = nn.Linear(1, d_model)
        self.pos_encoding = nn.Parameter(torch.randn(50, d_model))  # max length 50
        
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, 
            nhead=nhead, 
            dim_feedforward=d_model*2,
            batch_first=True
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


def count_parameters(model):
    """Count trainable parameters"""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def calculate_transformer_params_exact(d_model, nhead, num_layers):
    """Calculate exact transformer parameters"""
    # Input projection: 1 -> d_model
    input_proj = d_model + d_model  # weights + bias
    
    # Positional encoding: 50 * d_model (learnable parameters)
    pos_encoding = 50 * d_model
    
    # Each transformer layer has:
    # - Multi-head attention: 4 * d_model^2 + 4 * d_model (Q, K, V, O projections with bias)
    # - Layer norm: 2 * d_model (weight + bias)
    # - Feed forward: d_model * (d_model * 2) + (d_model * 2) + (d_model * 2) * d_model + d_model
    # - Layer norm: 2 * d_model
    
    attention_per_layer = 4 * d_model * d_model + 4 * d_model
    ff_per_layer = d_model * (d_model * 2) + (d_model * 2) + (d_model * 2) * d_model + d_model
    ln_per_layer = 2 * 2 * d_model  # 2 layer norms per layer
    
    per_layer = attention_per_layer + ff_per_layer + ln_per_layer
    transformer_layers = num_layers * per_layer
    
    # Output projection: d_model -> 1
    output_proj = d_model + 1  # weights + bias
    
    total = input_proj + pos_encoding + transformer_layers + output_proj
    return total


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


def train_transformer(model, dataset, max_epochs=5000, time_limit=3600):  # 1 hour limit
    """Train transformer with extended epochs"""
    if not dataset:
        return float('inf'), 0, []
    
    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    criterion = nn.MSELoss()
    
    start_time = time.time()
    model.train()
    
    best_loss = float('inf')
    loss_history = []
    convergence_milestones = {}
    
    print(f"    Starting training: {len(dataset)} samples, max {max_epochs} epochs")
    
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
        loss_history.append(avg_loss)
        
        if avg_loss < best_loss:
            best_loss = avg_loss
        
        # Track convergence milestones
        milestones = [1e-3, 1e-4, 1e-5, 1e-6]
        for milestone in milestones:
            if avg_loss < milestone and milestone not in convergence_milestones:
                convergence_milestones[milestone] = epoch
                print(f"    *** MILESTONE: {milestone:.0e} at epoch {epoch} ***")
        
        # Progress reporting
        if epoch % 500 == 0:
            print(f"    Epoch {epoch}: {avg_loss:.2e}")
        
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
    
    return best_loss, epoch + 1, loss_history, convergence_milestones, training_time


def test_transformer_scaling(config_name, d_model, nhead, num_layers, ode_name, ode_info, baseline_results):
    """Test specific transformer configuration"""
    print(f"\n{'='*70}")
    print(f"TESTING {config_name} ON {ode_name.upper()}")
    print(f"{'='*70}")
    print(f"Configuration: d_model={d_model}, nhead={nhead}, num_layers={num_layers}")
    
    # Calculate theoretical parameters
    theoretical_params = calculate_transformer_params_exact(d_model, nhead, num_layers)
    
    # Create recurrence function
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
        return None
    
    # Create dataset
    dataset = create_training_dataset(recurrence_fn, num_samples=200)
    
    if not dataset:
        print(f"❌ No training data created")
        return None
    
    print(f"Dataset size: {len(dataset)}")
    
    # Create model
    model = ScalableTransformer(d_model=d_model, nhead=nhead, num_layers=num_layers).to(device=DEVICE, dtype=DTYPE)
    actual_params = count_parameters(model)
    
    print(f"Theoretical parameters: {theoretical_params:,}")
    print(f"Actual parameters: {actual_params:,}")
    print(f"Parameter ratio vs baseline: {actual_params / baseline_results['baseline_params']:.1f}x")
    
    # Train model
    start_time = time.time()
    best_loss, epochs, loss_history, milestones, training_time = train_transformer(model, dataset)
    
    # Validation test
    val_dataset = create_training_dataset(recurrence_fn, num_samples=30)
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
    
    # Compare to baseline
    baseline_loss = baseline_results[ode_name]['loss']
    baseline_epochs = baseline_results[ode_name]['epochs']
    baseline_time = baseline_results[ode_name]['time']
    
    # Calculate improvements
    loss_improvement = baseline_loss / best_loss
    time_ratio = training_time / baseline_time
    
    # Success evaluation
    success_1e6 = best_loss < 1e-6
    success_1e4 = best_loss < 1e-4
    success_10x = loss_improvement >= 10
    
    print(f"\n{'='*70}")
    print(f"RESULTS SUMMARY - {config_name} ON {ode_name.upper()}")
    print(f"{'='*70}")
    print(f"{'Metric':<25} {'Baseline':<15} {'Scaled':<15} {'Improvement':<15}")
    print("-" * 70)
    print(f"{'Loss':<25} {baseline_loss:<15.2e} {best_loss:<15.2e} {loss_improvement:<15.1f}x")
    print(f"{'Parameters':<25} {baseline_results['baseline_params']:<15,} {actual_params:<15,} {actual_params/baseline_results['baseline_params']:<15.1f}x")
    print(f"{'Epochs':<25} {baseline_epochs:<15} {epochs:<15} {epochs/baseline_epochs:<15.1f}x")
    print(f"{'Time (s)':<25} {baseline_time:<15.1f} {training_time:<15.1f} {time_ratio:<15.1f}x")
    print(f"{'Val Error':<25} {'-':<15} {val_error:<15.2e} {'-':<15}")
    print()
    print(f"SUCCESS CRITERIA:")
    print(f"✓ <1e-6 loss: {'YES' if success_1e6 else 'NO'}")
    print(f"✓ <1e-4 loss: {'YES' if success_1e4 else 'NO'}")
    print(f"✓ >10x improvement: {'YES' if success_10x else 'NO'}")
    print(f"✓ Convergence milestones: {milestones}")
    
    # Evidence level determination
    if loss_improvement >= 100:
        evidence = "STRONG"
    elif loss_improvement >= 10:
        evidence = "MODERATE"
    elif loss_improvement >= 3:
        evidence = "WEAK"
    else:
        evidence = "NONE"
    
    print(f"✓ Evidence level: {evidence}")
    
    return {
        'config': config_name,
        'params': actual_params,
        'loss': best_loss,
        'improvement': loss_improvement,
        'epochs': epochs,
        'time': training_time,
        'val_error': val_error,
        'success_1e6': success_1e6,
        'success_1e4': success_1e4,
        'success_10x': success_10x,
        'evidence': evidence,
        'milestones': milestones,
        'loss_history': loss_history
    }


def main():
    print("TRANSFORMER SCALING INVESTIGATION")
    print("=" * 60)
    print("Goal: Test if transformers scale better than RNNs")
    print("Hypothesis: Transformers follow standard ML scaling laws")
    print("RNN Result: Scaling made performance worse (0.6-0.9x)")
    print()
    
    # Baseline results (transformer baseline from focused test)
    baseline_results = {
        'hermite': {
            'loss': 3.58e-03,
            'epochs': 100,
            'time': 31.0
        },
        'legendre': {
            'loss': 4.17e-02,
            'epochs': 100,
            'time': 34.0
        },
        'baseline_params': 69825  # SimpleTransformer(64, 4, 2)
    }
    
    # Get ODEs
    odes = get_corrected_ode_definitions()
    
    # Phase 1: Width Scaling Test
    print("PHASE 1: WIDTH SCALING TEST")
    print("-" * 50)
    print("Testing: d_model=128, num_layers=2, nhead=8 (4.6x parameters)")
    
    hermite_width_result = test_transformer_scaling(
        "WIDTH_SCALED", 128, 8, 2, 'hermite', odes['hermite'], baseline_results
    )
    
    # Phase 2: Depth Scaling Test
    print("\n\nPHASE 2: DEPTH SCALING TEST")
    print("-" * 50)
    print("Testing: d_model=64, num_layers=4, nhead=4 (2x parameters)")
    
    hermite_depth_result = test_transformer_scaling(
        "DEPTH_SCALED", 64, 4, 4, 'hermite', odes['hermite'], baseline_results
    )
    
    legendre_depth_result = test_transformer_scaling(
        "DEPTH_SCALED", 64, 4, 4, 'legendre', odes['legendre'], baseline_results
    )
    
    # Phase 3: Combined Scaling Test  
    print("\n\nPHASE 3: COMBINED SCALING TEST")
    print("-" * 50)
    print("Testing: d_model=128, num_layers=4, nhead=8 (9.3x parameters)")
    
    # Choose best performer from Phase 1-2
    results_so_far = [hermite_width_result, hermite_depth_result, legendre_depth_result]
    best_result = max([r for r in results_so_far if r], key=lambda x: x['improvement'])
    best_equation = 'hermite' if best_result in [hermite_width_result, hermite_depth_result] else 'legendre'
    
    print(f"Best performer so far: {best_result['config']} on {best_equation} ({best_result['improvement']:.1f}x improvement)")
    
    combined_result = test_transformer_scaling(
        "COMBINED_SCALED", 128, 8, 4, best_equation, odes[best_equation], baseline_results
    )
    
    # Final Analysis
    print(f"\n\n{'='*70}")
    print("FINAL ANALYSIS")
    print(f"{'='*70}")
    
    all_results = [r for r in [hermite_width_result, hermite_depth_result, legendre_depth_result, combined_result] if r]
    
    # Summary table
    print(f"{'Configuration':<20} {'Equation':<10} {'Params':<8} {'Loss':<12} {'Improvement':<12} {'Evidence':<10}")
    print("-" * 75)
    
    for result in all_results:
        eq_name = 'hermite' if result['config'] in ['WIDTH_SCALED', 'DEPTH_SCALED'] and result != legendre_depth_result else 'legendre'
        print(f"{result['config']:<20} {eq_name:<10} {result['params']:<8,} {result['loss']:<12.2e} {result['improvement']:<12.1f}x {result['evidence']:<10}")
    
    # Scaling trend analysis
    print(f"\nSCALING TREND ANALYSIS:")
    
    param_ratios = []
    improvements = []
    
    for result in all_results:
        param_ratio = result['params'] / baseline_results['baseline_params']
        param_ratios.append(param_ratio)
        improvements.append(result['improvement'])
    
    # Check if there's a positive correlation
    if len(param_ratios) >= 3:
        # Simple trend analysis
        avg_improvement = sum(improvements) / len(improvements)
        best_improvement = max(improvements)
        worst_improvement = min(improvements)
        
        print(f"  Parameter scaling range: {min(param_ratios):.1f}x to {max(param_ratios):.1f}x")
        print(f"  Improvement range: {worst_improvement:.1f}x to {best_improvement:.1f}x")
        print(f"  Average improvement: {avg_improvement:.1f}x")
        
        # Determine overall scaling success
        strong_evidence = sum(1 for r in all_results if r['evidence'] == 'STRONG')
        moderate_evidence = sum(1 for r in all_results if r['evidence'] == 'MODERATE')
        
        if strong_evidence >= 1:
            scaling_verdict = "STRONG SCALING SUCCESS"
        elif moderate_evidence >= 2:
            scaling_verdict = "MODERATE SCALING SUCCESS"
        elif best_improvement >= 3:
            scaling_verdict = "WEAK SCALING SUCCESS"
        else:
            scaling_verdict = "SCALING FAILURE"
    
    # GPU Projection
    print(f"\nGPU SCALING PROJECTION:")
    
    if scaling_verdict in ["STRONG SCALING SUCCESS", "MODERATE SCALING SUCCESS"]:
        print(f"✓ RECOMMENDATION: PROCEED TO GPU SCALING")
        print(f"  - Evidence shows transformers scale unlike RNNs")
        print(f"  - Target: d_model=256, num_layers=6, nhead=16 (~4M parameters)")
        print(f"  - Expected improvement: {best_improvement * 2:.0f}-{best_improvement * 5:.0f}x")
        print(f"  - GPU time estimate: 10-20x current training time")
    
    elif scaling_verdict == "WEAK SCALING SUCCESS":
        print(f"⚠ RECOMMENDATION: OPTIMIZE BEFORE GPU")
        print(f"  - Modest scaling evidence")
        print(f"  - Test intermediate sizes first")
        print(f"  - Investigate transformer-specific optimizations")
    
    else:
        print(f"❌ RECOMMENDATION: TRANSFORMERS ALSO DON'T SCALE")
        print(f"  - Similar to RNN scaling failure")
        print(f"  - Focus on algorithmic improvements")
        print(f"  - Recurrence learning may violate scaling laws universally")
    
    print(f"\nOverall scaling verdict: {scaling_verdict}")
    print(f"Transformer vs RNN: {'Better scaling' if best_improvement > 1 else 'Similar scaling failure'}")


if __name__ == "__main__":
    main()