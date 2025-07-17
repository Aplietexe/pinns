"""
Strategic scaling validation test - establish scaling trends within 2 hours
"""

import torch
import torch.nn as nn
import torch.optim as optim
import time
from single_ode import make_recurrence
from corrected_ode_definitions import get_corrected_ode_definitions

DTYPE = torch.float32
DEVICE = torch.device("cpu")


class SimpleRNN(nn.Module):
    def __init__(self, hidden_size=64):
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


def count_parameters(model):
    """Count trainable parameters"""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def create_scaled_dataset(recurrence_fn, num_samples=545):
    """Create scaled training dataset"""
    dataset = []
    
    for _ in range(num_samples):
        prefix_len = torch.randint(2, 12, (1,)).item()  # Slightly longer sequences
        prefix = torch.randn(prefix_len, dtype=DTYPE, device=DEVICE) * 0.1
        
        try:
            target = recurrence_fn(prefix)
            if torch.isfinite(target):
                dataset.append((prefix.clone(), target.clone()))
        except:
            continue
    
    return dataset


def train_with_scaling(model, dataset, max_epochs=993, time_limit=2700):  # 45 min limit
    """Train model with optimal scaling"""
    if not dataset:
        return float('inf'), 0, []
    
    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    criterion = nn.MSELoss()
    
    start_time = time.time()
    model.train()
    
    best_loss = float('inf')
    convergence_epoch = -1
    loss_history = []
    
    print(f"    Starting training: {len(dataset)} samples, {max_epochs} epochs")
    
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
        
        # Check for convergence milestones
        if avg_loss < 1e-6 and convergence_epoch == -1:
            convergence_epoch = epoch
            print(f"    *** CROSSED 1e-6 THRESHOLD AT EPOCH {epoch} ***")
        
        # Progress reporting
        if epoch % 200 == 0:
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
    
    return best_loss, epoch + 1, loss_history, convergence_epoch, training_time


def test_scaled_model(ode_name, ode_info, baseline_results):
    """Test scaled model and compare to baseline"""
    print(f"\n{'='*60}")
    print(f"TESTING SCALED RNN-256 ON {ode_name.upper()}")
    print(f"{'='*60}")
    
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
    
    # Create scaled dataset
    dataset = create_scaled_dataset(recurrence_fn, num_samples=545)
    
    if not dataset:
        print(f"❌ No training data created")
        return None
    
    print(f"Dataset size: {len(dataset)} (vs 200 baseline)")
    
    # Create scaled model
    model = SimpleRNN(hidden_size=256).to(device=DEVICE, dtype=DTYPE)
    param_count = count_parameters(model)
    print(f"Model parameters: {param_count:,} (vs 50,433 baseline)")
    
    # Train with scaling
    start_time = time.time()
    best_loss, epochs, loss_history, convergence_epoch, training_time = train_with_scaling(
        model, dataset, max_epochs=993
    )
    
    # Validation test
    val_dataset = create_scaled_dataset(recurrence_fn, num_samples=50)
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
    baseline_params = baseline_results[ode_name]['params']
    
    # Calculate improvements
    loss_improvement = baseline_loss / best_loss
    param_ratio = param_count / baseline_params
    time_ratio = training_time / baseline_time
    
    # Success evaluation
    success_1e6 = best_loss < 1e-6
    success_1e8 = best_loss < 1e-8
    
    print(f"\n{'='*60}")
    print(f"RESULTS SUMMARY - {ode_name.upper()}")
    print(f"{'='*60}")
    print(f"{'Metric':<20} {'Baseline':<15} {'Scaled':<15} {'Improvement':<15}")
    print("-" * 65)
    print(f"{'Loss':<20} {baseline_loss:<15.2e} {best_loss:<15.2e} {loss_improvement:<15.1f}x")
    print(f"{'Parameters':<20} {baseline_params:<15,} {param_count:<15,} {param_ratio:<15.1f}x")
    print(f"{'Epochs':<20} {baseline_epochs:<15} {epochs:<15} {epochs/baseline_epochs:<15.1f}x")
    print(f"{'Time (s)':<20} {baseline_time:<15.1f} {training_time:<15.1f} {time_ratio:<15.1f}x")
    print(f"{'Val Error':<20} {'-':<15} {val_error:<15.2e} {'-':<15}")
    print()
    print(f"SUCCESS CRITERIA:")
    print(f"✓ <1e-6 loss: {'YES' if success_1e6 else 'NO'}")
    print(f"✓ <1e-8 loss: {'YES' if success_1e8 else 'NO'}")
    print(f"✓ Convergence epoch: {convergence_epoch if convergence_epoch != -1 else 'No'}")
    
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
        'loss': best_loss,
        'improvement': loss_improvement,
        'epochs': epochs,
        'time': training_time,
        'params': param_count,
        'val_error': val_error,
        'success_1e6': success_1e6,
        'success_1e8': success_1e8,
        'convergence_epoch': convergence_epoch,
        'evidence': evidence,
        'loss_history': loss_history
    }


def ablation_study(ode_name, ode_info, baseline_results):
    """Quick ablation study on scaling dimensions"""
    print(f"\n{'='*60}")
    print(f"ABLATION STUDY - {ode_name.upper()}")
    print(f"{'='*60}")
    
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
    
    # Test configurations
    configs = [
        ("Model Only", {"hidden_size": 256, "dataset_size": 200, "epochs": 500}),
        ("Data Only", {"hidden_size": 128, "dataset_size": 545, "epochs": 500}),
        ("Training Only", {"hidden_size": 128, "dataset_size": 200, "epochs": 993}),
    ]
    
    results = {}
    
    for name, config in configs:
        print(f"\nTesting {name}: {config}")
        
        # Create model and dataset
        model = SimpleRNN(hidden_size=config['hidden_size']).to(device=DEVICE, dtype=DTYPE)
        dataset = create_scaled_dataset(recurrence_fn, num_samples=config['dataset_size'])
        
        if not dataset:
            print(f"❌ Failed to create dataset")
            continue
        
        # Quick training (10 minutes max)
        start_time = time.time()
        best_loss, epochs, _, _, training_time = train_with_scaling(
            model, dataset, max_epochs=config['epochs'], time_limit=600
        )
        
        # Calculate improvement
        baseline_loss = baseline_results[ode_name]['loss']
        improvement = baseline_loss / best_loss
        
        results[name] = {
            'loss': best_loss,
            'improvement': improvement,
            'time': training_time,
            'params': count_parameters(model)
        }
        
        print(f"  Result: {best_loss:.2e} ({improvement:.1f}x improvement)")
    
    # Compare ablation results
    print(f"\nABLATION SUMMARY:")
    print(f"{'Method':<15} {'Loss':<12} {'Improvement':<12} {'Time':<8}")
    print("-" * 50)
    
    for name, result in results.items():
        print(f"{name:<15} {result['loss']:<12.2e} {result['improvement']:<12.1f}x {result['time']:<8.1f}s")
    
    return results


def main():
    print("STRATEGIC SCALING VALIDATION TEST")
    print("=" * 50)
    print("Goal: Establish scaling trends within 2 hours")
    print("Focus: Near-miss cases (Hermite, Legendre)")
    print("Success: 10-100x improvement to cross 1e-6 threshold")
    print()
    
    # Baseline results from previous experiments
    baseline_results = {
        'hermite': {
            'loss': 1.26e-05,
            'epochs': 500,
            'time': 218.4,
            'params': 50433
        },
        'legendre': {
            'loss': 9.77e-05,
            'epochs': 500, 
            'time': 155.8,
            'params': 50433
        }
    }
    
    # Get ODEs
    odes = get_corrected_ode_definitions()
    
    # Phase 1: Direct scaling tests
    print("PHASE 1: DIRECT SCALING TESTS")
    print("-" * 40)
    
    results = {}
    
    # Test Hermite (needs 10x improvement)
    hermite_result = test_scaled_model('hermite', odes['hermite'], baseline_results)
    if hermite_result:
        results['hermite'] = hermite_result
    
    # Test Legendre (needs 100x improvement)
    legendre_result = test_scaled_model('legendre', odes['legendre'], baseline_results)
    if legendre_result:
        results['legendre'] = legendre_result
    
    # Phase 2: Ablation study on best performer
    print("\nPHASE 2: ABLATION STUDY")
    print("-" * 40)
    
    best_equation = None
    best_improvement = 0
    
    for eq_name, result in results.items():
        if result['improvement'] > best_improvement:
            best_improvement = result['improvement']
            best_equation = eq_name
    
    if best_equation:
        print(f"Running ablation on {best_equation} (best improvement: {best_improvement:.1f}x)")
        ablation_results = ablation_study(best_equation, odes[best_equation], baseline_results)
    
    # Final analysis
    print(f"\n{'='*60}")
    print("FINAL ANALYSIS")
    print(f"{'='*60}")
    
    # Evidence summary
    total_successes = 0
    evidence_levels = []
    
    for eq_name, result in results.items():
        print(f"\n{eq_name.upper()} RESULTS:")
        print(f"  Loss improvement: {result['improvement']:.1f}x")
        print(f"  Success (1e-6): {'YES' if result['success_1e6'] else 'NO'}")
        print(f"  Evidence level: {result['evidence']}")
        
        if result['success_1e6']:
            total_successes += 1
        evidence_levels.append(result['evidence'])
    
    # Overall assessment
    print(f"\nOVERALL ASSESSMENT:")
    print(f"  Equations tested: {len(results)}")
    print(f"  Successes (<1e-6): {total_successes}")
    print(f"  Evidence levels: {evidence_levels}")
    
    # Recommendations
    if total_successes >= 1 and any(evidence == "STRONG" for evidence in evidence_levels):
        print(f"\n✓ RECOMMENDATION: PROCEED TO GPU SCALING")
        print(f"  - Strong evidence that scaling works")
        print(f"  - Ready for RNN-512, RNN-1024 experiments")
        print(f"  - Target: <1e-8 on all equations")
    
    elif total_successes >= 1 or any(evidence == "MODERATE" for evidence in evidence_levels):
        print(f"\n⚠ RECOMMENDATION: OPTIMIZE SCALING")
        print(f"  - Moderate evidence for scaling")
        print(f"  - Test intermediate sizes (RNN-384)")
        print(f"  - Refine scaling ratios before GPU")
    
    else:
        print(f"\n❌ RECOMMENDATION: INVESTIGATE ALTERNATIVES")
        print(f"  - Scaling evidence is weak")
        print(f"  - Consider algorithmic improvements")
        print(f"  - Investigate training dynamics")
    
    print(f"\nExperiment completed. Ready for next phase based on results.")


if __name__ == "__main__":
    main()