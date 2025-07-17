"""
Deep test of RNN architecture with more training epochs
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


def create_training_dataset(recurrence_fn, num_samples=200):
    """Create larger training dataset"""
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


def extended_train_test(model, dataset, max_epochs=500, time_limit=600):
    """Extended training with 10-minute limit"""
    if not dataset:
        return float('inf'), 0
    
    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    criterion = nn.MSELoss()
    
    start_time = time.time()
    model.train()
    
    best_loss = float('inf')
    convergence_epoch = -1
    
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
        
        if avg_loss < best_loss:
            best_loss = avg_loss
        
        # Check for convergence to 1e-6 threshold
        if avg_loss < 1e-6 and convergence_epoch == -1:
            convergence_epoch = epoch
        
        # Early stopping if very good
        if avg_loss < 1e-8:
            break
        
        # Time limit
        if time.time() - start_time > time_limit:
            break
        
        # Progress report every 100 epochs
        if epoch % 100 == 0:
            print(f"    Epoch {epoch}: {avg_loss:.2e}")
    
    return best_loss, epoch + 1, convergence_epoch


def test_rnn_deep(ode_name, ode_info):
    """Deep test of RNN on specific equation"""
    print(f"\nDeep RNN test on {ode_name}")
    print("=" * 40)
    
    # Create recurrence function
    try:
        recurrence_fn = make_recurrence(
            ode_info['c_list'], 
            ode_info['f_expr'], 
            dtype=DTYPE, 
            device=DEVICE, 
            max_n=15
        )
    except Exception as e:
        print(f"❌ Failed to create recurrence: {e}")
        return None
    
    # Create larger dataset
    dataset = create_training_dataset(recurrence_fn, num_samples=200)
    
    if not dataset:
        print(f"❌ No training data created")
        return None
    
    print(f"Dataset size: {len(dataset)}")
    
    # Test different RNN sizes
    sizes = [32, 64, 128]
    results = {}
    
    for hidden_size in sizes:
        print(f"\nTesting RNN with hidden_size={hidden_size}")
        print("-" * 30)
        
        model = SimpleRNN(hidden_size=hidden_size).to(device=DEVICE, dtype=DTYPE)
        
        start_time = time.time()
        best_loss, epochs, convergence_epoch = extended_train_test(model, dataset)
        training_time = time.time() - start_time
        
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
        
        success = best_loss < 1e-6
        converged = convergence_epoch != -1
        
        print(f"Best loss: {best_loss:.2e}")
        print(f"Val error: {val_error:.2e}")
        print(f"Epochs: {epochs}")
        print(f"Time: {training_time:.1f}s")
        print(f"Success (< 1e-6): {'✓' if success else '❌'}")
        print(f"Converged at epoch: {convergence_epoch if converged else 'No'}")
        
        results[hidden_size] = {
            'loss': best_loss,
            'val_error': val_error,
            'epochs': epochs,
            'time': training_time,
            'success': success,
            'convergence_epoch': convergence_epoch
        }
    
    return results


def test_known_sequence(ode_name, ode_info):
    """Test RNN on known coefficient sequence"""
    print(f"\nTesting RNN on known {ode_name} sequence")
    print("-" * 40)
    
    # Create recurrence function
    try:
        recurrence_fn = make_recurrence(
            ode_info['c_list'], 
            ode_info['f_expr'], 
            dtype=DTYPE, 
            device=DEVICE, 
            max_n=15
        )
    except Exception as e:
        print(f"❌ Failed to create recurrence: {e}")
        return
    
    # Known sequences
    known_sequences = {
        'harmonic': [1.0, 0.0, -0.5, 0.0, 1.0/24, 0.0, -1.0/720, 0.0],  # cos(x)
        'airy': [0.35503, -0.25882, 0.0, 0.0592, -0.0217, 0.0, 0.0198, -0.0064],  # Airy Ai(x)
        'hermite': [-120.0, 0.0, 720.0, 0.0, -480.0, 0.0, 64.0, 0.0],  # H_6(x)
        'legendre': [0.0, 0.0, 0.0, 0.0, 0.0, 63.0/8, 0.0, -35.0/4]  # P_5(x)
    }
    
    if ode_name not in known_sequences:
        print(f"No known sequence for {ode_name}")
        return
    
    known_coeffs = known_sequences[ode_name]
    
    # Test recurrence function on known sequence
    print("Testing recurrence function on known sequence:")
    for i in range(2, min(len(known_coeffs), 6)):
        prefix = torch.tensor(known_coeffs[:i], dtype=DTYPE, device=DEVICE)
        try:
            pred = recurrence_fn(prefix).item()
            expected = known_coeffs[i]
            error = abs(pred - expected)
            print(f"  a_{i}: Expected={expected:.6f}, Recurrence={pred:.6f}, Error={error:.2e}")
        except Exception as e:
            print(f"  a_{i}: Error - {e}")
    
    # Train RNN on known sequence
    print(f"\nTraining RNN on known sequence:")
    
    # Create training data from known sequence
    train_data = []
    for i in range(2, min(len(known_coeffs), 8)):
        prefix = torch.tensor(known_coeffs[:i], dtype=DTYPE, device=DEVICE)
        target = torch.tensor(known_coeffs[i], dtype=DTYPE, device=DEVICE)
        train_data.append((prefix, target))
    
    model = SimpleRNN(hidden_size=64).to(device=DEVICE, dtype=DTYPE)
    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    criterion = nn.MSELoss()
    
    # Train on known sequence
    model.train()
    for epoch in range(300):
        total_loss = 0.0
        
        for prefix, target in train_data:
            optimizer.zero_grad()
            pred = model(prefix)
            loss = criterion(pred, target)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        
        avg_loss = total_loss / len(train_data)
        
        if avg_loss < 1e-8:
            break
        
        if epoch % 100 == 0:
            print(f"  Epoch {epoch}: {avg_loss:.2e}")
    
    print(f"Final loss: {avg_loss:.2e}")
    
    # Test on known sequence
    print("Testing RNN on known sequence:")
    model.eval()
    with torch.no_grad():
        for i, (prefix, target) in enumerate(train_data):
            pred = model(prefix)
            error = abs(pred - target).item()
            print(f"  Sample {i}: Target={target.item():.6f}, Pred={pred.item():.6f}, Error={error:.2e}")


def main():
    print("DEEP RNN RECURRENCE LEARNING TEST")
    print("=" * 50)
    print("Extended training with larger datasets and more epochs")
    print()
    
    # Get ODEs
    odes = get_corrected_ode_definitions()
    
    # Test priority equations
    priority_equations = ['harmonic', 'airy', 'hermite', 'legendre']
    
    all_results = {}
    
    for ode_name in priority_equations:
        if ode_name in odes:
            results = test_rnn_deep(ode_name, odes[ode_name])
            all_results[ode_name] = results
            
            # Also test on known sequence
            test_known_sequence(ode_name, odes[ode_name])
    
    # Summary
    print(f"\n{'='*60}")
    print("DEEP RNN RESULTS SUMMARY")
    print(f"{'='*60}")
    
    print(f"{'Equation':<10} {'Size':<6} {'Loss':<12} {'Val Error':<12} {'Success':<8} {'Converged':<10}")
    print("-" * 60)
    
    for ode_name in all_results:
        if all_results[ode_name]:
            for size, result in all_results[ode_name].items():
                success_mark = "✓" if result['success'] else "❌"
                conv_mark = f"Ep{result['convergence_epoch']}" if result['convergence_epoch'] != -1 else "No"
                print(f"{ode_name:<10} {size:<6} {result['loss']:<12.2e} {result['val_error']:<12.2e} {success_mark:<8} {conv_mark:<10}")
    
    # Overall analysis
    print(f"\n{'='*60}")
    print("OVERALL ANALYSIS")
    print(f"{'='*60}")
    
    total_successes = 0
    total_tests = 0
    
    for ode_name in all_results:
        if all_results[ode_name]:
            for size, result in all_results[ode_name].items():
                if result['success']:
                    total_successes += 1
                total_tests += 1
    
    print(f"Total successes: {total_successes}/{total_tests}")
    print(f"Success rate: {total_successes/total_tests*100:.1f}%")
    
    # Best configurations
    print(f"\nBest RNN configuration for each equation:")
    for ode_name in all_results:
        if all_results[ode_name]:
            best_size = min(all_results[ode_name].keys(), key=lambda s: all_results[ode_name][s]['loss'])
            best_result = all_results[ode_name][best_size]
            success_mark = "✓" if best_result['success'] else "❌"
            print(f"  {ode_name}: size={best_size}, loss={best_result['loss']:.2e} {success_mark}")


if __name__ == "__main__":
    main()