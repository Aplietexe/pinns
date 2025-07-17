"""
Focused recurrence learning test with short training times
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


class SimpleTransformer(nn.Module):
    def __init__(self, d_model=64, nhead=4, num_layers=2):
        super().__init__()
        self.d_model = d_model
        self.input_proj = nn.Linear(1, d_model)
        self.pos_encoding = nn.Parameter(torch.randn(20, d_model))
        
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
            h = torch.zeros(1, self.d_model, device=x.device, dtype=x.dtype)
        else:
            seq_len = len(x)
            x_shaped = x.unsqueeze(0).unsqueeze(-1)
            x_proj = self.input_proj(x_shaped)
            x_proj = x_proj + self.pos_encoding[:seq_len].unsqueeze(0)
            output = self.transformer(x_proj)
            h = output[0, -1, :]
        return self.output(h).squeeze()


class SimpleMLP(nn.Module):
    def __init__(self, input_size=8, hidden_size=64):
        super().__init__()
        self.input_size = input_size
        self.net = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, 1)
        )
        
    def forward(self, x):
        if x.numel() == 0:
            processed = torch.zeros(self.input_size, device=x.device, dtype=x.dtype)
        else:
            if len(x) > self.input_size:
                processed = x[-self.input_size:]
            else:
                processed = torch.cat([
                    torch.zeros(self.input_size - len(x), device=x.device, dtype=x.dtype), 
                    x
                ])
        return self.net(processed).squeeze()


def create_training_dataset(recurrence_fn, num_samples=100):
    """Create small training dataset"""
    dataset = []
    
    for _ in range(num_samples):
        prefix_len = torch.randint(2, 8, (1,)).item()
        prefix = torch.randn(prefix_len, dtype=DTYPE, device=DEVICE) * 0.1
        
        try:
            target = recurrence_fn(prefix)
            if torch.isfinite(target):
                dataset.append((prefix.clone(), target.clone()))
        except:
            continue
    
    return dataset


def quick_train_test(model, dataset, max_epochs=100, time_limit=300):
    """Quick training with 5-minute limit"""
    if not dataset:
        return float('inf'), 0
    
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
        
        # Early stopping
        if avg_loss < 1e-8:
            break
        
        # Time limit
        if time.time() - start_time > time_limit:
            break
    
    return avg_loss, epoch + 1


def test_arch_on_equation(arch_name, model_class, ode_name, ode_info):
    """Test single architecture on single equation"""
    print(f"\nTesting {arch_name} on {ode_name}")
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
        return None
    
    # Create dataset
    dataset = create_training_dataset(recurrence_fn, num_samples=100)
    
    if not dataset:
        print(f"❌ No training data created")
        return None
    
    print(f"Dataset size: {len(dataset)}")
    
    # Create and train model
    model = model_class().to(device=DEVICE, dtype=DTYPE)
    
    start_time = time.time()
    final_loss, epochs = quick_train_test(model, dataset)
    training_time = time.time() - start_time
    
    # Test on validation
    val_dataset = create_training_dataset(recurrence_fn, num_samples=20)
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
    
    # Results
    print(f"Final loss: {final_loss:.2e}")
    print(f"Val error: {val_error:.2e}")
    print(f"Epochs: {epochs}")
    print(f"Time: {training_time:.1f}s")
    
    success = final_loss < 1e-6
    print(f"Success: {'✓' if success else '❌'}")
    
    return {
        'loss': final_loss,
        'val_error': val_error,
        'epochs': epochs,
        'time': training_time,
        'success': success
    }


def main():
    print("FOCUSED RECURRENCE LEARNING TEST")
    print("=" * 50)
    print("Quick test with 5-minute limit per architecture-equation pair")
    print()
    
    # Get key ODEs
    odes = get_corrected_ode_definitions()
    
    # Test priority equations
    priority_equations = ['harmonic', 'airy', 'hermite', 'legendre']
    
    # Test key architectures
    architectures = [
        ("RNN", SimpleRNN),
        ("Transformer", SimpleTransformer),
        ("MLP", SimpleMLP)
    ]
    
    results = {}
    
    for arch_name, model_class in architectures:
        print(f"\n{'='*50}")
        print(f"TESTING {arch_name} ARCHITECTURE")
        print(f"{'='*50}")
        
        results[arch_name] = {}
        
        for ode_name in priority_equations:
            if ode_name in odes:
                result = test_arch_on_equation(arch_name, model_class, ode_name, odes[ode_name])
                results[arch_name][ode_name] = result
    
    # Summary
    print(f"\n{'='*50}")
    print("SUMMARY")
    print(f"{'='*50}")
    
    print(f"{'Architecture':<12} {'Equation':<10} {'Loss':<12} {'Val Error':<12} {'Success':<8}")
    print("-" * 55)
    
    for arch_name in results:
        for ode_name in results[arch_name]:
            result = results[arch_name][ode_name]
            if result:
                success_mark = "✓" if result['success'] else "❌"
                print(f"{arch_name:<12} {ode_name:<10} {result['loss']:<12.2e} {result['val_error']:<12.2e} {success_mark:<8}")
    
    # Analysis
    print(f"\n{'='*50}")
    print("ANALYSIS")
    print(f"{'='*50}")
    
    for arch_name in results:
        successes = sum(1 for r in results[arch_name].values() if r and r['success'])
        total = len([r for r in results[arch_name].values() if r])
        print(f"{arch_name}: {successes}/{total} equations successful")
    
    # Best for each equation
    print(f"\nBest architecture for each equation:")
    for ode_name in priority_equations:
        best_arch = None
        best_loss = float('inf')
        
        for arch_name in results:
            if ode_name in results[arch_name] and results[arch_name][ode_name]:
                if results[arch_name][ode_name]['loss'] < best_loss:
                    best_loss = results[arch_name][ode_name]['loss']
                    best_arch = arch_name
        
        if best_arch:
            print(f"  {ode_name}: {best_arch} (loss: {best_loss:.2e})")


if __name__ == "__main__":
    main()