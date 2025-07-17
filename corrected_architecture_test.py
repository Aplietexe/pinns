"""
Test architectures with corrected recurrence relations
"""

import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import trange
import sympy as sp
from single_ode import make_recurrence
import time

DTYPE = torch.float32
DEVICE = torch.device("cpu")

def get_corrected_ode_for_testing():
    """Get the corrected harmonic oscillator ODE for testing"""
    x = sp.symbols("x")
    
    # Use harmonic oscillator since it has a simple, well-known solution
    return {
        "name": "Harmonic oscillator",
        "equation": "u'' + u = 0",
        "c_list": [sp.Integer(1), sp.Integer(0), sp.Integer(1)],
        "f_expr": sp.Integer(0),
        "expected_coeffs": [1.0, 0.0, -0.5, 0.0, 1.0/24, 0.0, -1.0/720, 0.0],  # cos(x)
        "description": "Should learn cos(x) series"
    }

def create_training_dataset(recurrence_fn, num_samples=200):
    """Create training dataset for recurrence learning"""
    dataset = []
    
    # Create diverse training data
    for _ in range(num_samples):
        # Random prefix length (2-6 for harmonic oscillator)
        prefix_len = torch.randint(2, 7, (1,)).item()
        
        # Random coefficients with reasonable magnitudes
        prefix = torch.randn(prefix_len, dtype=DTYPE, device=DEVICE) * 0.5
        
        try:
            target = recurrence_fn(prefix)
            dataset.append((prefix.clone(), target.clone()))
        except:
            continue
    
    return dataset

class SimpleRNN(nn.Module):
    """Simple RNN for testing"""
    def __init__(self, hidden_size=64):
        super().__init__()
        self.hidden_size = hidden_size
        self.rnn = nn.GRU(1, hidden_size, batch_first=True)
        self.output = nn.Linear(hidden_size, 1)
    
    def forward(self, x):
        if x.numel() == 0:
            # Empty input
            h = torch.zeros(1, self.hidden_size, device=x.device, dtype=x.dtype)
        else:
            x_shaped = x.unsqueeze(0).unsqueeze(-1)  # [1, seq_len, 1]
            _, h = self.rnn(x_shaped)
            h = h.squeeze(0)  # [hidden_size]
        
        return self.output(h).squeeze()

class SimpleMLP(nn.Module):
    """Simple MLP for testing (fixed input size)"""
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
        # Pad or truncate to fixed size
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

def train_and_test_model(model_name, model, train_dataset, test_dataset, epochs=300):
    """Train and test a model"""
    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    
    print(f"\\nTraining {model_name}...")
    start_time = time.time()
    
    # Training
    model.train()
    for epoch in range(epochs):
        total_loss = 0.0
        
        for prefix, target in train_dataset:
            optimizer.zero_grad()
            pred = model(prefix)
            loss = nn.MSELoss()(pred, target)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        
        avg_loss = total_loss / len(train_dataset)
        
        # Early stopping
        if avg_loss < 1e-8:
            break
        
        if epoch % 100 == 0:
            print(f"  Epoch {epoch}: Loss = {avg_loss:.2e}")
    
    train_time = time.time() - start_time
    
    # Testing
    model.eval()
    test_errors = []
    
    with torch.no_grad():
        for prefix, target in test_dataset:
            pred = model(prefix)
            error = abs(pred - target).item()
            test_errors.append(error)
    
    avg_test_error = sum(test_errors) / len(test_errors)
    
    print(f"  Final train loss: {avg_loss:.2e}")
    print(f"  Test error: {avg_test_error:.2e}")
    print(f"  Training time: {train_time:.2f}s")
    print(f"  Epochs used: {epoch + 1}")
    
    return {
        'train_loss': avg_loss,
        'test_error': avg_test_error,
        'train_time': train_time,
        'epochs': epoch + 1
    }

def test_with_known_sequence(model, recurrence_fn, known_coeffs):
    """Test model with known coefficient sequence"""
    print(f"  Testing with known cos(x) sequence:")
    
    model.eval()
    with torch.no_grad():
        for i in range(2, min(len(known_coeffs), 6)):
            prefix = torch.tensor(known_coeffs[:i], dtype=DTYPE, device=DEVICE)
            
            try:
                model_pred = model(prefix).item()
                recurrence_pred = recurrence_fn(prefix).item()
                expected = known_coeffs[i]
                
                model_error = abs(model_pred - expected)
                recurrence_error = abs(recurrence_pred - expected)
                
                print(f"    a_{i}: Model={model_pred:.6f}, Expected={expected:.6f}, Error={model_error:.2e}")
                
            except Exception as e:
                print(f"    a_{i}: Error - {e}")

def main():
    print("ARCHITECTURE TEST WITH CORRECTED RECURRENCE")
    print("=" * 50)
    
    # Get corrected ODE
    ode = get_corrected_ode_for_testing()
    print(f"Testing with: {ode['name']}")
    print(f"Equation: {ode['equation']}")
    print(f"Expected: {ode['description']}")
    print()
    
    # Create recurrence function
    recurrence_fn = make_recurrence(
        ode['c_list'], 
        ode['f_expr'], 
        dtype=DTYPE, 
        device=DEVICE, 
        max_n=15
    )
    
    # Create datasets
    print("Creating datasets...")
    train_dataset = create_training_dataset(recurrence_fn, num_samples=300)
    test_dataset = create_training_dataset(recurrence_fn, num_samples=50)
    
    print(f"Train samples: {len(train_dataset)}")
    print(f"Test samples: {len(test_dataset)}")
    
    # Test different architectures
    architectures = [
        ("Simple RNN", SimpleRNN(hidden_size=64)),
        ("Simple MLP", SimpleMLP(input_size=8, hidden_size=64))
    ]
    
    results = []
    
    for name, model in architectures:
        model.to(device=DEVICE, dtype=DTYPE)
        
        result = train_and_test_model(name, model, train_dataset, test_dataset)
        result['name'] = name
        results.append(result)
        
        # Test with known sequence
        test_with_known_sequence(model, recurrence_fn, ode['expected_coeffs'])
    
    # Summary
    print(f"\\n{'='*50}")
    print("ARCHITECTURE COMPARISON SUMMARY")
    print(f"{'='*50}")
    print(f"{'Architecture':<15} {'Train Loss':<12} {'Test Error':<12} {'Time(s)':<8} {'Epochs':<8}")
    print("-" * 55)
    
    for result in results:
        print(f"{result['name']:<15} {result['train_loss']:<12.2e} {result['test_error']:<12.2e} "
              f"{result['train_time']:<8.2f} {result['epochs']:<8}")
    
    print(f"\\nExpected results:")
    print(f"- Train loss should be < 1e-6 for good architecture")
    print(f"- Test error should be < 1e-4 for good generalization")
    print(f"- Should converge quickly (< 100 epochs)")
    print(f"- If all succeed, the recurrence learning works!")

if __name__ == "__main__":
    main()