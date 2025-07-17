"""
Architecture comparison for recurrence learning
Based on minimal test insights
"""

import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import trange
import sympy as sp
from single_ode import make_recurrence, _ARTransformerCoeffNet
from improved_rnn_pinn import ImprovedRNNCoeffNet
import time

DTYPE = torch.float32
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def generate_recurrence_dataset(recurrence_fn, num_samples=200, max_prefix_len=8):
    """Generate training dataset for recurrence learning"""
    dataset = []
    
    for _ in range(num_samples):
        # Generate random prefix of length 2 to max_prefix_len
        prefix_len = torch.randint(2, max_prefix_len + 1, (1,)).item()
        prefix = torch.randn(prefix_len, dtype=DTYPE, device=DEVICE) * 0.5
        
        try:
            target = recurrence_fn(prefix)
            dataset.append((prefix.clone(), target.clone()))
        except:
            continue
    
    return dataset

def train_model(model, dataset, epochs=100, lr=1e-3):
    """Train a model on the dataset"""
    optimizer = optim.Adam(model.parameters(), lr=lr)
    
    start_time = time.time()
    
    for epoch in range(epochs):
        total_loss = 0.0
        
        for prefix, target in dataset:
            optimizer.zero_grad()
            pred = model(prefix)
            loss = nn.MSELoss()(pred, target)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        
        avg_loss = total_loss / len(dataset)
        
        # Early stopping
        if avg_loss < 1e-8:
            break
    
    train_time = time.time() - start_time
    return avg_loss, train_time, epoch + 1

def test_model(model, test_dataset):
    """Test a model on test dataset"""
    model.eval()
    total_error = 0.0
    
    with torch.no_grad():
        for prefix, target in test_dataset:
            pred = model(prefix)
            error = abs(pred - target).item()
            total_error += error
    
    return total_error / len(test_dataset)

def main():
    print("ARCHITECTURE COMPARISON FOR RECURRENCE LEARNING")
    print("=" * 55)
    
    # Create recurrence function
    x = sp.symbols("x")
    c_list = [sp.Integer(0), sp.Integer(0), 1 - x]  # Airy equation
    f_expr = sp.Integer(0)
    
    recurrence_fn = make_recurrence(c_list, f_expr, dtype=DTYPE, device=DEVICE, max_n=15)
    
    # Generate datasets
    print("Generating datasets...")
    train_dataset = generate_recurrence_dataset(recurrence_fn, num_samples=300, max_prefix_len=8)
    test_dataset = generate_recurrence_dataset(recurrence_fn, num_samples=50, max_prefix_len=8)
    
    print(f"Train samples: {len(train_dataset)}")
    print(f"Test samples: {len(test_dataset)}")
    print()
    
    # Define architectures to test
    architectures = [
        {
            'name': 'Simple RNN',
            'model': 'rnn',
            'rnn_type': 'RNN'
        },
        {
            'name': 'Simple GRU',
            'model': 'rnn',
            'rnn_type': 'GRU'
        },
        {
            'name': 'Simple LSTM',
            'model': 'rnn',
            'rnn_type': 'LSTM'
        },
        {
            'name': 'Simple MLP (fixed-size)',
            'model': nn.Sequential(
                nn.Linear(8, 64),
                nn.ReLU(),
                nn.Linear(64, 32),
                nn.ReLU(),
                nn.Linear(32, 1)
            ),
            'input_processor': lambda x: torch.nn.functional.pad(x, (0, max(0, 8 - len(x))))[:8]  # Pad to size 8
        }
    ]
    
    # Test each architecture
    results = []
    
    for arch in architectures:
        print(f"Testing {arch['name']}...")
        print("-" * 30)
        
        # Create model based on type
        if arch['model'] == 'rnn':
            # Create RNN model
            class SimpleRNNModel(nn.Module):
                def __init__(self, rnn_type, hidden_size=32):
                    super().__init__()
                    self.hidden_size = hidden_size
                    
                    if rnn_type == 'RNN':
                        self.rnn = nn.RNN(1, hidden_size, batch_first=True)
                    elif rnn_type == 'GRU':
                        self.rnn = nn.GRU(1, hidden_size, batch_first=True)
                    elif rnn_type == 'LSTM':
                        self.rnn = nn.LSTM(1, hidden_size, batch_first=True)
                    
                    self.output = nn.Linear(hidden_size, 1)
                
                def forward(self, x):
                    if x.numel() == 0:
                        # Empty input
                        hidden = torch.zeros(1, self.hidden_size, device=x.device, dtype=x.dtype)
                    else:
                        x_shaped = x.unsqueeze(0).unsqueeze(-1)  # [1, seq_len, 1]
                        rnn_out, _ = self.rnn(x_shaped)
                        hidden = rnn_out[0, -1, :]  # Last output
                    
                    return self.output(hidden).squeeze()
            
            model = SimpleRNNModel(arch['rnn_type']).to(device=DEVICE, dtype=DTYPE)
        
        else:
            # Regular model
            model = arch['model'].to(device=DEVICE, dtype=DTYPE)
            
            # Create wrapped model for variable-length input
            class WrappedModel(nn.Module):
                def __init__(self, model, input_processor):
                    super().__init__()
                    self.model = model
                    self.input_processor = input_processor
                
                def forward(self, x):
                    processed = self.input_processor(x)
                    return self.model(processed).squeeze()
            
            model = WrappedModel(model, arch['input_processor'])
        
        # Train
        train_loss, train_time, epochs_used = train_model(model, train_dataset, epochs=200)
        
        # Test
        test_error = test_model(model, test_dataset)
        
        results.append({
            'name': arch['name'],
            'train_loss': train_loss,
            'test_error': test_error,
            'train_time': train_time,
            'epochs_used': epochs_used
        })
        
        print(f"Train loss: {train_loss:.2e}")
        print(f"Test error: {test_error:.2e}")
        print(f"Train time: {train_time:.2f}s")
        print(f"Epochs used: {epochs_used}")
        print()
    
    # Summary
    print("=" * 55)
    print("ARCHITECTURE COMPARISON SUMMARY")
    print("=" * 55)
    
    print(f"{'Architecture':<20} {'Train Loss':<12} {'Test Error':<12} {'Time (s)':<10} {'Epochs':<8}")
    print("-" * 65)
    
    for result in results:
        print(f"{result['name']:<20} {result['train_loss']:<12.2e} {result['test_error']:<12.2e} "
              f"{result['train_time']:<10.2f} {result['epochs_used']:<8}")
    
    print()
    print("Key insights:")
    print("1. Simple architectures should achieve <1e-6 loss easily")
    print("2. Fast convergence (< 50 epochs) indicates good architecture")
    print("3. Low test error indicates good generalization")
    print("4. If all fail, there's a fundamental issue with the setup")

if __name__ == "__main__":
    main()