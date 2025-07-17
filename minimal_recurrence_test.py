"""
Minimal recurrence learning test - absolute basics
"""

import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import trange
import sympy as sp
from single_ode import make_recurrence

DTYPE = torch.float32
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def test_recurrence_function():
    """Test the recurrence function itself"""
    print("Testing recurrence function...")
    
    # Simple Airy equation: u'' = xu
    x = sp.symbols("x")
    c_list = [sp.Integer(0), sp.Integer(0), 1 - x]  # Coefficients: 0, 0, 1-x
    f_expr = sp.Integer(0)
    
    recurrence_fn = make_recurrence(c_list, f_expr, dtype=DTYPE, device=DEVICE, max_n=10)
    
    # Test with some simple inputs
    test_cases = [
        torch.tensor([], dtype=DTYPE, device=DEVICE),  # Empty
        torch.tensor([1.0], dtype=DTYPE, device=DEVICE),  # a0 = 1
        torch.tensor([1.0, 0.0], dtype=DTYPE, device=DEVICE),  # a0=1, a1=0
        torch.tensor([1.0, 0.0, 0.5], dtype=DTYPE, device=DEVICE),  # a0=1, a1=0, a2=0.5
    ]
    
    for i, test_case in enumerate(test_cases):
        try:
            result = recurrence_fn(test_case)
            print(f"Input {i}: {test_case.tolist() if test_case.numel() > 0 else 'empty'} -> {result.item():.6f}")
        except Exception as e:
            print(f"Input {i}: Failed - {str(e)}")
    
    return recurrence_fn

def test_simple_learning():
    """Test if a simple model can learn a trivial pattern"""
    print("\nTesting simple pattern learning...")
    
    # Create a very simple model
    model = nn.Sequential(
        nn.Linear(3, 32),
        nn.ReLU(),
        nn.Linear(32, 1)
    ).to(device=DEVICE, dtype=DTYPE)
    
    # Create trivial training data: f(x) = sum(x)
    def simple_target(x):
        return torch.sum(x)
    
    # Generate training data
    train_data = []
    for _ in range(100):
        x = torch.randn(3, dtype=DTYPE, device=DEVICE)
        y = simple_target(x)
        train_data.append((x, y))
    
    # Train
    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    
    print("Training simple model...")
    for epoch in trange(100, desc="Training"):
        total_loss = 0.0
        
        for x, y in train_data:
            optimizer.zero_grad()
            pred = model(x)
            loss = nn.MSELoss()(pred.squeeze(), y)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        
        avg_loss = total_loss / len(train_data)
        if avg_loss < 1e-6:
            print(f"Converged at epoch {epoch}")
            break
    
    print(f"Final loss: {avg_loss:.2e}")
    
    # Test
    with torch.no_grad():
        test_x = torch.tensor([1.0, 2.0, 3.0], dtype=DTYPE, device=DEVICE)
        expected = simple_target(test_x)
        predicted = model(test_x)
        print(f"Test: Expected {expected:.6f}, Got {predicted.item():.6f}")

class MinimalRNN(nn.Module):
    """Minimal RNN for testing"""
    def __init__(self, hidden_size: int = 32):
        super().__init__()
        self.hidden_size = hidden_size
        
        # Very simple: just encode the last coefficient
        self.encoder = nn.Linear(1, hidden_size)
        self.decoder = nn.Linear(hidden_size, 1)
    
    def forward(self, prefix: torch.Tensor) -> torch.Tensor:
        if prefix.numel() == 0:
            # Empty prefix -> use zero
            hidden = torch.zeros(self.hidden_size, dtype=prefix.dtype, device=prefix.device)
        else:
            # Use only the last coefficient
            last_coeff = prefix[-1:].unsqueeze(0)  # [1, 1]
            hidden = self.encoder(last_coeff).squeeze(0)  # [hidden_size]
        
        output = self.decoder(hidden)  # [1]
        return output.squeeze()

def test_minimal_recurrence():
    """Test minimal recurrence learning"""
    print("\nTesting minimal recurrence learning...")
    
    # Get recurrence function
    recurrence_fn = test_recurrence_function()
    
    # Create simple model
    model = MinimalRNN(hidden_size=32).to(device=DEVICE, dtype=DTYPE)
    
    # Generate minimal training data
    train_data = []
    for _ in range(50):
        # Simple fixed-length sequences
        prefix = torch.randn(2, dtype=DTYPE, device=DEVICE) * 0.5  # Small values
        try:
            target = recurrence_fn(prefix)
            train_data.append((prefix, target))
        except:
            continue
    
    print(f"Generated {len(train_data)} training samples")
    
    if len(train_data) == 0:
        print("No training data generated - recurrence function issue")
        return
    
    # Train
    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    
    for epoch in trange(100, desc="Training"):
        total_loss = 0.0
        
        for prefix, target in train_data:
            optimizer.zero_grad()
            pred = model(prefix)
            loss = nn.MSELoss()(pred, target)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        
        avg_loss = total_loss / len(train_data)
        if avg_loss < 1e-6:
            print(f"Converged at epoch {epoch}")
            break
    
    print(f"Final loss: {avg_loss:.2e}")
    
    # Test on first few training examples
    model.eval()
    with torch.no_grad():
        for i, (prefix, target) in enumerate(train_data[:3]):
            pred = model(prefix)
            error = abs(pred - target).item()
            print(f"Sample {i}: Expected {target.item():.6f}, Got {pred.item():.6f}, Error {error:.2e}")

def main():
    print("MINIMAL RECURRENCE LEARNING TEST")
    print("=" * 40)
    
    # Test 1: Simple pattern learning
    test_simple_learning()
    
    # Test 2: Minimal recurrence learning
    test_minimal_recurrence()
    
    print("\nDone!")

if __name__ == "__main__":
    main()