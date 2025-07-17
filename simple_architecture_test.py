"""
Simple architecture test - just the basics to understand the fundamental issue
"""

import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import trange
import sympy as sp
from single_ode import make_recurrence

DTYPE = torch.float32
DEVICE = torch.device("cpu")  # Use CPU for simplicity

class VerySimpleRNN(nn.Module):
    """Extremely simple RNN for testing"""
    def __init__(self):
        super().__init__()
        self.rnn = nn.GRU(1, 16, batch_first=True)
        self.out = nn.Linear(16, 1)
    
    def forward(self, x):
        if x.numel() == 0:
            # Empty input - predict based on zero state
            h = torch.zeros(1, 16, device=x.device, dtype=x.dtype)
        else:
            x_shaped = x.unsqueeze(0).unsqueeze(-1)  # [1, seq_len, 1]
            _, h = self.rnn(x_shaped)
            h = h.squeeze(0)  # [16]
        
        return self.out(h).squeeze()

def create_simple_dataset():
    """Create a very simple dataset to test basic functionality"""
    # Use Airy equation recurrence
    x = sp.symbols("x")
    c_list = [sp.Integer(0), sp.Integer(0), 1 - x]
    f_expr = sp.Integer(0)
    
    recurrence_fn = make_recurrence(c_list, f_expr, dtype=DTYPE, device=DEVICE, max_n=10)
    
    # Create fixed training data
    train_data = []
    
    # Test with specific known patterns
    test_inputs = [
        [0.0, 0.0],
        [1.0, 0.0], 
        [0.0, 1.0],
        [1.0, 1.0],
        [0.5, -0.5],
        [0.35503, -0.25882],  # Airy-like
    ]
    
    for inp in test_inputs:
        prefix = torch.tensor(inp, dtype=DTYPE, device=DEVICE)
        try:
            target = recurrence_fn(prefix)
            train_data.append((prefix, target))
            print(f"Input: {inp} -> Target: {target.item():.6f}")
        except Exception as e:
            print(f"Input: {inp} -> Error: {e}")
    
    return train_data, recurrence_fn

def test_architecture():
    """Test if simple architecture can learn"""
    print("SIMPLE ARCHITECTURE TEST")
    print("=" * 30)
    
    # Create dataset
    train_data, recurrence_fn = create_simple_dataset()
    
    if not train_data:
        print("No training data - recurrence function issue")
        return
    
    print(f"\\nTraining on {len(train_data)} samples")
    
    # Create model
    model = VerySimpleRNN().to(device=DEVICE, dtype=DTYPE)
    optimizer = optim.Adam(model.parameters(), lr=0.01)
    
    # Train
    print("\\nTraining...")
    for epoch in trange(500, desc="Epochs"):
        total_loss = 0.0
        
        for prefix, target in train_data:
            optimizer.zero_grad()
            pred = model(prefix)
            loss = nn.MSELoss()(pred, target)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        
        avg_loss = total_loss / len(train_data)
        
        if epoch % 100 == 0:
            print(f"Epoch {epoch}: Loss = {avg_loss:.2e}")
        
        if avg_loss < 1e-6:
            print(f"Converged at epoch {epoch}")
            break
    
    # Test
    print(f"\\nFinal training loss: {avg_loss:.2e}")
    print("\\nTesting on training data:")
    
    model.eval()
    with torch.no_grad():
        for i, (prefix, target) in enumerate(train_data):
            pred = model(prefix)
            error = abs(pred - target).item()
            print(f"Sample {i}: Target={target.item():.6f}, Pred={pred.item():.6f}, Error={error:.2e}")
    
    # Test on some new data
    print("\\nTesting on new data:")
    test_cases = [
        [0.1, 0.2],
        [0.5, 0.3],
        [-0.1, 0.4]
    ]
    
    for test_case in test_cases:
        prefix = torch.tensor(test_case, dtype=DTYPE, device=DEVICE)
        try:
            target = recurrence_fn(prefix)
            pred = model(prefix)
            error = abs(pred - target).item()
            print(f"Test {test_case}: Target={target.item():.6f}, Pred={pred.item():.6f}, Error={error:.2e}")
        except Exception as e:
            print(f"Test {test_case}: Error - {e}")

def analyze_recurrence():
    """Analyze the recurrence function to understand the pattern"""
    print("\\nRECURRENCE ANALYSIS")
    print("=" * 20)
    
    x = sp.symbols("x")
    c_list = [sp.Integer(0), sp.Integer(0), 1 - x]
    f_expr = sp.Integer(0)
    
    print("Airy equation: u'' = xu")
    print("Recurrence relation: a_{n+2} = a_n / ((n+2)(n+1))")
    print()
    
    # Manual calculation
    print("Manual recurrence calculation:")
    
    # For Airy equation: a_{n+2} = a_n / ((n+2)(n+1))
    # Given a_0, a_1, we can compute a_2, a_3, ...
    
    test_cases = [
        ([1.0, 0.0], "a_0=1, a_1=0"),
        ([0.0, 1.0], "a_0=0, a_1=1"),
        ([0.35503, -0.25882], "Airy solution values")
    ]
    
    for coeffs, desc in test_cases:
        print(f"\\n{desc}:")
        a0, a1 = coeffs
        
        # Calculate a_2 manually: a_2 = a_0 / (2*1) = a_0 / 2
        a2_manual = a0 / 2.0
        print(f"a_2 = a_0 / 2 = {a0} / 2 = {a2_manual}")
        
        # Test with recurrence function
        recurrence_fn = make_recurrence(c_list, f_expr, dtype=DTYPE, device=DEVICE, max_n=10)
        prefix = torch.tensor(coeffs, dtype=DTYPE, device=DEVICE)
        try:
            a2_recurrence = recurrence_fn(prefix).item()
            print(f"Recurrence function gives: {a2_recurrence}")
            print(f"Difference: {abs(a2_manual - a2_recurrence)}")
        except Exception as e:
            print(f"Recurrence function error: {e}")

if __name__ == "__main__":
    analyze_recurrence()
    test_architecture()