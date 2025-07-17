"""
Quick recurrence learning test - focused on basic functionality
"""

import math
import random
import numpy as np
import sympy as sp
import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import trange
from typing import List, Tuple
from single_ode import make_recurrence

DTYPE = torch.float32
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class SimpleRNN(nn.Module):
    """Simple RNN for recurrence learning"""
    def __init__(self, hidden_size: int = 64):
        super().__init__()
        self.hidden_size = hidden_size
        self.input_proj = nn.Linear(1, hidden_size)
        self.rnn = nn.GRU(hidden_size, hidden_size, batch_first=True)
        self.output_proj = nn.Linear(hidden_size, 1)
    
    def forward(self, prefix: torch.Tensor) -> torch.Tensor:
        if prefix.numel() == 0:
            # Empty prefix - use zero hidden state
            hidden = torch.zeros(1, 1, self.hidden_size, device=prefix.device, dtype=prefix.dtype)
            final_hidden = hidden[0, 0]
        else:
            # Process prefix
            x = prefix.unsqueeze(0).unsqueeze(-1)  # [1, seq_len, 1]
            x = self.input_proj(x)  # [1, seq_len, hidden_size]
            _, hidden = self.rnn(x)  # hidden: [1, 1, hidden_size]
            final_hidden = hidden[0, 0]  # [hidden_size]
        
        output = self.output_proj(final_hidden)  # [1]
        return output.squeeze()

class SimpleMLP(nn.Module):
    """Simple MLP for recurrence learning - uses fixed-size input"""
    def __init__(self, max_len: int = 10, hidden_size: int = 64):
        super().__init__()
        self.max_len = max_len
        self.net = nn.Sequential(
            nn.Linear(max_len, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, 1)
        )
    
    def forward(self, prefix: torch.Tensor) -> torch.Tensor:
        # Pad or truncate to fixed size
        if prefix.numel() == 0:
            x = torch.zeros(self.max_len, device=prefix.device, dtype=prefix.dtype)
        else:
            if len(prefix) > self.max_len:
                x = prefix[-self.max_len:]
            else:
                x = torch.cat([torch.zeros(self.max_len - len(prefix), device=prefix.device, dtype=prefix.dtype), prefix])
        
        output = self.net(x)
        return output.squeeze()

def generate_training_data(recurrence_fn, num_samples: int = 500, max_len: int = 8):
    """Generate training data"""
    dataset = []
    
    for _ in range(num_samples):
        # Generate random coefficient sequence
        length = random.randint(1, max_len)
        coeffs = torch.randn(length, dtype=DTYPE, device=DEVICE) * 2.0
        
        # Create prefixes and targets
        for i in range(length):
            prefix = coeffs[:i]
            
            try:
                target = recurrence_fn(prefix)
                dataset.append((prefix, target))
            except:
                continue
    
    return dataset

def test_model(model_name: str, model: nn.Module, dataset: List[Tuple[torch.Tensor, torch.Tensor]]):
    """Test a model on recurrence learning"""
    print(f"\nTesting {model_name}")
    print("-" * 30)
    
    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    
    # Training
    model.train()
    losses = []
    
    for epoch in trange(200, desc="Training"):
        total_loss = 0.0
        random.shuffle(dataset)
        
        for prefix, target in dataset:
            optimizer.zero_grad()
            pred = model(prefix)
            loss = nn.MSELoss()(pred, target)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        
        avg_loss = total_loss / len(dataset)
        losses.append(avg_loss)
        
        if avg_loss < 1e-6:
            print(f"\\nConverged at epoch {epoch}")
            break
    
    final_loss = losses[-1] if losses else float('inf')
    print(f"Final loss: {final_loss:.2e}")
    
    # Test on some specific patterns
    model.eval()
    with torch.no_grad():
        test_cases = [
            torch.tensor([], dtype=DTYPE, device=DEVICE),  # Empty
            torch.tensor([1.0], dtype=DTYPE, device=DEVICE),  # Single element
            torch.tensor([0.35503, -0.25882], dtype=DTYPE, device=DEVICE),  # Airy-like
            torch.tensor([1.0, 0.0, -0.5], dtype=DTYPE, device=DEVICE),  # Harmonic-like
        ]
        
        print("\\nTest cases:")
        for i, test_case in enumerate(test_cases):
            try:
                model_pred = model(test_case)
                recurrence_pred = recurrence_fn(test_case)
                error = abs(model_pred - recurrence_pred).item()
                print(f"Case {i}: Error = {error:.2e}")
            except Exception as e:
                print(f"Case {i}: Failed ({str(e)})")
    
    return final_loss

def main():
    print("QUICK RECURRENCE LEARNING TEST")
    print("=" * 40)
    print("Testing if models can learn simple recurrence function")
    print()
    
    # Use Airy equation recurrence
    x = sp.symbols("x")
    c_list = [sp.Integer(0), sp.Integer(0), 1 - x]  # Airy: u'' = xu
    f_expr = sp.Integer(0)
    
    print("Creating recurrence function...")
    recurrence_fn = make_recurrence(c_list, f_expr, dtype=DTYPE, device=DEVICE, max_n=15)
    
    print("Generating training data...")
    dataset = generate_training_data(recurrence_fn, num_samples=500, max_len=8)
    print(f"Dataset size: {len(dataset)}")
    
    # Test different models
    models = [
        ("Simple RNN", SimpleRNN(hidden_size=64)),
        ("Simple MLP", SimpleMLP(max_len=10, hidden_size=64)),
    ]
    
    results = []
    for name, model in models:
        model.to(device=DEVICE, dtype=DTYPE)
        final_loss = test_model(name, model, dataset)
        results.append((name, final_loss))
    
    print(f"\\n{'='*40}")
    print("SUMMARY")
    print(f"{'='*40}")
    for name, loss in results:
        print(f"{name}: {loss:.2e}")
    
    print(f"\\nExpected: Loss should be < 1e-6 for simple recurrence learning")
    print(f"If not, there's a fundamental issue with the model architectures")

if __name__ == "__main__":
    main()