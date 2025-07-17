"""Debug RNN tensor shapes"""

import torch
import sympy as sp
from rnn_pinn import SimpleRNNCoeffNet

# Test what happens with tensor shapes
dtype = torch.float64
device = torch.device("cpu")

# Create a simple RNN
net = SimpleRNNCoeffNet(hidden_size=128, num_layers=1, rnn_type="LSTM", init_scale=0.1)
net = net.to(device=device, dtype=dtype)

# Test empty prefix
prefix = torch.empty(0, dtype=dtype, device=device)
print(f"Empty prefix shape: {prefix.shape}, ndim: {prefix.ndim}")

try:
    coef1 = net(prefix)
    print(f"First coefficient shape: {coef1.shape}, ndim: {coef1.ndim}")
    print(f"First coefficient: {coef1}")
    
    # Check what's happening step by step
    print(f"coef1 is 0-d: {coef1.ndim == 0}")
    print(f"coef1 is 1-d: {coef1.ndim == 1}")
    
    # Try to concatenate with empty tensor
    try:
        empty = torch.empty(0, dtype=dtype, device=device)
        print(f"Empty tensor shape: {empty.shape}")
        result = torch.cat([empty, coef1])
        print(f"Cat result shape: {result.shape}")
    except Exception as e:
        print(f"Cat error: {e}")
        
    # Test with unsqueeze if needed
    if coef1.ndim == 0:
        coef1_fixed = coef1.unsqueeze(0)
        print(f"Fixed coef1 shape: {coef1_fixed.shape}")
        result = torch.cat([empty, coef1_fixed])
        print(f"Cat with fixed coef1 shape: {result.shape}")
    
except Exception as e:
    print(f"Error: {e}")
    print(f"coef1 shape: {coef1.shape}")
    print(f"coef1 ndim: {coef1.ndim}")
    print(f"coef1 is scalar: {coef1.ndim == 0}")
    
    # Try manual fix
    if coef1.ndim == 0:
        prefix = coef1.unsqueeze(0)
        print(f"Fixed prefix shape: {prefix.shape}")
        coef2 = net(prefix)
        print(f"Second coefficient: {coef2}")