"""
Quick test with corrected recurrence - just verify basic functionality
"""

import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import trange
import sympy as sp
from single_ode import make_recurrence

DTYPE = torch.float32
DEVICE = torch.device("cpu")


class VerySimpleRNN(nn.Module):
    """Very simple RNN for quick testing"""

    def __init__(self):
        super().__init__()
        self.rnn = nn.GRU(1, 32, batch_first=True)
        self.out = nn.Linear(32, 1)

    def forward(self, x):
        if x.numel() == 0:
            h = torch.zeros(1, 32, device=x.device, dtype=x.dtype)
        else:
            x_shaped = x.unsqueeze(0).unsqueeze(-1)
            _, h = self.rnn(x_shaped)
            h = h.squeeze(0)
        return self.out(h).squeeze()


def main():
    print("QUICK TEST: CORRECTED RECURRENCE LEARNING")
    print("=" * 45)

    # Use harmonic oscillator (simple and well-known)
    x = sp.symbols("x")
    c_list = [sp.Integer(1), sp.Integer(0), sp.Integer(1)]  # u'' + u = 0
    f_expr = sp.Integer(0)

    print("Testing harmonic oscillator: u'' + u = 0")
    print("Expected to learn cos(x) series: [1, 0, -1/2, 0, 1/24, ...]")
    print()

    # Create recurrence function
    recurrence_fn = make_recurrence(
        c_list, f_expr, dtype=DTYPE, device=DEVICE, max_n=10
    )

    # Create simple training data
    train_data = []
    cos_coeffs = [1.0, 0.0, -0.5, 0.0, 1.0 / 24, 0.0, -1.0 / 720, 0.0]

    # Create training samples from cos(x) series
    for i in range(2, 6):
        prefix = torch.tensor(cos_coeffs[:i], dtype=DTYPE, device=DEVICE)
        target = recurrence_fn(prefix)
        train_data.append((prefix, target))
        print(f"Training sample: {cos_coeffs[:i]} -> {target.item():.6f}")

    if not train_data:
        print("No training data created!")
        return

    # Create and train model
    print(f"\\nTraining simple RNN on {len(train_data)} samples...")
    model = VerySimpleRNN().to(device=DEVICE, dtype=DTYPE)
    optimizer = optim.Adam(model.parameters(), lr=0.01)

    # Train
    for epoch in trange(200, desc="Training"):
        total_loss = 0.0

        for prefix, target in train_data:
            optimizer.zero_grad()
            pred = model(prefix)
            loss = nn.MSELoss()(pred, target)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        avg_loss = total_loss / len(train_data)

    print(f"Final loss: {avg_loss:.2e}")

    # Test
    print(f"\\nTesting model vs recurrence function:")
    model.eval()
    with torch.no_grad():
        for i, (prefix, target) in enumerate(train_data):
            pred = model(prefix)
            error = abs(pred - target).item()
            print(
                f"Sample {i}: Target={target.item():.6f}, Pred={pred.item():.6f}, Error={error:.2e}"
            )

    # Test with new data
    print(f"\\nTesting with new coefficient patterns:")
    test_cases = [
        [1.0, 0.0],  # Should predict -0.5
        [0.0, 1.0],  # Should predict 0.0
        [1.0, 0.0, -0.5],  # Should predict 0.0
    ]

    for test_case in test_cases:
        prefix = torch.tensor(test_case, dtype=DTYPE, device=DEVICE)
        try:
            recurrence_pred = recurrence_fn(prefix).item()
            model_pred = model(prefix).item()
            error = abs(model_pred - recurrence_pred)

            print(
                f"  {test_case} -> Recurrence: {recurrence_pred:.6f}, Model: {model_pred:.6f}, Error: {error:.2e}"
            )
        except Exception as e:
            print(f"  {test_case} -> Error: {e}")


if __name__ == "__main__":
    main()
