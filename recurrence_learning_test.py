"""
Fundamental Test: Can our models learn the recurrence function?
Bottom-up approach: Test pure recurrence learning before attempting full PINN
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
from single_ode import make_recurrence, _ARTransformerCoeffNet
from improved_rnn_pinn import ImprovedRNNCoeffNet

DTYPE = torch.float32  # Use float32 for compatibility
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def generate_random_coefficients(num_samples: int, min_length: int, max_length: int) -> List[torch.Tensor]:
    """Generate random coefficient sequences of varying lengths"""
    sequences = []
    
    for _ in range(num_samples):
        length = random.randint(min_length, max_length)
        # Generate coefficients with varying magnitudes to test robustness
        coeffs = torch.randn(length, dtype=DTYPE, device=DEVICE) * random.uniform(0.1, 10.0)
        sequences.append(coeffs)
    
    return sequences

def get_ode_validation_data() -> List[Tuple[str, torch.Tensor]]:
    """Get actual ODE solution coefficients for validation"""
    x = sp.symbols("x")
    
    # Known coefficient patterns from ODE solutions
    validation_data = []
    
    # Airy equation solution (should work well)
    airy_coeffs = torch.tensor([0.35503, -0.25882, 0.0, 0.059, -0.019, 0.001], dtype=DTYPE, device=DEVICE)
    validation_data.append(("Airy", airy_coeffs))
    
    # Simple harmonic oscillator (cos(x))
    harmonic_coeffs = torch.tensor([1.0, 0.0, -0.5, 0.0, 0.041667, 0.0], dtype=DTYPE, device=DEVICE)
    validation_data.append(("Harmonic", harmonic_coeffs))
    
    # Exponential (e^x)
    exp_coeffs = torch.tensor([1.0, 1.0, 0.5, 0.166667, 0.041667, 0.008333], dtype=DTYPE, device=DEVICE)
    validation_data.append(("Exponential", exp_coeffs))
    
    # Hermite-like pattern (alternating zeros)
    hermite_coeffs = torch.tensor([-120.0, 0.0, 720.0, 0.0, -120.0, 0.0], dtype=DTYPE, device=DEVICE)
    validation_data.append(("Hermite", hermite_coeffs))
    
    # Legendre-like pattern (mostly zeros)
    legendre_coeffs = torch.tensor([0.0, 0.0, 0.0, 0.0, 0.0, 30.0], dtype=DTYPE, device=DEVICE)
    validation_data.append(("Legendre", legendre_coeffs))
    
    return validation_data

def create_recurrence_dataset(
    recurrence_fn, 
    sequences: List[torch.Tensor]
) -> List[Tuple[torch.Tensor, torch.Tensor]]:
    """Create training dataset: (prefix, next_coefficient)"""
    dataset = []
    
    for seq in sequences:
        # Create all possible prefixes and their next coefficients
        for i in range(len(seq)):
            prefix = seq[:i]  # Can be empty
            
            # Get next coefficient from recurrence
            try:
                next_coeff = recurrence_fn(prefix)
                dataset.append((prefix, next_coeff))
            except:
                # Skip if recurrence fails (e.g., insufficient coefficients)
                continue
    
    return dataset

def test_recurrence_learning(
    model_name: str,
    model: nn.Module,
    recurrence_fn,
    train_sequences: List[torch.Tensor],
    val_data: List[Tuple[str, torch.Tensor]],
    epochs: int = 2000,
    learning_rate: float = 1e-3
) -> dict:
    """Test if model can learn the recurrence function"""
    
    print(f"\n{'='*50}")
    print(f"TESTING: {model_name}")
    print(f"{'='*50}")
    
    # Create training dataset
    train_dataset = create_recurrence_dataset(recurrence_fn, train_sequences)
    print(f"Training samples: {len(train_dataset)}")
    
    # Optimizer
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    
    # Training loop
    model.train()
    losses = []
    
    pbar = trange(epochs, desc="Training")
    for epoch in pbar:
        total_loss = 0.0
        
        # Shuffle training data
        random.shuffle(train_dataset)
        
        for prefix, target in train_dataset:
            optimizer.zero_grad()
            
            # Forward pass
            prediction = model(prefix)
            
            # MSE loss
            loss = nn.MSELoss()(prediction, target)
            
            # Backward pass
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
        
        avg_loss = total_loss / len(train_dataset)
        losses.append(avg_loss)
        pbar.set_postfix(loss=avg_loss)
        
        # Early stopping if converged
        if avg_loss < 1e-12:
            print(f"\\nConverged at epoch {epoch}")
            break
    
    # Validation on actual ODE coefficients
    model.eval()
    val_results = {}
    
    print(f"\\nValidation on ODE coefficients:")
    print("-" * 30)
    
    with torch.no_grad():
        for ode_name, coeffs in val_data:
            val_errors = []
            
            # Test all possible prefixes
            for i in range(len(coeffs)):
                prefix = coeffs[:i]
                
                try:
                    # Get predictions
                    model_pred = model(prefix)
                    recurrence_pred = recurrence_fn(prefix)
                    
                    # Compare with actual next coefficient
                    if i < len(coeffs) - 1:
                        actual_next = coeffs[i]
                        model_error = abs(model_pred - actual_next).item()
                        recurrence_error = abs(recurrence_pred - actual_next).item()
                        
                        val_errors.append({
                            'prefix_len': i,
                            'model_error': model_error,
                            'recurrence_error': recurrence_error,
                            'actual': actual_next.item(),
                            'model_pred': model_pred.item(),
                            'recurrence_pred': recurrence_pred.item()
                        })
                except:
                    continue
            
            if val_errors:
                avg_model_error = np.mean([e['model_error'] for e in val_errors])
                avg_recurrence_error = np.mean([e['recurrence_error'] for e in val_errors])
                
                print(f"{ode_name}: Model={avg_model_error:.2e}, Recurrence={avg_recurrence_error:.2e}")
                val_results[ode_name] = {
                    'avg_model_error': avg_model_error,
                    'avg_recurrence_error': avg_recurrence_error,
                    'details': val_errors
                }
    
    final_train_loss = losses[-1] if losses else float('inf')
    print(f"\\nFinal training loss: {final_train_loss:.2e}")
    
    return {
        'model_name': model_name,
        'final_train_loss': final_train_loss,
        'training_losses': losses,
        'validation_results': val_results
    }

def run_recurrence_tests():
    """Run recurrence learning tests on all architectures"""
    
    print("RECURRENCE LEARNING TESTS")
    print("=" * 60)
    print("Testing if models can learn recurrence function with perfect data")
    print("This should be MUCH easier than the full PINN problem")
    print()
    
    # Use Airy equation recurrence as test case
    x = sp.symbols("x")
    c_list = [sp.Integer(0), sp.Integer(0), 1 - x]  # Airy equation
    f_expr = sp.Integer(0)
    
    recurrence_fn = make_recurrence(
        c_list, f_expr, 
        dtype=DTYPE, device=DEVICE, max_n=20
    )
    
    # Generate training data
    print("Generating training data...")
    train_sequences = generate_random_coefficients(
        num_samples=1000, 
        min_length=1, 
        max_length=15
    )
    
    # Get validation data
    val_data = get_ode_validation_data()
    
    # Test architectures
    models_to_test = [
        {
            'name': 'Transformer',
            'model': _ARTransformerCoeffNet(
                d_model=64,
                nhead=4,
                n_layers=2,
                max_len=20,
                dtype=DTYPE
            ).to(device=DEVICE)
        },
        {
            'name': 'RNN-GRU',
            'model': ImprovedRNNCoeffNet(
                hidden_size=128,
                num_layers=1,
                rnn_type="GRU",
                normalization_type="adaptive"
            ).to(device=DEVICE).to(dtype=DTYPE)
        },
        {
            'name': 'RNN-LSTM',
            'model': ImprovedRNNCoeffNet(
                hidden_size=128,
                num_layers=1,
                rnn_type="LSTM",
                normalization_type="adaptive"
            ).to(device=DEVICE).to(dtype=DTYPE)
        }
    ]
    
    results = []
    
    for model_config in models_to_test:
        result = test_recurrence_learning(
            model_name=model_config['name'],
            model=model_config['model'],
            recurrence_fn=recurrence_fn,
            train_sequences=train_sequences,
            val_data=val_data,
            epochs=1000,
            learning_rate=1e-3
        )
        results.append(result)
    
    # Summary
    print(f"\\n{'='*60}")
    print("RECURRENCE LEARNING SUMMARY")
    print(f"{'='*60}")
    
    for result in results:
        print(f"{result['model_name']}: Final loss = {result['final_train_loss']:.2e}")
    
    print(f"\\nExpected: Loss should be < 1e-10 for simple recurrence learning")
    print(f"If not, we need to investigate gradient flow and training dynamics")
    
    return results

if __name__ == "__main__":
    results = run_recurrence_tests()