"""
Calculate exact parameter counts and scaling relationships for all models
"""

import torch
import torch.nn as nn
import math

def count_parameters(model):
    """Count total parameters in a model"""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def calculate_gru_params(input_size, hidden_size):
    """Calculate GRU parameters exactly"""
    # GRU has 3 gates (reset, update, new), each with input and hidden weights + bias
    # Input weights: 3 * input_size * hidden_size
    # Hidden weights: 3 * hidden_size * hidden_size  
    # Bias: 3 * hidden_size (input bias) + 3 * hidden_size (hidden bias)
    
    input_weights = 3 * input_size * hidden_size
    hidden_weights = 3 * hidden_size * hidden_size
    bias = 6 * hidden_size
    
    return input_weights + hidden_weights + bias

def calculate_linear_params(input_size, output_size):
    """Calculate Linear layer parameters"""
    weights = input_size * output_size
    bias = output_size
    return weights + bias

def calculate_rnn_total_params(hidden_size):
    """Calculate total parameters for our RNN architecture"""
    # Input is 1D (coefficient values)
    # GRU: 1 -> hidden_size
    gru_params = calculate_gru_params(1, hidden_size)
    
    # Linear: hidden_size -> 1
    linear_params = calculate_linear_params(hidden_size, 1)
    
    total = gru_params + linear_params
    return total, gru_params, linear_params

def calculate_transformer_params(d_model, nhead, num_layers):
    """Calculate transformer parameters"""
    # Input projection: 1 -> d_model
    input_proj = calculate_linear_params(1, d_model)
    
    # Positional encoding: 50 * d_model (learnable parameters)
    pos_encoding = 50 * d_model
    
    # Each transformer layer has:
    # - Multi-head attention: 4 * d_model^2 (Q, K, V, O projections)
    # - Layer norm: 2 * d_model (weight + bias)
    # - Feed forward: 2 * d_model * (d_model * 2) + d_model * 2 + d_model (two linear layers)
    # - Layer norm: 2 * d_model
    
    attention_per_layer = 4 * d_model * d_model
    ff_per_layer = d_model * (d_model * 2) + (d_model * 2) * d_model + d_model * 2 + d_model
    ln_per_layer = 2 * 2 * d_model  # 2 layer norms per layer
    
    per_layer = attention_per_layer + ff_per_layer + ln_per_layer
    transformer_layers = num_layers * per_layer
    
    # Output projection: d_model -> 1
    output_proj = calculate_linear_params(d_model, 1)
    
    total = input_proj + pos_encoding + transformer_layers + output_proj
    return total, input_proj, pos_encoding, transformer_layers, output_proj

def calculate_mlp_params(input_size, hidden_size):
    """Calculate MLP parameters"""
    # input_size -> hidden_size -> hidden_size -> 1
    layer1 = calculate_linear_params(input_size, hidden_size)
    layer2 = calculate_linear_params(hidden_size, hidden_size)
    layer3 = calculate_linear_params(hidden_size, 1)
    
    total = layer1 + layer2 + layer3
    return total, layer1, layer2, layer3

def calculate_flops_per_forward(model_type, **kwargs):
    """Calculate FLOPs per forward pass"""
    if model_type == "rnn":
        hidden_size = kwargs['hidden_size']
        seq_len = kwargs.get('seq_len', 5)  # average sequence length
        
        # GRU FLOPs: 6 * hidden_size * (input_size + hidden_size) * seq_len
        gru_flops = 6 * hidden_size * (1 + hidden_size) * seq_len
        
        # Linear FLOPs: 2 * hidden_size * 1
        linear_flops = 2 * hidden_size * 1
        
        return gru_flops + linear_flops
    
    elif model_type == "transformer":
        d_model = kwargs['d_model']
        nhead = kwargs['nhead']
        num_layers = kwargs['num_layers']
        seq_len = kwargs.get('seq_len', 5)
        
        # Simplified transformer FLOPs (attention + FFN)
        attention_flops = 2 * seq_len * d_model * d_model * nhead * num_layers
        ffn_flops = 2 * seq_len * d_model * (d_model * 2) * num_layers
        
        return attention_flops + ffn_flops
    
    elif model_type == "mlp":
        input_size = kwargs['input_size']
        hidden_size = kwargs['hidden_size']
        
        # 3 linear layers
        layer1_flops = 2 * input_size * hidden_size
        layer2_flops = 2 * hidden_size * hidden_size
        layer3_flops = 2 * hidden_size * 1
        
        return layer1_flops + layer2_flops + layer3_flops

def calculate_total_compute(model_type, dataset_size, epochs, **kwargs):
    """Calculate total compute for training"""
    flops_per_forward = calculate_flops_per_forward(model_type, **kwargs)
    
    # Each training step: forward + backward (roughly 3x forward)
    flops_per_step = flops_per_forward * 3
    
    # Total steps
    total_steps = dataset_size * epochs
    
    # Total FLOPs
    total_flops = flops_per_step * total_steps
    
    return total_flops, flops_per_forward, flops_per_step

def analyze_scaling_laws(param_counts, compute_counts, performances):
    """Analyze if our scaling follows expected laws"""
    print("SCALING LAW ANALYSIS")
    print("=" * 50)
    
    base_params = param_counts[0]
    base_compute = compute_counts[0]
    
    print(f"{'Model':<15} {'Params':<10} {'Ratio':<8} {'Compute':<12} {'Ratio':<8} {'Performance':<12}")
    print("-" * 75)
    
    for i, (params, compute, perf) in enumerate(zip(param_counts, compute_counts, performances)):
        param_ratio = params / base_params
        compute_ratio = compute / base_compute
        print(f"Model {i+1:<8} {params:<10,} {param_ratio:<8.1f} {compute:<12.2e} {compute_ratio:<8.1f} {perf:<12.2e}")
    
    print(f"\nScaling Law Expectations:")
    print(f"- Compute should scale as N^2: {[1, 4, 16, 64]}")
    print(f"- Dataset should scale as N^0.73: {[1, 1.7, 2.8, 4.6]}")
    print(f"- Training should scale as N^0.5: {[1, 1.4, 2.0, 2.8]}")

def main():
    print("MODEL PARAMETER AND COMPUTE ANALYSIS")
    print("=" * 60)
    
    # Current models used in experiments
    current_models = [
        ("RNN-32", "rnn", {"hidden_size": 32}),
        ("RNN-64", "rnn", {"hidden_size": 64}),
        ("RNN-128", "rnn", {"hidden_size": 128}),
        ("Transformer-64", "transformer", {"d_model": 64, "nhead": 4, "num_layers": 2}),
        ("MLP-64", "mlp", {"input_size": 8, "hidden_size": 64}),
    ]
    
    # Proposed larger models
    proposed_models = [
        ("RNN-256", "rnn", {"hidden_size": 256}),
        ("RNN-512", "rnn", {"hidden_size": 512}),
        ("RNN-1024", "rnn", {"hidden_size": 1024}),
        ("RNN-2048", "rnn", {"hidden_size": 2048}),
    ]
    
    print("\n1. CURRENT MODELS")
    print("-" * 40)
    print(f"{'Model':<15} {'Parameters':<12} {'Breakdown':<30}")
    print("-" * 60)
    
    for name, model_type, kwargs in current_models:
        if model_type == "rnn":
            total, gru, linear = calculate_rnn_total_params(kwargs['hidden_size'])
            print(f"{name:<15} {total:<12,} GRU:{gru:,} + Linear:{linear:,}")
        elif model_type == "transformer":
            total, inp, pos, layers, out = calculate_transformer_params(
                kwargs['d_model'], kwargs['nhead'], kwargs['num_layers']
            )
            print(f"{name:<15} {total:<12,} Input:{inp:,} + Pos:{pos:,} + Layers:{layers:,} + Out:{out:,}")
        elif model_type == "mlp":
            total, l1, l2, l3 = calculate_mlp_params(kwargs['input_size'], kwargs['hidden_size'])
            print(f"{name:<15} {total:<12,} L1:{l1:,} + L2:{l2:,} + L3:{l3:,}")
    
    print("\n2. PROPOSED LARGER MODELS")
    print("-" * 40)
    print(f"{'Model':<15} {'Parameters':<12} {'Ratio vs RNN-128':<15}")
    print("-" * 45)
    
    base_params = calculate_rnn_total_params(128)[0]
    
    for name, model_type, kwargs in proposed_models:
        if model_type == "rnn":
            total, gru, linear = calculate_rnn_total_params(kwargs['hidden_size'])
            ratio = total / base_params
            print(f"{name:<15} {total:<12,} {ratio:<15.1f}x")
    
    print("\n3. COMPUTE ANALYSIS (Current Training)")
    print("-" * 50)
    
    # Current training setup
    current_dataset = 200
    current_epochs = 500
    
    print(f"Dataset size: {current_dataset}")
    print(f"Epochs: {current_epochs}")
    print()
    print(f"{'Model':<15} {'FLOPs/Forward':<15} {'Total FLOPs':<15} {'Ratio':<8}")
    print("-" * 55)
    
    base_compute = None
    for name, model_type, kwargs in current_models[:3]:  # Just RNN models
        total_flops, flops_forward, flops_step = calculate_total_compute(
            model_type, current_dataset, current_epochs, **kwargs
        )
        
        if base_compute is None:
            base_compute = total_flops
        
        ratio = total_flops / base_compute
        print(f"{name:<15} {flops_forward:<15.2e} {total_flops:<15.2e} {ratio:<8.1f}x")
    
    print("\n4. OPTIMAL SCALING CALCULATIONS")
    print("-" * 40)
    
    # Calculate optimal dataset and epoch scaling
    hidden_sizes = [128, 256, 512, 1024, 2048]
    base_hidden = 128
    
    print(f"{'Model':<10} {'Params':<8} {'Dataset':<8} {'Epochs':<8} {'Total FLOPs':<12} {'Ratio':<8}")
    print("-" * 65)
    
    for hidden_size in hidden_sizes:
        params = calculate_rnn_total_params(hidden_size)[0]
        param_ratio = params / calculate_rnn_total_params(base_hidden)[0]
        
        # Scaling laws
        dataset_scale = param_ratio ** 0.73
        epoch_scale = param_ratio ** 0.5
        
        optimal_dataset = int(current_dataset * dataset_scale)
        optimal_epochs = int(current_epochs * epoch_scale)
        
        total_flops, _, _ = calculate_total_compute(
            "rnn", optimal_dataset, optimal_epochs, hidden_size=hidden_size
        )
        
        flop_ratio = total_flops / calculate_total_compute(
            "rnn", current_dataset, current_epochs, hidden_size=base_hidden
        )[0]
        
        print(f"RNN-{hidden_size:<5} {params:<8,} {optimal_dataset:<8} {optimal_epochs:<8} {total_flops:<12.2e} {flop_ratio:<8.1f}x")
    
    print("\n5. PERFORMANCE PREDICTIONS")
    print("-" * 30)
    
    # Based on current results
    current_results = {
        "RNN-32": {"harmonic": 1.21e-06, "airy": 5.95e-07, "hermite": 1.37e-05, "legendre": 6.83e-05},
        "RNN-64": {"harmonic": 9.93e-07, "airy": 5.47e-07, "hermite": 1.36e-05, "legendre": 1.11e-04},
        "RNN-128": {"harmonic": 1.29e-06, "airy": 2.75e-07, "hermite": 1.26e-05, "legendre": 9.77e-05},
    }
    
    print("Current performance with optimal scaling should improve by:")
    print("- Hermite: ~1.3e-05 → target ~1e-8 (1000x improvement needed)")
    print("- Legendre: ~8e-05 → target ~1e-8 (8000x improvement needed)")
    print()
    print("With 4x, 16x, 64x parameter scaling + optimal data/compute:")
    print("- Expected improvement: 10-1000x (based on scaling laws)")
    print("- Should achieve <1e-8 on all equations")
    
    print("\n6. RESOURCE REQUIREMENTS")
    print("-" * 25)
    
    # Estimate training times
    base_time = 140  # seconds for RNN-128, 500 epochs, 200 samples
    
    for hidden_size in [256, 512, 1024, 2048]:
        params = calculate_rnn_total_params(hidden_size)[0]
        param_ratio = params / calculate_rnn_total_params(128)[0]
        
        dataset_scale = param_ratio ** 0.73
        epoch_scale = param_ratio ** 0.5
        
        # Time scales roughly with compute
        compute_scale = param_ratio * dataset_scale * epoch_scale
        estimated_time = base_time * compute_scale
        
        print(f"RNN-{hidden_size}: ~{estimated_time/60:.1f} minutes per equation")
    
    print(f"\nTotal estimated time for all 4 equations with largest model (RNN-2048):")
    largest_time = base_time * (calculate_rnn_total_params(2048)[0] / calculate_rnn_total_params(128)[0]) * (2048/128)**0.73 * (2048/128)**0.5
    print(f"~{largest_time * 4 / 3600:.1f} hours")

if __name__ == "__main__":
    main()