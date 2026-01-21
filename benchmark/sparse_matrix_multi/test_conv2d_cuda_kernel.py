import torch
import torch.nn as nn
import torch.nn.functional as F
import time
import numpy as np
import matplotlib.pyplot as plt
from sparse_tensor_multiply import conv2d_sparse_tensor_multiply


def create_sparse_weights(Cout, Cin, Kh, Kw, sparsity=0.0):
    """
    Create sparse conv2d weights with specified sparsity level.
    
    Args:
        Cout: output channels
        Cin: input channels
        Kh, Kw: kernel dimensions
        sparsity: sparsity level (0.0 = dense, 1.0 = all zeros)
    
    Returns:
        weight tensor with specified sparsity
    """
    weight = torch.randn(Cout, Cin, Kh, Kw)
    if sparsity > 0:
        num_params = Cout * Cin * Kh * Kw
        num_zeros = int(num_params * sparsity)
        # Randomly zero out weights
        flat_weight = weight.flatten()
        indices = torch.randperm(num_params)[:num_zeros]
        flat_weight[indices] = 0
        weight = flat_weight.reshape(Cout, Cin, Kh, Kw)
    return weight


def dense_to_sparse_conv_weights(weight):
    """
    Convert dense conv2d weights to sparse representation.
    weight shape: (Cout, Cin, Kh, Kw)
    
    Returns:
        indices_f: starting index for each output channel
        indices_r: input channel indices
        indices_kw: kernel width indices  
        indices_kh: kernel height indices
        values: non-zero weight values
    """
    Cout, Cin, Kh, Kw = weight.shape
    
    indices_f = []
    indices_r = []
    indices_kw = []
    indices_kh = []
    values = []
    
    current_idx = 0
    for cout in range(Cout):
        indices_f.append(current_idx)
        for cin in range(Cin):
            for kh in range(Kh):
                for kw in range(Kw):
                    val = weight[cout, cin, kh, kw].item()
                    if val != 0:  # Only store non-zero values
                        indices_r.append(cin)
                        indices_kh.append(kh)
                        indices_kw.append(kw)
                        values.append(val)
                        current_idx += 1
    indices_f.append(current_idx)  # End marker
    
    return (
        torch.tensor(indices_f, dtype=torch.int32),
        torch.tensor(indices_r, dtype=torch.int32),
        torch.tensor(indices_kw, dtype=torch.int32),
        torch.tensor(indices_kh, dtype=torch.int32),
        torch.tensor(values, dtype=torch.float32)
    )


def test_conv2d_comparison():
    """Compare sparse conv2d with PyTorch conv2d"""
    
    # Parameters
    N = 2           # batch size
    Cin = 3         # input channels
    Cout = 4        # output channels
    H, W = 32, 32   # input size
    Kh, Kw = 3, 3   # kernel size
    stride = 1
    padding = 0
    
    # Create input
    input_feature_map = torch.randn(N, Cin, H, W).cuda()
    
    # Create PyTorch conv2d layer
    conv = nn.Conv2d(Cin, Cout, (Kh, Kw), stride=stride, padding=padding, bias=False).cuda()
    
    # Get dense weights and convert to sparse
    weight = conv.weight.data  # (Cout, Cin, Kh, Kw)
    indices_f, indices_r, indices_kw, indices_kh, values = dense_to_sparse_conv_weights(weight)
    
    # Move to CUDA
    indices_f = indices_f.cuda()
    indices_r = indices_r.cuda()
    indices_kw = indices_kw.cuda()
    indices_kh = indices_kh.cuda()
    values = values.cuda()
    
    filters_len = len(values)
    
    # Warmup runs
    for _ in range(10):
        _ = conv(input_feature_map)
        _ = conv2d_sparse_tensor_multiply(
            input_feature_map, values, indices_f, indices_r, indices_kw, indices_kh,
            Kw, Kh, filters_len, stride, padding
        )
    torch.cuda.synchronize()
    
    # Benchmark PyTorch conv2d
    num_iterations = 100
    start_event = torch.cuda.Event(enable_timing=True)
    end_event = torch.cuda.Event(enable_timing=True)
    
    start_event.record()
    for _ in range(num_iterations):
        torch_output = conv(input_feature_map)
    end_event.record()
    torch.cuda.synchronize()
    torch_time = start_event.elapsed_time(end_event) / num_iterations  # ms
    
    # Benchmark sparse conv2d
    start_event.record()
    for _ in range(num_iterations):
        sparse_output = conv2d_sparse_tensor_multiply(
            input_feature_map,
            values,
            indices_f,
            indices_r,
            indices_kw,
            indices_kh,
            Kw,
            Kh,
            filters_len,
            stride,
            padding
        )
    end_event.record()
    torch.cuda.synchronize()
    sparse_time = start_event.elapsed_time(end_event) / num_iterations  # ms
    
    # Compare outputs
    print(f"\n{'='*60}")
    print(f"Performance Comparison")
    print(f"{'='*60}")
    print(f"PyTorch output shape: {torch_output.shape}")
    print(f"Sparse output shape: {sparse_output.shape}")
    
    # Check if shapes match
    assert torch_output.shape == sparse_output.shape, f"Shape mismatch: {torch_output.shape} vs {sparse_output.shape}"
    
    # Compute difference
    diff = torch.abs(torch_output - sparse_output)
    max_diff = diff.max().item()
    mean_diff = diff.mean().item()
    
    print(f"\nAccuracy:")
    print(f"  Max absolute difference: {max_diff:.6f}")
    print(f"  Mean absolute difference: {mean_diff:.6f}")
    
    # Check if outputs are close
    is_close = torch.allclose(torch_output, sparse_output, rtol=1e-5, atol=1e-5)
    print(f"  Outputs are close: {is_close}")
    
    # Speed comparison
    print(f"\nSpeed (average over {num_iterations} iterations):")
    print(f"  PyTorch Conv2d:     {torch_time:.4f} ms")
    print(f"  Sparse Conv2d:      {sparse_time:.4f} ms")
    speedup = torch_time / sparse_time if sparse_time > 0 else 0
    slowdown = sparse_time / torch_time if torch_time > 0 else 0
    if speedup > 1:
        print(f"  Speedup:            {speedup:.2f}x faster")
    else:
        print(f"  Slowdown:           {slowdown:.2f}x slower")
    
    # Sparsity info
    total_params = Cout * Cin * Kh * Kw
    sparse_params = filters_len
    sparsity = (1 - sparse_params / total_params) * 100
    print(f"\nSparsity Info:")
    print(f"  Total parameters:   {total_params}")
    print(f"  Sparse parameters:  {sparse_params}")
    print(f"  Sparsity:           {sparsity:.2f}%")
    print(f"{'='*60}\n")
    
    return is_close


def benchmark_single_config(input_feature_map, conv, indices_f, indices_r, indices_kw, indices_kh, 
                           values, Kw, Kh, filters_len, stride, padding, num_iterations=100):
    """Benchmark a single configuration and return times"""
    # Warmup
    for _ in range(10):
        _ = conv(input_feature_map)
        _ = conv2d_sparse_tensor_multiply(
            input_feature_map, values, indices_f, indices_r, indices_kw, indices_kh,
            Kw, Kh, filters_len, stride, padding
        )
    torch.cuda.synchronize()
    
    # Benchmark PyTorch
    start_event = torch.cuda.Event(enable_timing=True)
    end_event = torch.cuda.Event(enable_timing=True)
    start_event.record()
    for _ in range(num_iterations):
        _ = conv(input_feature_map)
    end_event.record()
    torch.cuda.synchronize()
    torch_time = start_event.elapsed_time(end_event) / num_iterations
    
    # Benchmark Sparse
    start_event.record()
    for _ in range(num_iterations):
        _ = conv2d_sparse_tensor_multiply(
            input_feature_map, values, indices_f, indices_r, indices_kw, indices_kh,
            Kw, Kh, filters_len, stride, padding
        )
    end_event.record()
    torch.cuda.synchronize()
    sparse_time = start_event.elapsed_time(end_event) / num_iterations
    
    return torch_time, sparse_time


def test_sparsity_sweep():
    """Test speedup across different sparsity levels"""
    # Parameters
    N = 8
    Cin = 64
    Cout = 128
    H, W = 512, 512
    Kh, Kw = 3, 3
    stride = 1
    padding = 0
    
    sparsity_levels = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.95, 0.99]
    torch_times = []
    sparse_times = []
    speedups = []
    
    print("Testing across sparsity levels...")
    for sparsity in sparsity_levels:
        # Create weights with specified sparsity
        weight = create_sparse_weights(Cout, Cin, Kh, Kw, sparsity).cuda()
        
        # Create conv layer
        conv = nn.Conv2d(Cin, Cout, (Kh, Kw), stride=stride, padding=padding, bias=False).cuda()
        conv.weight.data = weight
        
        # Convert to sparse
        indices_f, indices_r, indices_kw, indices_kh, values = dense_to_sparse_conv_weights(weight)
        indices_f = indices_f.cuda()
        indices_r = indices_r.cuda()
        indices_kw = indices_kw.cuda()
        indices_kh = indices_kh.cuda()
        values = values.cuda()
        filters_len = len(values)
        
        # Create input
        input_feature_map = torch.randn(N, Cin, H, W).cuda()
        
        # Benchmark
        torch_time, sparse_time = benchmark_single_config(
            input_feature_map, conv, indices_f, indices_r, indices_kw, indices_kh,
            values, Kw, Kh, filters_len, stride, padding
        )
        
        speedup = torch_time / sparse_time if sparse_time > 0 else 0
        torch_times.append(torch_time)
        sparse_times.append(sparse_time)
        speedups.append(speedup)
        
        print(f"Sparsity {sparsity:.2f}: PyTorch={torch_time:.4f}ms, Sparse={sparse_time:.4f}ms, Speedup={speedup:.2f}x")
    
    # Plot
    plt.figure(figsize=(10, 6))
    plt.plot([s*100 for s in sparsity_levels], torch_times, 'r-o', linewidth=2, markersize=8, label='PyTorch Dense')
    plt.plot([s*100 for s in sparsity_levels], sparse_times, 'b-o', linewidth=2, markersize=8, label='Sparse')
    plt.xlabel('Sparsity (%)', fontsize=12)
    plt.ylabel('Execution Time (ms)', fontsize=12)
    plt.title('Execution Time vs Sparsity Level', fontsize=14, fontweight='bold')
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig('time_vs_sparsity.png', dpi=300, bbox_inches='tight')
    print(f"\nGraph saved to time_vs_sparsity.png")
    
    return sparsity_levels, speedups


def test_batch_size_sweep(sparsity=0.9):
    """Test speedup across different batch sizes"""
    # Parameters
    Cin = 64
    Cout = 128
    H, W = 32, 32
    Kh, Kw = 3, 3
    stride = 1
    padding = 0
    
    batch_sizes = [1, 2, 4, 8, 16, 32, 64, 128, 132]
    torch_times = []
    sparse_times = []
    speedups = []
    
    print(f"\nTesting across batch sizes (sparsity={sparsity:.2f})...")
    for N in batch_sizes:
        # Create weights with specified sparsity
        weight = create_sparse_weights(Cout, Cin, Kh, Kw, sparsity).cuda()
        
        # Create conv layer
        conv = nn.Conv2d(Cin, Cout, (Kh, Kw), stride=stride, padding=padding, bias=False).cuda()
        conv.weight.data = weight
        
        # Convert to sparse
        indices_f, indices_r, indices_kw, indices_kh, values = dense_to_sparse_conv_weights(weight)
        indices_f = indices_f.cuda()
        indices_r = indices_r.cuda()
        indices_kw = indices_kw.cuda()
        indices_kh = indices_kh.cuda()
        values = values.cuda()
        filters_len = len(values)
        
        # Create input
        input_feature_map = torch.randn(N, Cin, H, W).cuda()
        
        # Benchmark
        torch_time, sparse_time = benchmark_single_config(
            input_feature_map, conv, indices_f, indices_r, indices_kw, indices_kh,
            values, Kw, Kh, filters_len, stride, padding
        )
        
        speedup = torch_time / sparse_time if sparse_time > 0 else 0
        torch_times.append(torch_time)
        sparse_times.append(sparse_time)
        speedups.append(speedup)
        
        print(f"Batch {N:3d}: PyTorch={torch_time:.4f}ms, Sparse={sparse_time:.4f}ms, Speedup={speedup:.2f}x")
    
    # Plot
    plt.figure(figsize=(10, 6))
    plt.plot(batch_sizes, torch_times, 'r-o', linewidth=2, markersize=8, label='PyTorch Dense')
    plt.plot(batch_sizes, sparse_times, 'b-o', linewidth=2, markersize=8, label='Sparse')
    plt.xlabel('Batch Size', fontsize=12)
    plt.ylabel('Execution Time (ms)', fontsize=12)
    plt.title(f'Execution Time vs Batch Size (Sparsity={sparsity*100:.0f}%)', fontsize=14, fontweight='bold')
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig('time_vs_batch_size.png', dpi=300, bbox_inches='tight')
    print(f"\nGraph saved to time_vs_batch_size.png")
    
    return batch_sizes, speedups


def test_optimal_configs():
    """
    Test configurations optimized for sparse convolution benefits.
    Sparse operations are most beneficial when:
    1. High sparsity (90%+): More zeros to skip
    2. Large channels (64-512): More parameters to skip per output
    3. Moderate spatial dims (32-224): Enough work to amortize overhead
    4. Moderate batch size (8-32): Good parallelism without memory pressure
    5. Larger kernels (5x5, 7x7): More parameters per output pixel
    """
    configs = [
        # (name, N, Cin, Cout, H, W, Kh, Kw, sparsity, description)
        ("Small EfficientNet-like", 8, 32, 64, 112, 112, 3, 3, 0.90, "Early EfficientNet block"),
        ("Medium EfficientNet-like", 8, 80, 112, 56, 56, 3, 3, 0.95, "Middle EfficientNet block"),
        ("Large EfficientNet-like", 8, 192, 320, 28, 28, 3, 3, 0.95, "Later EfficientNet block"),
        ("High Channel", 16, 256, 512, 64, 64, 3, 3, 0.95, "High channel count"),
        ("Large Kernel", 8, 64, 128, 32, 32, 7, 7, 0.90, "Large kernel size"),
        ("Extreme Sparsity", 8, 128, 256, 64, 64, 3, 3, 0.99, "99% sparsity"),
        ("Large Spatial", 4, 64, 128, 224, 224, 3, 3, 0.90, "Large input size"),
    ]
    
    print("\n" + "="*80)
    print("Testing Optimal Configurations for Sparse Convolution Benefits")
    print("="*80)
    
    results = []
    for name, N, Cin, Cout, H, W, Kh, Kw, sparsity, desc in configs:
        print(f"\n{name} ({desc})")
        print(f"  Config: N={N}, Cin={Cin}, Cout={Cout}, H={H}, W={W}, K={Kh}x{Kw}, Sparsity={sparsity*100:.0f}%")
        
        # Create weights with specified sparsity
        weight = create_sparse_weights(Cout, Cin, Kh, Kw, sparsity).cuda()
        
        # Create conv layer
        conv = nn.Conv2d(Cin, Cout, (Kh, Kw), stride=1, padding=1, bias=False).cuda()
        conv.weight.data = weight
        
        # Convert to sparse
        indices_f, indices_r, indices_kw, indices_kh, values = dense_to_sparse_conv_weights(weight)
        indices_f = indices_f.cuda()
        indices_r = indices_r.cuda()
        indices_kw = indices_kw.cuda()
        indices_kh = indices_kh.cuda()
        values = values.cuda()
        filters_len = len(values)
        
        # Create input
        input_feature_map = torch.randn(N, Cin, H, W).cuda()
        
        # Compute output dimensions
        padding = 1
        stride = 1
        H_out = (H + 2 * padding - Kh) // stride + 1
        W_out = (W + 2 * padding - Kw) // stride + 1
        
        # Verify correctness first
        torch.cuda.synchronize()
        torch_output = conv(input_feature_map)
        sparse_output = conv2d_sparse_tensor_multiply(
            input_feature_map, values, indices_f, indices_r, indices_kw, indices_kh,
            Kw, Kh, filters_len, stride, padding
        )
        torch.cuda.synchronize()
        
        is_correct = torch.allclose(torch_output, sparse_output, rtol=1e-4, atol=1e-4)
        
        # Benchmark
        torch_time, sparse_time = benchmark_single_config(
            input_feature_map, conv, indices_f, indices_r, indices_kw, indices_kh,
            values, Kw, Kh, filters_len, stride, padding, num_iterations=50
        )
        
        speedup = torch_time / sparse_time if sparse_time > 0 else 0
        total_params = Cout * Cin * Kh * Kw
        sparse_params = filters_len
        actual_sparsity = (1 - sparse_params / total_params) * 100
        
        print(f"  Correct: {is_correct}")
        print(f"  PyTorch: {torch_time:.4f}ms, Sparse: {sparse_time:.4f}ms, Speedup: {speedup:.2f}x")
        print(f"  Params: {total_params:,} total, {sparse_params:,} sparse ({actual_sparsity:.1f}% sparsity)")
        print(f"  Output: {H_out}x{W_out}, {N} batches, {Cout} channels")
        
        results.append({
            'name': name,
            'speedup': speedup,
            'torch_time': torch_time,
            'sparse_time': sparse_time,
            'is_correct': is_correct,
            'sparsity': actual_sparsity
        })
    
    print("\n" + "="*80)
    print("Summary: Best Configurations for Sparse Benefits")
    print("="*80)
    results.sort(key=lambda x: x['speedup'], reverse=True)
    for i, r in enumerate(results[:3], 1):
        print(f"{i}. {r['name']}: {r['speedup']:.2f}x speedup ({r['sparsity']:.1f}% sparsity)")
    
    return results


if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1:
        if sys.argv[1] == "sparsity":
            test_sparsity_sweep()
        elif sys.argv[1] == "batch":
            sparsity = float(sys.argv[2]) if len(sys.argv) > 2 else 0.9
            test_batch_size_sweep(sparsity)
        elif sys.argv[1] == "both":
            sparsity = float(sys.argv[2]) if len(sys.argv) > 2 else 0.9
            test_sparsity_sweep()
            test_batch_size_sweep(sparsity)
        elif sys.argv[1] == "optimal":
            test_optimal_configs()
        else:
            print("Usage: python test_conv2d_cuda_kernel.py [sparsity|batch|both|optimal] [sparsity_level]")
    else:
        test_conv2d_comparison()


