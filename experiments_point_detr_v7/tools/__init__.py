"""Tools for benchmarking Point-RT vs V5 vs Traditional Methods."""

import torch
import time
from pathlib import Path

def benchmark_model(model, input_data, num_runs=100, device='cuda'):
    """Benchmark model inference speed.
    
    Args:
        model: PyTorch model
        input_data: Dict with model inputs
        num_runs: Number of inference runs
        device: 'cuda' or 'cpu'
    
    Returns:
        (mean_time_ms, std_time_ms, throughput_images_per_sec)
    """
    model.eval()
    
    # Warmup
    with torch.no_grad():
        _ = model(input_data)
    
    torch.cuda.synchronize() if torch.cuda.is_available() else None
    
    times = []
    with torch.no_grad():
        for _ in range(num_runs):
            start = time.time()
            _ = model(input_data)
            torch.cuda.synchronize() if torch.cuda.is_available() else None
            times.append((time.time() - start) * 1000)  # ms
    
    times = times[10:]  # Skip first 10 warmup runs
    
    mean_time = sum(times) / len(times)
    variance = sum((t - mean_time) ** 2 for t in times) / len(times)
    std_time = variance ** 0.5
    
    batch_size = input_data[list(input_data.keys())[0]].shape[0]
    throughput = (1000 / mean_time) * batch_size
    
    return mean_time, std_time, throughput


if __name__ == "__main__":
    print("Model benchmarking utilities")
