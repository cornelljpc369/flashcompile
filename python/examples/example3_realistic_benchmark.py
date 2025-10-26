#!/usr/bin/env python3
"""
Example 3: Realistic benchmarking (compile once, run many times)
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import time
import flashcompile as fc

# Set seed for reproducibility and avoid numerical issues
np.random.seed(42)

print("=" * 70)
print("  Realistic Benchmark: Compile Once, Execute Many Times")
print("=" * 70)
print()

sizes = [
    (32, 32, 32),
    (64, 64, 64),
    (128, 128, 128),
    (256, 256, 256),
]

print(f"{'Size':<15} {'Compile (ms)':<15} {'Execute (ms)':<15} {'NumPy (ms)':<15} {'Speedup':<10}")
print("-" * 70)

for M, K, N in sizes:
    # Use smaller random values to avoid overflow
    A = np.random.uniform(-1.0, 1.0, (M, K)).astype(np.float32)
    B = np.random.uniform(-1.0, 1.0, (K, N)).astype(np.float32)
    
    # Measure compilation time (first call)
    start = time.perf_counter()
    _ = fc.matmul(A, B, execute=True, verbose=False)
    compile_time = (time.perf_counter() - start) * 1000  # ms
    
    # Measure execution time (subsequent calls - includes recompilation!)
    num_runs = 10
    exec_times = []
    for _ in range(num_runs):
        start = time.perf_counter()
        _ = fc.matmul(A, B, execute=True, verbose=False)
        exec_times.append((time.perf_counter() - start) * 1000)
    
    avg_exec = sum(exec_times) / len(exec_times)
    
    # Measure NumPy baseline
    numpy_times = []
    for _ in range(num_runs):
        start = time.perf_counter()
        _ = A @ B
        numpy_times.append((time.perf_counter() - start) * 1000)
    
    avg_numpy = sum(numpy_times) / len(numpy_times)
    
    # Calculate speedup (note: includes compilation overhead!)
    speedup = avg_numpy / avg_exec if avg_exec > 0 else 0
    
    size_str = f"{M}×{K}×{N}"
    print(f"{size_str:<15} {compile_time:<15.2f} {avg_exec:<15.2f} {avg_numpy:<15.2f} {speedup:<10.2f}x")

print()
print("INTERPRETATION:")
print("- Compile time: First call (includes all passes)")
print("- Execute time: Average of 10 runs (currently includes recompilation)")
print("- In production: Compile once, execute 1000s of times → much faster!")
print()
print("CURRENT LIMITATION:")
print("- We recompile on every call (not caching compiled code)")
print("- This is why execution time ≈ compilation time")
print("- Real speedup would be much higher with proper caching!")