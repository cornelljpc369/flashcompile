"""
Benchmarking utilities for Flash compiler
"""

import numpy as np
import time
from typing import List, Tuple, Dict, Callable
from dataclasses import dataclass
import json

@dataclass
class BenchmarkResult:
    """Results from a benchmark run"""
    operation: str
    shape: Tuple[int, ...]
    flash_time: float  # seconds
    numpy_time: float  # seconds
    speedup: float
    num_runs: int
    
    def __repr__(self):
        return (f"BenchmarkResult(op={self.operation}, shape={self.shape}, "
                f"speedup={self.speedup:.2f}x)")

def benchmark_operation(
    flash_fn: Callable,
    numpy_fn: Callable,
    inputs: List[np.ndarray],
    operation_name: str,
    num_runs: int = 100,
    warmup: int = 10
) -> BenchmarkResult:
    """
    Benchmark a single operation
    
    Args:
        flash_fn: Flash compiled function
        numpy_fn: NumPy baseline function
        inputs: Input arrays
        operation_name: Name for reporting
        num_runs: Number of benchmark runs
        warmup: Number of warmup runs
    
    Returns:
        BenchmarkResult with timing comparison
    """
    # Warmup
    for _ in range(warmup):
        _ = flash_fn(*inputs)
        _ = numpy_fn(*inputs)
    
    # Benchmark Flash
    start = time.perf_counter()
    for _ in range(num_runs):
        _ = flash_fn(*inputs)
    flash_time = (time.perf_counter() - start) / num_runs
    
    # Benchmark NumPy
    start = time.perf_counter()
    for _ in range(num_runs):
        _ = numpy_fn(*inputs)
    numpy_time = (time.perf_counter() - start) / num_runs
    
    speedup = numpy_time / flash_time if flash_time > 0 else 0
    
    shape = inputs[0].shape if inputs else ()
    
    return BenchmarkResult(
        operation=operation_name,
        shape=shape,
        flash_time=flash_time,
        numpy_time=numpy_time,
        speedup=speedup,
        num_runs=num_runs
    )

def benchmark_matmul(
    sizes: List[Tuple[int, int, int]],
    num_runs: int = 100,
    warmup: int = 10
) -> List[BenchmarkResult]:
    """
    Benchmark matrix multiplication for various sizes
    
    Args:
        sizes: List of (M, K, N) tuples
        num_runs: Number of runs per size
        warmup: Warmup runs
    
    Returns:
        List of BenchmarkResults
    """
    from .api import matmul
    
    # Set seed for reproducibility
    np.random.seed(42)
    
    results = []
    
    for M, K, N in sizes:
        print(f"Benchmarking MatMul {M}×{K} @ {K}×{N}...", end=' ', flush=True)
        
        # Use uniform distribution to avoid overflow
        A = np.random.uniform(-1.0, 1.0, (M, K)).astype(np.float32)
        B = np.random.uniform(-1.0, 1.0, (K, N)).astype(np.float32)
        
        # Flash: compile + execute each time
        flash_times = []
        for i in range(warmup + num_runs):
            start = time.perf_counter()
            _ = matmul(A, B, execute=True, verbose=False)
            elapsed = time.perf_counter() - start
            if i >= warmup:  # Skip warmup runs
                flash_times.append(elapsed)
        
        flash_time = sum(flash_times) / len(flash_times)
        
        # NumPy: just execution
        for _ in range(warmup):
            _ = A @ B
        
        numpy_times = []
        for _ in range(num_runs):
            start = time.perf_counter()
            _ = A @ B
            numpy_times.append(time.perf_counter() - start)
        
        numpy_time = sum(numpy_times) / len(numpy_times)
        
        speedup = numpy_time / flash_time if flash_time > 0 else 0
        
        result = BenchmarkResult(
            operation=f"matmul_{M}x{K}x{N}",
            shape=(M, K, N),
            flash_time=flash_time,
            numpy_time=numpy_time,
            speedup=speedup,
            num_runs=num_runs
        )
        
        results.append(result)
        
        # Color code speedup
        if speedup >= 1.0:
            color = '\033[92m'  # Green
        elif speedup >= 0.5:
            color = '\033[93m'  # Yellow
        else:
            color = '\033[91m'  # Red
        
        print(f"{color}✓ Speedup: {result.speedup:.2f}x\033[0m")
    
    return results

def benchmark_suite() -> Dict[str, List[BenchmarkResult]]:
    """
    Run comprehensive benchmark suite
    
    Returns:
        Dictionary mapping operation type to results
    """
    print("=" * 70)
    print("  FlashCompile Benchmark Suite")
    print("=" * 70)
    print()
    
    print("\033[93mNOTE: Timings include compilation overhead.\033[0m")
    print("\033[93mIn production, compile once and execute many times!\033[0m")
    print()
    
    # Matrix multiply benchmarks
    print("Matrix Multiplication:")
    print("-" * 70)
    matmul_sizes = [
        (8, 8, 8),      # Tiny (compilation dominates)
        (32, 32, 32),   # Small
        (64, 64, 64),   # Medium
        (128, 128, 128), # Large
    ]
    matmul_results = benchmark_matmul(matmul_sizes, num_runs=10, warmup=3)
    print()
    
    return {
        'matmul': matmul_results
    }

def print_summary(results: Dict[str, List[BenchmarkResult]]):
    """Print benchmark summary"""
    print("=" * 70)
    print("  Benchmark Summary")
    print("=" * 70)
    print()
    
    for op_type, op_results in results.items():
        print(f"{op_type.upper()}:")
        print(f"{'Size':<20} {'Flash (ms)':<15} {'NumPy (ms)':<15} {'Speedup':<10}")
        print("-" * 70)
        
        for result in op_results:
            size_str = 'x'.join(str(d) for d in result.shape)
            flash_ms = result.flash_time * 1000
            numpy_ms = result.numpy_time * 1000
            
            print(f"{size_str:<20} {flash_ms:<15.3f} {numpy_ms:<15.3f} {result.speedup:<10.2f}x")
        
        print()
        
        # Statistics
        speedups = [r.speedup for r in op_results]
        avg_speedup = sum(speedups) / len(speedups)
        max_speedup = max(speedups)
        min_speedup = min(speedups)
        
        print(f"  Average speedup: {avg_speedup:.2f}x")
        print(f"  Max speedup:     {max_speedup:.2f}x")
        print(f"  Min speedup:     {min_speedup:.2f}x")
        print()

def save_results(results: Dict[str, List[BenchmarkResult]], filename: str):
    """Save benchmark results to JSON"""
    data = {}
    for op_type, op_results in results.items():
        data[op_type] = [
            {
                'shape': list(r.shape),
                'flash_time': r.flash_time,
                'numpy_time': r.numpy_time,
                'speedup': r.speedup,
                'num_runs': r.num_runs
            }
            for r in op_results
        ]
    
    with open(filename, 'w') as f:
        json.dump(data, f, indent=2)
    
    print(f"Results saved to {filename}")