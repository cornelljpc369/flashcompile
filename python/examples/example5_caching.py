#!/usr/bin/env python3
"""
Example 5: Demonstrate compilation caching
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import time
import flashcompile as fc

np.random.seed(42)

print("=" * 70)
print("  Compilation Caching Demo")
print("=" * 70)
print()

# Create test data
A = np.random.uniform(-1, 1, (64, 64)).astype(np.float32)
B = np.random.uniform(-1, 1, (64, 64)).astype(np.float32)

# Clear cache to start fresh
fc.get_cache().clear()

print("Test 1: First call (cache miss - compiles)")
print("-" * 70)
start = time.perf_counter()
result1 = fc.matmul(A, B, use_cache=True, verbose=True)
first_time = (time.perf_counter() - start) * 1000
print(f"Time: {first_time:.2f}ms")
print()

print("Test 2: Second call (cache hit - no compilation)")
print("-" * 70)
start = time.perf_counter()
result2 = fc.matmul(A, B, use_cache=True, verbose=True)
second_time = (time.perf_counter() - start) * 1000
print(f"Time: {second_time:.2f}ms")
print()

print("Test 3: Many calls (all cache hits)")
print("-" * 70)
num_calls = 100
total_time = 0

for i in range(num_calls):
    start = time.perf_counter()
    result = fc.matmul(A, B, use_cache=True, verbose=False)
    total_time += (time.perf_counter() - start) * 1000

avg_time = total_time / num_calls
print(f"Average time over {num_calls} calls: {avg_time:.2f}ms")
print()

print("Test 4: Without cache (recompiles every time)")
print("-" * 70)
no_cache_times = []

for i in range(10):
    start = time.perf_counter()
    result = fc.matmul(A, B, use_cache=False, verbose=False)
    no_cache_times.append((time.perf_counter() - start) * 1000)

avg_no_cache = sum(no_cache_times) / len(no_cache_times)
print(f"Average time without cache (10 calls): {avg_no_cache:.2f}ms")
print()

# Summary
print("=" * 70)
print("  Summary")
print("=" * 70)
print(f"First call (compile):     {first_time:.2f}ms")
print(f"Second call (cached):     {second_time:.2f}ms")
print(f"Average (100 cached):     {avg_time:.2f}ms")
print(f"Average (no cache):       {avg_no_cache:.2f}ms")
print()
print(f"Speedup with caching:     {avg_no_cache / avg_time:.1f}x faster!")
print()

# Verify correctness
print("Correctness check:")
if np.allclose(result1, result2):
    print("  ✓ Cached result matches initial result")
else:
    print("  ✗ Results don't match!")

# Cache info
print()
fc.get_cache().info()