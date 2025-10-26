#!/usr/bin/env python3
"""
Example 4: Correctness validation
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import flashcompile as fc

# Set seed for reproducibility
np.random.seed(42)

print("=" * 70)
print("  Correctness Validation: Flash vs NumPy")
print("=" * 70)
print()

def test_matmul():
    """Test matrix multiplication correctness"""
    print("Testing MatMul...")
    
    test_cases = [
        (2, 2, 2),
        (3, 4, 5),
        (8, 8, 8),
    ]
    
    for M, K, N in test_cases:
        # Use uniform distribution to avoid numerical issues
        A = np.random.uniform(-1.0, 1.0, (M, K)).astype(np.float32)
        B = np.random.uniform(-1.0, 1.0, (K, N)).astype(np.float32)
        
        # Flash result
        flash_result = fc.matmul(A, B, execute=True, verbose=False)
        
        # NumPy result
        numpy_result = np.matmul(A, B)
        
        # Compare
        if np.allclose(flash_result, numpy_result, rtol=1e-5):
            print(f"  ✓ MatMul {M}×{K}×{N}: PASS")
        else:
            print(f"  ✗ MatMul {M}×{K}×{N}: FAIL")
            print(f"    Max error: {np.abs(flash_result - numpy_result).max()}")

def test_add():
    """Test element-wise addition"""
    print("\nTesting Add...")
    
    shapes = [(2, 2), (4, 4), (8, 8)]
    
    for shape in shapes:
        A = np.random.uniform(-1.0, 1.0, shape).astype(np.float32)
        B = np.random.uniform(-1.0, 1.0, shape).astype(np.float32)
        
        flash_result = fc.add(A, B, execute=True, verbose=False)
        numpy_result = A + B
        
        if np.allclose(flash_result, numpy_result, rtol=1e-5):
            print(f"  ✓ Add {shape}: PASS")
        else:
            print(f"  ✗ Add {shape}: FAIL")

def test_relu():
    """Test ReLU activation"""
    print("\nTesting ReLU...")
    
    shapes = [(2, 2), (4, 4), (8, 8)]
    
    for shape in shapes:
        A = np.random.uniform(-2.0, 2.0, shape).astype(np.float32)
        
        flash_result = fc.relu(A, execute=True, verbose=False)
        numpy_result = np.maximum(0, A)
        
        if np.allclose(flash_result, numpy_result, rtol=1e-5):
            print(f"  ✓ ReLU {shape}: PASS")
        else:
            print(f"  ✗ ReLU {shape}: FAIL")

# Run tests
test_matmul()
test_add()
test_relu()

print()
print("=" * 70)
print("  All correctness tests completed!")
print("=" * 70)