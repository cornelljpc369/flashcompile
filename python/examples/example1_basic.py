#!/usr/bin/env python3
"""
Example 1: Basic usage of Flash compiler Python API
"""

import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import flashcompile as fc

print("=" * 70)
print("  FlashCompile Python API - Basic Example")
print("=" * 70)
print()

# Example 1: Matrix Multiplication
print("Example 1: Matrix Multiplication")
print("-" * 70)

A = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32)
B = np.array([[5.0, 6.0], [7.0, 8.0]], dtype=np.float32)

print(f"A = \n{A}")
print(f"\nB = \n{B}")
print()

# Compile and execute
result = fc.matmul(A, B, execute=True, verbose=True)

print(f"\nC = A @ B = \n{result}")
print()

# Example 2: Element-wise Addition
print("Example 2: Element-wise Addition")
print("-" * 70)

X = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32)
Y = np.array([[0.5, 0.5], [0.5, 0.5]], dtype=np.float32)

result = fc.add(X, Y, execute=True, verbose=False)
print(f"{X} + {Y} = \n{result}")
print()

# Example 3: ReLU
print("Example 3: ReLU Activation")
print("-" * 70)

Z = np.array([[-1.0, 2.0], [-3.0, 4.0]], dtype=np.float32)

result = fc.relu(Z, execute=True, verbose=False)
print(f"ReLU({Z}) = \n{result}")
print()

print("✓ All examples completed successfully!")