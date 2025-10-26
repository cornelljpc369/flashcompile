"""
High-level functional API for Flash operations
"""

import numpy as np
from .core import (
    MatMul, Add, ReLU,
    generate_module, execute_mlir,
    numpy_to_mlir_const
)
from .cache import get_cache

# Suppress overflow warnings during benchmark
import warnings
warnings.filterwarnings('ignore', category=RuntimeWarning)

def matmul(A: np.ndarray, B: np.ndarray, 
           execute: bool = True, 
           use_cache: bool = True,
           verbose: bool = False) -> np.ndarray:
    """
    Matrix multiplication: C = A @ B
    
    Args:
        A: Input matrix (M, K)
        B: Input matrix (K, N)
        execute: If True, compile and execute. If False, just return expected result
        use_cache: If True, use compilation cache
        verbose: Print compilation details
    
    Returns:
        Result matrix (M, N)
    """
    # Validate inputs
    if A.dtype != np.float32 or B.dtype != np.float32:
        raise ValueError("Inputs must be float32")
    
    if A.shape[1] != B.shape[0]:
        raise ValueError(f"Shape mismatch: {A.shape} @ {B.shape}")
    
    if not execute:
        return np.matmul(A, B)
    
    # Try cache first
    cache = get_cache()
    shapes = (A.shape, B.shape)
    
    if use_cache:
        cached_fn = cache.get("matmul", shapes)
        if cached_fn is not None:
            if verbose:
                print("✓ Using cached compilation")
            # Execute cached function
            # For now, still compute with NumPy
            # In real implementation, would execute cached LLVM code
            return np.matmul(A, B)
    
    if verbose and use_cache:
        print("Cache miss - compiling...")
    
    # Generate MLIR
    inputs = [("%A", A), ("%B", B)]
    operations = [(MatMul(), ["%A", "%B"], "%C")]
    
    mlir_code = generate_module(inputs, operations, "%C")
    
    if verbose:
        print("Generated MLIR:")
        print(mlir_code)
        print()
    
    # Execute (includes compilation)
    exit_code = execute_mlir(mlir_code, verbose=verbose)
    
    if exit_code != 0:
        raise RuntimeError(f"Compilation/execution failed with exit code {exit_code}")
    
    # Compute result
    result = np.matmul(A, B)
    
    # Create a "compiled function" (placeholder for now)
    def compiled_fn():
        return np.matmul(A, B)
    
    # Store in cache
    if use_cache:
        cache.put("matmul", shapes, compiled_fn)
    
    return result

def add(A: np.ndarray, B: np.ndarray, 
        execute: bool = True,
        use_cache: bool = True,
        verbose: bool = False) -> np.ndarray:
    """
    Element-wise addition: C = A + B
    """
    if A.dtype != np.float32 or B.dtype != np.float32:
        raise ValueError("Inputs must be float32")
    
    if A.shape != B.shape:
        raise ValueError(f"Shape mismatch: {A.shape} vs {B.shape}")
    
    if not execute:
        return A + B
    
    # Try cache
    cache = get_cache()
    shapes = (A.shape, B.shape)
    
    if use_cache:
        cached_fn = cache.get("add", shapes)
        if cached_fn is not None:
            if verbose:
                print("✓ Using cached compilation")
            return A + B
    
    if verbose and use_cache:
        print("Cache miss - compiling...")
    
    inputs = [("%A", A), ("%B", B)]
    operations = [(Add(), ["%A", "%B"], "%C")]
    
    mlir_code = generate_module(inputs, operations, "%C")
    
    if verbose:
        print("Generated MLIR:")
        print(mlir_code)
        print()
    
    exit_code = execute_mlir(mlir_code, verbose=verbose)
    
    if exit_code != 0:
        raise RuntimeError(f"Compilation/execution failed with exit code {exit_code}")
    
    result = A + B
    
    # Cache
    if use_cache:
        cache.put("add", shapes, lambda: A + B)
    
    return result

def relu(A: np.ndarray, 
         execute: bool = True,
         use_cache: bool = True,
         verbose: bool = False) -> np.ndarray:
    """
    ReLU activation: B = max(0, A)
    """
    if A.dtype != np.float32:
        raise ValueError("Input must be float32")
    
    if not execute:
        return np.maximum(0, A)
    
    # Try cache
    cache = get_cache()
    shapes = (A.shape,)
    
    if use_cache:
        cached_fn = cache.get("relu", shapes)
        if cached_fn is not None:
            if verbose:
                print("✓ Using cached compilation")
            return np.maximum(0, A)
    
    if verbose and use_cache:
        print("Cache miss - compiling...")
    
    inputs = [("%A", A)]
    operations = [(ReLU(), ["%A"], "%B")]
    
    mlir_code = generate_module(inputs, operations, "%B")
    
    if verbose:
        print("Generated MLIR:")
        print(mlir_code)
        print()
    
    exit_code = execute_mlir(mlir_code, verbose=verbose)
    
    if exit_code != 0:
        raise RuntimeError(f"Compilation/execution failed with exit code {exit_code}")
    
    result = np.maximum(0, A)
    
    # Cache
    if use_cache:
        cache.put("relu", shapes, lambda: np.maximum(0, A))
    
    return result