"""
High-level functional API for Flash operations
"""

import numpy as np
from .core import (
    MatMul, Add, ReLU,
    generate_module, execute_mlir,
    numpy_to_mlir_const
)

def matmul(A: np.ndarray, B: np.ndarray, 
           execute: bool = True, 
           verbose: bool = False) -> np.ndarray:
    """
    Matrix multiplication: C = A @ B
    
    Args:
        A: Input matrix (M, K)
        B: Input matrix (K, N)
        execute: If True, compile and execute. If False, just return expected result
        verbose: Print compilation details
    
    Returns:
        Result matrix (M, N)
    """
    if not execute:
        # Just compute with NumPy (for comparison)
        return A @ B
    
    # Generate MLIR
    inputs = [("%A", A), ("%B", B)]
    operations = [(MatMul(), ["%A", "%B"], "%C")]
    
    mlir_code = generate_module(inputs, operations, "%C")
    
    if verbose:
        print("Generated MLIR:")
        print(mlir_code)
        print()
    
    # Execute
    exit_code = execute_mlir(mlir_code, verbose=verbose)
    
    if exit_code != 0:
        raise RuntimeError(f"Compilation/execution failed with exit code {exit_code}")
    
    # For now, return NumPy result
    # In full implementation, would extract result from execution
    return A @ B

def add(A: np.ndarray, B: np.ndarray, 
        execute: bool = True, 
        verbose: bool = False) -> np.ndarray:
    """
    Element-wise addition: C = A + B
    """
    if not execute:
        return A + B
    
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
    
    return A + B

def relu(A: np.ndarray, 
         execute: bool = True, 
         verbose: bool = False) -> np.ndarray:
    """
    ReLU activation: B = max(0, A)
    """
    if not execute:
        return np.maximum(0, A)
    
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
    
    return np.maximum(0, A)