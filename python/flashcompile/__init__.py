"""
FlashCompile - Python API for Flash ML Compiler

A production-ready ML compiler with Python interface, providing:
- NumPy integration
- Automatic IR generation
- Compilation and execution
- Performance benchmarking
"""

__version__ = "0.1.0"

from .core import (
    compile,
    MatMul,
    Add,
    ReLU,
    Sequential,
)

from .api import (
    matmul,
    add,
    relu,
)

__all__ = [
    # Core API
    'compile',
    'MatMul',
    'Add', 
    'ReLU',
    'Sequential',
    
    # Functional API
    'matmul',
    'add',
    'relu',
]