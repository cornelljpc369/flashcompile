"""
FlashCompile - Python API for Flash ML Compiler
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

from .cache import get_cache

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
    
    # Cache
    'get_cache',
]