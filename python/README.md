# FlashCompile Python API

Python interface for the Flash ML Compiler.

## Installation
```bash
# From the python/ directory
pip install -e .
```

## Quick Start
```python
import flashcompile as fc
import numpy as np

# Matrix multiplication
A = np.random.randn(128, 256).astype(np.float32)
B = np.random.randn(256, 64).astype(np.float32)

C = fc.matmul(A, B)  # Compiles and executes!
```

## Examples
```bash
# Basic operations
python examples/example1_basic.py

# Benchmarking
python examples/example2_benchmark.py
```

## API Reference

### Operations

- `fc.matmul(A, B)` - Matrix multiplication
- `fc.add(A, B)` - Element-wise addition
- `fc.relu(A)` - ReLU activation

### Benchmarking

- `fc.benchmark.benchmark_matmul(sizes)` - Benchmark matmul
- `fc.benchmark.benchmark_suite()` - Full benchmark suite