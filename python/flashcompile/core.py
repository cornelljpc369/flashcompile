"""
Core module: IR generation and compilation
"""

import numpy as np
import subprocess
import tempfile
import os
from pathlib import Path
from typing import List, Union, Tuple, Optional
from dataclasses import dataclass

#==============================================================================
# Configuration
#==============================================================================

# Find project root
SCRIPT_DIR = Path(__file__).parent
PROJECT_ROOT = SCRIPT_DIR.parent.parent

FLASH_OPT = PROJECT_ROOT / "build" / "tools" / "flash-opt" / "flash-opt"
COMPILE_SCRIPT = PROJECT_ROOT / "tools" / "flash-compile-and-run.sh"

#==============================================================================
# IR Generation Utilities
#==============================================================================

def numpy_to_mlir_const(arr: np.ndarray) -> str:
    """
    Convert NumPy array to MLIR dense constant format
    
    Args:
        arr: NumPy array (must be float32)
    
    Returns:
        MLIR dense constant string
    """

    if arr.dtype != np.float32:
        raise ValueError(f"Array must be float32, got {arr.dtype}")
    
    def format_array(a):
        if a.ndim == 0:  
            return f"{float(a):.6e}"
        elif a.ndim == 1:
            values = ", ".join(f"{float(x):.6e}" for x in a)
            return f"[{values}]"
        else:
            rows = [format_array(row) for row in a]
            return '[' + ', '.join(rows) + ']'
        
    return format_array(arr)

def shape_to_mlir(shape: Tuple[int, ...]) -> str:
    """Convert shape tuple to MLIR shape string"""
    return 'x'.join(str(d) for d in shape)

#==============================================================================
# Operation Base Classes
#==============================================================================

@dataclass
class Tensor:
    """Represents a tensor in the computation graph"""
    name: str
    shape: Tuple[int, ...]
    dtype: str = "f32"
    
    def mlir_type(self) -> str:
        return f"tensor<{shape_to_mlir(self.shape)}x{self.dtype}>"
    
class Operation:
    """Base class for all operations"""
    
    def __init__(self):
        self.output_tensor = None
    
    def forward(self, *inputs):
        """Execute operation (for eager mode)"""
        raise NotImplementedError
    
    def to_mlir(self, input_names: List[str], output_name: str) -> str:
        """Generate MLIR code for this operation"""
        raise NotImplementedError

#==============================================================================
# Concrete Operations
#==============================================================================

class MatMul(Operation):
    """Matrix multiplication operation"""
    def __init__(self, input_dim: Optional[int] = None, output_dim: Optional[int] = None):
        super().__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
    
    def forward(self, A: np.ndarray, B: np.ndarray) -> np.ndarray:
        """Eager execution: A @ B"""
        return A @ B
    
    def to_mlir(self, input_names: List[str], output_name: str, 
                input_shapes: List[Tuple]) -> str:
        """Generate Flash matmul op"""
        A_name, B_name = input_names
        M, K = input_shapes[0]
        K2, N = input_shapes[1]
        
        assert K == K2, f"Matmul dimension mismatch: {K} != {K2}"
        
        return (
            f"    {output_name} = flash.matmul {A_name}, {B_name} : "
            f"tensor<{M}x{K}xf32>, tensor<{K}x{N}xf32> -> tensor<{M}x{N}xf32>"
        )

class Add(Operation):
    def forward(self, A:np.ndarray, B: np.ndarray) -> np.ndarray:
        """Eager execution: A + B"""
        return A + B
    
    def to_mlir(self, input_names: List[str], output_name: str, input_shapes: List[Tuple]) -> str:
        A_name, B_name = input_names
        shape = input_shapes[0]
        shape_str = shape_to_mlir(shape)

        return (
            f"    {output_name} = flash.add {A_name}, {B_name} : "
            f"tensor<{shape_str}xf32>, tensor<{shape_str}xf32> -> tensor<{shape_str}xf32>"
        )

class ReLU(Operation):
    """ReLU activation"""
    
    def forward(self, A: np.ndarray) -> np.ndarray:
        return np.maximum(0, A)
    
    def to_mlir(self, input_names: List[str], output_name: str,
                input_shapes: List[Tuple]) -> str:
        A_name = input_names[0]
        shape = input_shapes[0]
        shape_str = shape_to_mlir(shape)
        
        return (
            f"    {output_name} = flash.relu {A_name} : "
            f"tensor<{shape_str}xf32> -> tensor<{shape_str}xf32>"
        )
    

#==============================================================================
# Model Container
#==============================================================================

class Sequential:
    """Sequential model container"""
    
    def __init__(self, layers: List[Operation]):
        self.layers = layers
    
    def forward(self, x: np.ndarray) -> np.ndarray:
        """Eager execution through all layers"""
        for layer in self.layers:
            if isinstance(layer, MatMul):
                # For MatMul in Sequential, assume weights are stored
                if not hasattr(layer, 'weights'):
                    raise ValueError("MatMul in Sequential needs weights")
                x = layer.forward(x, layer.weights)
            else:
                x = layer.forward(x)
        return x
    
    def __call__(self, x: np.ndarray) -> np.ndarray:
        return self.forward(x)
    
#==============================================================================
# Compilation
#==============================================================================

class CompiledModel:
    """Compiled Flash model ready for execution"""
    
    def __init__(self, mlir_code: str, execute_fn):
        self.mlir_code = mlir_code
        self.execute_fn = execute_fn
    
    def __call__(self, *args):
        return self.execute_fn(*args)
    
    def save_mlir(self, path: str):
        """Save MLIR code to file"""
        with open(path, 'w') as f:
            f.write(self.mlir_code)


def compile(model_or_fn, optimization_level: int = 2, verbose: bool = False):
    """
    Compile a model or function to Flash IR
    
    Args:
        model_or_fn: Model or function to compile
        optimization_level: 0-3 (higher = more optimization)
        verbose: Print compilation details
    
    Returns:
        CompiledModel ready for execution
    """
    # For now, we'll implement direct function compilation
    # Model compilation would be similar but more complex
    
    if verbose:
        print(f"Compiling with optimization level {optimization_level}...")
    
    # This is a simplified version - full implementation would:
    # 1. Trace the model/function
    # 2. Generate Flash IR
    # 3. Run optimization passes
    # 4. Return executable
    
    return model_or_fn  # Placeholder

#==============================================================================
# Module Generation
#==============================================================================

def generate_module(inputs: List[Tuple[str, np.ndarray]], 
                   operations: List[Tuple[Operation, List[str], str]],
                   output_var: str) -> str:
    """
    Generate complete MLIR module
    
    Args:
        inputs: List of (name, numpy_array) for inputs
        operations: List of (op, input_names, output_name)
        output_var: Final output variable name
    
    Returns:
        Complete MLIR module string
    """
    lines = ["module {", "  func.func @main() -> i32 {"]
    
    # Generate input constants
    for name, arr in inputs:
        mlir_const = numpy_to_mlir_const(arr)
        shape_str = shape_to_mlir(arr.shape)
        lines.append(f"    {name} = arith.constant dense<{mlir_const}> : tensor<{shape_str}xf32>")
    
    # Generate operations
    for op, input_names, output_name in operations:
        input_shapes = []
        for inp_name in input_names:
            # Find shape from inputs
            for name, arr in inputs:
                if name == inp_name:
                    input_shapes.append(arr.shape)
                    break
        
        mlir_line = op.to_mlir(input_names, output_name, input_shapes)
        lines.append(mlir_line)
    
    # Return success
    lines.append("    %c0 = arith.constant 0 : i32")
    lines.append("    return %c0 : i32")
    lines.append("  }")
    lines.append("}")
    
    return '\n'.join(lines)

#==============================================================================
# Execution
#==============================================================================

def execute_mlir(mlir_code: str, verbose: bool = False) -> int:
    """
    Execute MLIR code using flash-compile-and-run.sh
    
    Returns:
        Exit code (0 = success)
    """
    # Write to temp file
    with tempfile.NamedTemporaryFile(mode='w', suffix='.mlir', delete=False) as f:
        f.write(mlir_code)
        temp_file = f.name
    
    try:
        result = subprocess.run(
            [str(COMPILE_SCRIPT), temp_file],
            capture_output=True,
            text=True,
            timeout=30
        )
        
        if verbose:
            print(result.stdout)
            if result.stderr:
                print(result.stderr)
        
        return result.returncode
    
    finally:
        if os.path.exists(temp_file):
            os.remove(temp_file)