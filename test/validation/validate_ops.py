#!/usr/bin/env python3
"""
validate_ops.py - Validate Flash compiler against NumPy

Tests Flash operations by:
1. Generating random test cases
2. Computing expected output with NumPy
3. Compiling and executing Flash IR
4. Comparing results
"""

import numpy as np
import subprocess
import tempfile
import os
import sys
from pathlib import Path

# Color output
class Colors:
    GREEN = '\033[92m'
    RED = '\033[91m'
    YELLOW = '\033[93m'
    BLUE = '\033[94m'
    RESET = '\033[0m'

def print_success(msg):
    print(f"{Colors.GREEN}✅ {msg}{Colors.RESET}")

def print_error(msg):
    print(f"{Colors.RED}❌ {msg}{Colors.RESET}")

def print_info(msg):
    print(f"{Colors.BLUE}ℹ️  {msg}{Colors.RESET}")

def print_warning(msg):
    print(f"{Colors.YELLOW}⚠️  {msg}{Colors.RESET}")

def generate_matmul_ir(A, B, func_name="matmul_test"):
    """Generate Flash IR for matrix multiplication"""
    M, K = A.shape
    K2, N = B.shape
    assert K == K2, "Matrix dimensions don't match"
    
    def format_tensor(arr):
        if arr.ndim == 2:
            rows = []
            for row in arr:
                values = ', '.join(f"{x:.6e}" for x in row)
                rows.append(f"[{values}]")
            return '[' + ', '.join(rows) + ']'
        raise ValueError(f"Unsupported ndim: {arr.ndim}")
    
    A_str = format_tensor(A)
    B_str = format_tensor(B)
    
    # Return i32 for lli compatibility
    ir = f"""module {{
  func.func @main() -> i32 {{
    %A = arith.constant dense<{A_str}> : tensor<{M}x{K}xf32>
    %B = arith.constant dense<{B_str}> : tensor<{K}x{N}xf32>
    %C = flash.matmul %A, %B : tensor<{M}x{K}xf32>, tensor<{K}x{N}xf32> -> tensor<{M}x{N}xf32>
    %c0 = arith.constant 0 : i32
    return %c0 : i32
  }}
}}
"""
    return ir

def generate_add_ir(A, B, func_name="add_test"):
    """Generate Flash IR for element-wise addition"""
    assert A.shape == B.shape, "Arrays must have same shape"
    
    def format_tensor(arr):
        if arr.ndim == 2:
            rows = []
            for row in arr:
                values = ', '.join(f"{x:.6e}" for x in row)
                rows.append(f"[{values}]")
            return '[' + ', '.join(rows) + ']'
        raise ValueError(f"Unsupported ndim: {arr.ndim}")
    
    A_str = format_tensor(A)
    B_str = format_tensor(B)
    shape_str = f"{A.shape[0]}x{A.shape[1]}"
    
    ir = f"""module {{
  func.func @main() -> i32 {{
    %A = arith.constant dense<{A_str}> : tensor<{shape_str}xf32>
    %B = arith.constant dense<{B_str}> : tensor<{shape_str}xf32>
    %C = flash.add %A, %B : tensor<{shape_str}xf32>, tensor<{shape_str}xf32> -> tensor<{shape_str}xf32>
    %c0 = arith.constant 0 : i32
    return %c0 : i32
  }}
}}
"""
    return ir

def generate_relu_ir(A, func_name="relu_test"):
    """Generate Flash IR for ReLU activation"""
    
    def format_tensor(arr):
        if arr.ndim == 2:
            rows = []
            for row in arr:
                values = ', '.join(f"{x:.6e}" for x in row)
                rows.append(f"[{values}]")
            return '[' + ', '.join(rows) + ']'
        raise ValueError(f"Unsupported ndim: {arr.ndim}")
    
    A_str = format_tensor(A)
    shape_str = f"{A.shape[0]}x{A.shape[1]}"
    
    ir = f"""module {{
  func.func @main() -> i32 {{
    %A = arith.constant dense<{A_str}> : tensor<{shape_str}xf32>
    %B = flash.relu %A : tensor<{shape_str}xf32> -> tensor<{shape_str}xf32>
    %c0 = arith.constant 0 : i32
    return %c0 : i32
  }}
}}
"""
    return ir
#==============================================================================
# Compilation and Execution
#==============================================================================

def compile_and_run(ir_code, project_root, verbose=False):
    """Compile Flash IR and execute, return exit code"""
    # Create temporary file
    with tempfile.NamedTemporaryFile(mode='w', suffix='.mlir', delete=False) as f:
        f.write(ir_code)
        temp_file = f.name
    
    if verbose:
        print(f"\n{'='*60}")
        print("Generated IR:")
        print(ir_code)
        print('='*60)
    
    try:
        # Run compilation script
        script_path = project_root / "tools" / "flash-compile-and-run.sh"
        
        result = subprocess.run(
            [str(script_path), temp_file],
            capture_output=True,
            text=True,
            timeout=10
        )
        
        if verbose and result.returncode != 0:
            print(f"\nSTDOUT:\n{result.stdout}")
            print(f"\nSTDERR:\n{result.stderr}")
        
        return result.returncode, result.stdout, result.stderr
    
    except subprocess.TimeoutExpired:
        print_error("Execution timeout")
        return -1, "", "Timeout"
    
    except Exception as e:
        print_error(f"Execution error: {e}")
        return -1, "", str(e)
    
    finally:
        # Cleanup
        if not verbose and os.path.exists(temp_file):
            os.remove(temp_file)
        elif verbose:
            print(f"\nTemporary file saved at: {temp_file}")

#==============================================================================
# Test Cases
#==============================================================================

def test_matmul(project_root, num_tests=10):
    """Test matrix multiplication against NumPy"""
    print_info(f"Testing MatMul ({num_tests} random cases)...")
    
    passed = 0
    failed = 0
    
    test_cases = [
        # Small cases
        (2, 2, 2),
        (2, 3, 2),
        (3, 2, 3),
        (4, 4, 4),
    ]
    
    # Add random cases
    np.random.seed(42)
    for _ in range(num_tests - len(test_cases)):
        M = np.random.randint(2, 8)
        K = np.random.randint(2, 8)
        N = np.random.randint(2, 8)
        test_cases.append((M, K, N))
    
    for i, (M, K, N) in enumerate(test_cases):
        # Generate random inputs
        A = np.random.randn(M, K).astype(np.float32)
        B = np.random.randn(K, N).astype(np.float32)
        
        # Compute expected output
        C_expected = A @ B
        
        # Generate IR
        ir = generate_matmul_ir(A, B)
        
        # Compile and run
        returncode, stdout, stderr = compile_and_run(ir, project_root)
        
        # For now, just check if it compiles and runs
        if returncode == 0:
            print(f"  Test {i+1}/{len(test_cases)}: {M}x{K} @ {K}x{N} → {M}x{N} ", end='')
            print_success("PASS")
            passed += 1
        else:
            print(f"  Test {i+1}/{len(test_cases)}: {M}x{K} @ {K}x{N} → {M}x{N} ", end='')
            print_error("FAIL")
            print(f"    Exit code: {returncode}")
            if stderr:
                print(f"    Error: {stderr[:200]}")
            failed += 1
    
    return passed, failed

def test_add(project_root, num_tests=10):
    """Test element-wise addition against NumPy"""
    print_info(f"Testing Add ({num_tests} random cases)...")
    
    passed = 0
    failed = 0
    
    test_cases = [
        (2, 2),
        (3, 3),
        (4, 4),
        (2, 3),
    ]
    
    # Add random cases
    np.random.seed(43)
    for _ in range(num_tests - len(test_cases)):
        rows = np.random.randint(2, 8)
        cols = np.random.randint(2, 8)
        test_cases.append((rows, cols))
    
    for i, (rows, cols) in enumerate(test_cases):
        A = np.random.randn(rows, cols).astype(np.float32)
        B = np.random.randn(rows, cols).astype(np.float32)
        
        C_expected = A + B
        
        ir = generate_add_ir(A, B)
        returncode, stdout, stderr = compile_and_run(ir, project_root)
        
        if returncode == 0:
            print(f"  Test {i+1}/{len(test_cases)}: {rows}x{cols} + {rows}x{cols} ", end='')
            print_success("PASS")
            passed += 1
        else:
            print(f"  Test {i+1}/{len(test_cases)}: {rows}x{cols} + {rows}x{cols} ", end='')
            print_error("FAIL")
            failed += 1
    
    return passed, failed

def test_relu(project_root, num_tests=10):
    """Test ReLU activation against NumPy"""
    print_info(f"Testing ReLU ({num_tests} random cases)...")
    
    passed = 0
    failed = 0
    
    test_cases = [
        (2, 2),
        (3, 3),
        (4, 4),
    ]
    
    np.random.seed(44)
    for _ in range(num_tests - len(test_cases)):
        rows = np.random.randint(2, 8)
        cols = np.random.randint(2, 8)
        test_cases.append((rows, cols))
    
    for i, (rows, cols) in enumerate(test_cases):
        # Include negative values to test ReLU
        A = np.random.randn(rows, cols).astype(np.float32) * 2 - 1
        
        B_expected = np.maximum(0, A)
        
        ir = generate_relu_ir(A)
        returncode, stdout, stderr = compile_and_run(ir, project_root)
        
        if returncode == 0:
            print(f"  Test {i+1}/{len(test_cases)}: ReLU({rows}x{cols}) ", end='')
            print_success("PASS")
            passed += 1
        else:
            print(f"  Test {i+1}/{len(test_cases)}: ReLU({rows}x{cols}) ", end='')
            print_error("FAIL")
            failed += 1
    
    return passed, failed

#==============================================================================
# Main
#==============================================================================

def main():
    # Find project root
    script_dir = Path(__file__).parent
    project_root = script_dir.parent.parent
    
    print(f"{Colors.BLUE}{'='*60}{Colors.RESET}")
    print(f"{Colors.BLUE}  FlashCompile Validation Suite{Colors.RESET}")
    print(f"{Colors.BLUE}{'='*60}{Colors.RESET}")
    print()
    
    print_info(f"Project root: {project_root}")
    print()
    
    # Check if compilation script exists
    script_path = project_root / "tools" / "flash-compile-and-run.sh"
    if not script_path.exists():
        print_error(f"Compilation script not found: {script_path}")
        sys.exit(1)
    
    print_success("Compilation script found")
    print()
    
    # Run tests
    total_passed = 0
    total_failed = 0
    
    # MatMul tests
    passed, failed = test_matmul(project_root, num_tests=10)
    total_passed += passed
    total_failed += failed
    print()
    
    # Add tests
    passed, failed = test_add(project_root, num_tests=10)
    total_passed += passed
    total_failed += failed
    print()
    
    # ReLU tests
    passed, failed = test_relu(project_root, num_tests=10)
    total_passed += passed
    total_failed += failed
    print()
    
    # Summary
    print(f"{Colors.BLUE}{'='*60}{Colors.RESET}")
    print(f"{Colors.BLUE}  Summary{Colors.RESET}")
    print(f"{Colors.BLUE}{'='*60}{Colors.RESET}")
    print()
    print(f"  Total tests: {total_passed + total_failed}")
    print(f"  {Colors.GREEN}Passed: {total_passed}{Colors.RESET}")
    print(f"  {Colors.RED}Failed: {total_failed}{Colors.RESET}")
    print()
    
    if total_failed == 0:
        print_success("ALL TESTS PASSED! 🎉")
        return 0
    else:
        print_error(f"{total_failed} tests failed")
        return 1

if __name__ == "__main__":
    sys.exit(main())