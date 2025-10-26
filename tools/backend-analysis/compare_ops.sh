#!/bin/bash

echo "Comparing Backend Analysis: MatMul vs Add vs ReLU"
echo "=================================================="
echo ""

echo "1. MatMul (2x3 @ 3x2):"
./tools/backend_analysis/analyze_backend.py /tmp/matmul_backend_test.mlir | grep -E "(Total instructions|Float operations|Vector operations|register)"

echo ""
echo "2. Add (2x2 + 2x2):"
./tools/backend_analysis/analyze_backend.py /tmp/add_test.mlir | grep -E "(Total instructions|Float operations|Vector operations|register)"

echo ""
echo "3. ReLU (2x2):"
./tools/backend_analysis/analyze_backend.py /tmp/relu_test.mlir | grep -E "(Total instructions|Float operations|Vector operations|register)"