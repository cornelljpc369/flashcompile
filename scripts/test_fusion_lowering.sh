#!/bin/bash
# Visual comparison of lowering with and without fusion

cat > /tmp/test_neural_layer.mlir << 'EOF'
module {
  func.func @neural_network_layer(%input: tensor<128x784xf32>,
                                   %W: tensor<784x256xf32>,
                                   %bias: tensor<128x256xf32>) -> tensor<128x256xf32> {
    // Linear layer with bias and ReLU activation
    %mm = flash.matmul %input, %W : tensor<128x784xf32>, tensor<784x256xf32> -> tensor<128x256xf32>
    %add = flash.add %mm, %bias : tensor<128x256xf32>, tensor<128x256xf32> -> tensor<128x256xf32>
    %result = flash.relu %add : tensor<128x256xf32> -> tensor<128x256xf32>
    
    return %result : tensor<128x256xf32>
  }
}
EOF

echo "=========================================="
echo "  WITHOUT FUSION (3 separate operations)"
echo "=========================================="
.././build/tools/flash-opt/flash-opt /tmp/test_neural_layer.mlir \
  --convert-flash-to-linalg

echo ""
echo ""
echo "=========================================="
echo "  WITH FUSION (optimized!)"
echo "=========================================="
.././build/tools/flash-opt/flash-opt /tmp/test_neural_layer.mlir \
  --flash-fusion \
  --convert-flash-to-linalg

echo ""
echo ""
echo "=========================================="
echo "  COMPARISON"
echo "=========================================="

echo "Without fusion:"
.././build/tools/flash-opt/flash-opt /tmp/test_neural_layer.mlir \
  --convert-flash-to-linalg | grep -c "linalg\."

echo "With fusion:"
.././build/tools/flash-opt/flash-opt /tmp/test_neural_layer.mlir \
  --flash-fusion \
  --convert-flash-to-linalg | grep -c "linalg\."

echo ""
echo "✓ Fewer linalg operations = better performance!"