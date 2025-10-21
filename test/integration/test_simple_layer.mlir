// RUN: %flash-opt %s | FileCheck %s

// Simple neural network layer: relu(matmul(input, weights) + bias)
module {
  // CHECK-LABEL: func @simple_layer
  func.func @simple_layer(%input: tensor<8x16xf32>, 
                          %weights: tensor<16x32xf32>,
                          %bias: tensor<8x32xf32>) -> tensor<8x32xf32> {
    // Matrix multiplication: [8x16] * [16x32] = [8x32]
    // CHECK: flash.matmul
    %0 = flash.matmul %input, %weights : tensor<8x16xf32>, tensor<16x32xf32> -> tensor<8x32xf32>
    
    // Add bias: [8x32] + [8x32] = [8x32]
    // CHECK: flash.add
    %1 = flash.add %0, %bias : tensor<8x32xf32>, tensor<8x32xf32> -> tensor<8x32xf32>
    
    // Activation: relu([8x32]) = [8x32]
    // CHECK: flash.relu
    %2 = flash.relu %1 : tensor<8x32xf32> -> tensor<8x32xf32>
    
    return %2 : tensor<8x32xf32>
  }
}