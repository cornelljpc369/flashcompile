// RUN: flash-opt %s | FileCheck %s

// Typical CNN block: Conv → Add (bias) → ReLU
module {
  // CHECK-LABEL: func @cnn_block
  func.func @cnn_block(%input: tensor<1x3x32x32xf32>,
                       %kernel: tensor<64x3x3x3xf32>,
                       %bias: tensor<1x64x30x30xf32>) -> tensor<1x64x30x30xf32> {
    // Convolution: [1x3x32x32] * [64x3x3x3] = [1x64x30x30]
    // CHECK: flash.conv2d
    %0 = flash.conv2d %input, %kernel : tensor<1x3x32x32xf32>, tensor<64x3x3x3xf32> -> tensor<1x64x30x30xf32>
    
    // Add bias: [1x64x30x30] + [1x64x30x30] = [1x64x30x30]
    // CHECK: flash.add
    %1 = flash.add %0, %bias : tensor<1x64x30x30xf32>, tensor<1x64x30x30xf32> -> tensor<1x64x30x30xf32>
    
    // Activation: relu([1x64x30x30]) = [1x64x30x30]
    // CHECK: flash.relu
    %2 = flash.relu %1 : tensor<1x64x30x30xf32> -> tensor<1x64x30x30xf32>
    
    return %2 : tensor<1x64x30x30xf32>
  }
}