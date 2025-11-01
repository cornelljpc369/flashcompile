// RUN: flash-opt %s --flash-fusion --convert-flash-to-linalg | FileCheck %s

module {
  // Test 1: MatMul + ReLU → linalg.generic
  func.func @test_matmul_relu_lowering() -> tensor<2x2xf32> {
    // CHECK-LABEL: func.func @test_matmul_relu_lowering
    
    %A = arith.constant dense<[[1.0, 2.0], [3.0, 4.0]]> : tensor<2x2xf32>
    %B = arith.constant dense<[[5.0, 6.0], [7.0, 8.0]]> : tensor<2x2xf32>
    
    // Should fuse to flash.matmul_relu
    %mm = flash.matmul %A, %B : tensor<2x2xf32>, tensor<2x2xf32> -> tensor<2x2xf32>
    %result = flash.relu %mm : tensor<2x2xf32> -> tensor<2x2xf32>
    
    // CHECK: linalg.generic
    // CHECK-SAME: iterator_types = ["parallel", "parallel", "reduction"]
    // CHECK: ^bb0(%[[A:.*]]: f32, %[[B:.*]]: f32, %[[C:.*]]: f32):
    // CHECK:   %[[MUL:.*]] = arith.mulf %[[A]], %[[B]]
    // CHECK:   %[[ADD:.*]] = arith.addf %[[C]], %[[MUL]]
    // CHECK:   %[[ZERO:.*]] = arith.constant 0.0
    // CHECK:   %[[RELU:.*]] = arith.maximumf %[[ADD]], %[[ZERO]]
    // CHECK:   linalg.yield %[[RELU]]
    
    return %result : tensor<2x2xf32>
  }
  
  // Test 2: MatMul + Add → linalg.generic with bias
  func.func @test_matmul_add_lowering() -> tensor<2x2xf32> {
    // CHECK-LABEL: func.func @test_matmul_add_lowering
    
    %A = arith.constant dense<[[1.0, 2.0], [3.0, 4.0]]> : tensor<2x2xf32>
    %B = arith.constant dense<[[5.0, 6.0], [7.0, 8.0]]> : tensor<2x2xf32>
    %bias = arith.constant dense<[[0.1, 0.1], [0.1, 0.1]]> : tensor<2x2xf32>
    
    %mm = flash.matmul %A, %B : tensor<2x2xf32>, tensor<2x2xf32> -> tensor<2x2xf32>
    %result = flash.add %mm, %bias : tensor<2x2xf32>, tensor<2x2xf32> -> tensor<2x2xf32>
    
    // CHECK: linalg.copy
    // CHECK: linalg.generic
    // CHECK: arith.mulf
    // CHECK: arith.addf
    
    return %result : tensor<2x2xf32>
  }
  
  // Test 3: Add + ReLU → linalg.generic (element-wise)
  func.func @test_add_relu_lowering() -> tensor<2x2xf32> {
    // CHECK-LABEL: func.func @test_add_relu_lowering
    
    %X = arith.constant dense<[[1.0, -2.0], [-3.0, 4.0]]> : tensor<2x2xf32>
    %Y = arith.constant dense<[[1.0, 1.0], [1.0, 1.0]]> : tensor<2x2xf32>
    
    %add = flash.add %X, %Y : tensor<2x2xf32>, tensor<2x2xf32> -> tensor<2x2xf32>
    %result = flash.relu %add : tensor<2x2xf32> -> tensor<2x2xf32>
    
    // CHECK: linalg.generic
    // CHECK-SAME: iterator_types = ["parallel", "parallel"]
    // CHECK: ^bb0(%[[X:.*]]: f32, %[[Y:.*]]: f32):
    // CHECK:   %[[ADD:.*]] = arith.addf %[[X]], %[[Y]]
    // CHECK:   %[[ZERO:.*]] = arith.constant 0.0
    // CHECK:   %[[RELU:.*]] = arith.maximumf %[[ADD]], %[[ZERO]]
    // CHECK:   linalg.yield %[[RELU]]
    
    return %result : tensor<2x2xf32>
  }
}