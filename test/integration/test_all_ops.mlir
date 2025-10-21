// RUN: %flash-opt %s | FileCheck %s

// Test all Flash operations individually
module {
  // CHECK-LABEL: func @test_matmul
  func.func @test_matmul(%a: tensor<4x8xf32>, %b: tensor<8x4xf32>) -> tensor<4x4xf32> {
    // CHECK: flash.matmul
    %c = flash.matmul %a, %b : tensor<4x8xf32>, tensor<8x4xf32> -> tensor<4x4xf32>
    return %c : tensor<4x4xf32>
  }
  
  // CHECK-LABEL: func @test_add
  func.func @test_add(%a: tensor<4x4xf32>, %b: tensor<4x4xf32>) -> tensor<4x4xf32> {
    // CHECK: flash.add
    %c = flash.add %a, %b : tensor<4x4xf32>, tensor<4x4xf32> -> tensor<4x4xf32>
    return %c : tensor<4x4xf32>
  }
  
  // CHECK-LABEL: func @test_relu
  func.func @test_relu(%a: tensor<4x4xf32>) -> tensor<4x4xf32> {
    // CHECK: flash.relu
    %b = flash.relu %a : tensor<4x4xf32> -> tensor<4x4xf32>
    return %b : tensor<4x4xf32>
  }
}