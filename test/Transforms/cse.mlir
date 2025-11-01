// RUN: flash-opt %s --flash-cse | FileCheck %s

module {
  func.func @test_cse() -> tensor<2x2xf32> {
    // CHECK-LABEL: func.func @test_cse
    
    %x = arith.constant dense<[[1.0, 2.0], [3.0, 4.0]]> : tensor<2x2xf32>
    %y = arith.constant dense<[[5.0, 6.0], [7.0, 8.0]]> : tensor<2x2xf32>
    
    // CHECK: %[[R1:.*]] = flash.add
    %r1 = flash.add %x, %y : tensor<2x2xf32>, tensor<2x2xf32> -> tensor<2x2xf32>
    
    // CHECK-NOT: flash.add %x, %y
    %r2 = flash.add %x, %y : tensor<2x2xf32>, tensor<2x2xf32> -> tensor<2x2xf32>
    
    // CHECK: flash.add %[[R1]], %[[R1]]
    %result = flash.add %r1, %r2 : tensor<2x2xf32>, tensor<2x2xf32> -> tensor<2x2xf32>
    
    return %result : tensor<2x2xf32>
  }
}