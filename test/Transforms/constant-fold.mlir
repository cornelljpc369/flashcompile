// RUN: flash-opt %s --flash-constant-fold | FileCheck %s

module {
  func.func @test_fold_add() -> tensor<2x2xf32> {
    // CHECK-LABEL: func.func @test_fold_add
    
    %c1 = arith.constant dense<[[1.0, 2.0], [3.0, 4.0]]> : tensor<2x2xf32>
    %c2 = arith.constant dense<[[5.0, 6.0], [7.0, 8.0]]> : tensor<2x2xf32>
    
    // CHECK: %[[RESULT:.*]] = arith.constant dense<{{\[}}[6.000000e+00, 8.000000e+00], [1.000000e+01, 1.200000e+01]]> : tensor<2x2xf32>
    %result = flash.add %c1, %c2 : tensor<2x2xf32>, tensor<2x2xf32> -> tensor<2x2xf32>
    
    // CHECK: return %[[RESULT]]
    return %result : tensor<2x2xf32>
  }
  
  func.func @test_fold_relu() -> tensor<2x2xf32> {
    // CHECK-LABEL: func.func @test_fold_relu
    
    %c = arith.constant dense<[[-1.0, 2.0], [-3.0, 4.0]]> : tensor<2x2xf32>
    
    // CHECK: %[[RESULT:.*]] = arith.constant dense<{{\[}}[0.000000e+00, 2.000000e+00], [0.000000e+00, 4.000000e+00]]> : tensor<2x2xf32>
    %result = flash.relu %c : tensor<2x2xf32> -> tensor<2x2xf32>
    
    // CHECK: return %[[RESULT]]
    return %result : tensor<2x2xf32>
  }
  
  func.func @test_simplify_add_zero() -> tensor<2x2xf32> {
    // CHECK-LABEL: func.func @test_simplify_add_zero
    
    %x = arith.constant dense<[[1.0, 2.0], [3.0, 4.0]]> : tensor<2x2xf32>
    %zero = arith.constant dense<0.0> : tensor<2x2xf32>
    
    // CHECK-NOT: flash.add
    // CHECK: return %{{.*}} : tensor<2x2xf32>
    %result = flash.add %x, %zero : tensor<2x2xf32>, tensor<2x2xf32> -> tensor<2x2xf32>
    
    return %result : tensor<2x2xf32>
  }
}