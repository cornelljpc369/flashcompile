// RUN: flash-opt %s --flash-loop-tiling | FileCheck %s

module {
  func.func @test_tiling() {
    // CHECK-LABEL: func.func @test_tiling
    
    %c0 = arith.constant 0 : index
    %c1024 = arith.constant 1024 : index
    %c1 = arith.constant 1 : index
    
    // CHECK: scf.for %{{.*}} = %c0 to %c1024 step %c32
    // CHECK:   scf.for %{{.*}} = %{{.*}} to %{{.*}} step %c1
    scf.for %i = %c0 to %c1024 step %c1 {
      // Loop body
      scf.yield
    }
    
    return
  }
}