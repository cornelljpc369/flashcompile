//===- Passes.td - Flash Transform Passes ------------------*- tablegen -*-===//
//
// TableGen definitions for Flash optimization passes
//
//===----------------------------------------------------------------------===//

// #ifndef FLASH_TRANSFORMS_PASSES
// #define FLASH_TRANSFORMS_PASSES

include "mlir/Pass/PassBase.td"

//===----------------------------------------------------------------------===//
// Operator Fusion Pass
//===----------------------------------------------------------------------===//

def OperatorFusionPass : Pass<"flash-operator-fusion", "func::FuncOp"> {
  let summary = "Fuse compatible Flash operations";
  let description = [{
    Fuses sequences of operations that can be computed together efficiently.
    
    Example fusions:
    - MatMul + Add → FusedMatMulAdd
    - Add + ReLU → FusedAddReLU
    - MatMul + Add + ReLU → FusedMatMulAddReLU
    
    Benefits:
    - Reduced memory traffic
    - Better cache locality
    - Kernel launch overhead reduction (for GPU)
  }];
  
  let constructor = "mlir::flash::createOperatorFusionPass()";
  
  let dependentDialects = ["flash::FlashDialect"];
}

//===----------------------------------------------------------------------===//
// Constant Folding Pass
//===----------------------------------------------------------------------===//

def ConstantFoldingPass : Pass<"flash-constant-folding", "func::FuncOp"> {
  let summary = "Fold constant operations at compile time";
  let description = [{
    Evaluates operations with constant inputs at compile time.
    
    Example:
      %c1 = arith.constant 2.0
      %c2 = arith.constant 3.0
      %sum = flash.add %c1, %c2  →  %sum = arith.constant 5.0
    
    Benefits:
    - Reduces runtime computation
    - Exposes more optimization opportunities
    - Smaller graphs
  }];
  
  let constructor = "mlir::flash::createConstantFoldingPass()";
  
  let dependentDialects = ["flash::FlashDialect", "arith::ArithDialect"];
}

//===----------------------------------------------------------------------===//
// Common Subexpression Elimination (CSE)
//===----------------------------------------------------------------------===//

def CommonSubexpressionEliminationPass : Pass<"flash-cse", "func::FuncOp"> {
  let summary = "Eliminate redundant computations";
  let description = [{
    Removes duplicate computations with the same inputs.
    
    Example:
      %r1 = flash.add %a, %b
      %r2 = flash.add %a, %b  →  Use %r1 instead of computing again
    
    Benefits:
    - Fewer operations
    - Better register usage
    - Reduced computation time
  }];
  
  let constructor = "mlir::flash::createCommonSubexpressionEliminationPass()";
  
  let dependentDialects = ["flash::FlashDialect"];
}

//===----------------------------------------------------------------------===//
// Dead Code Elimination (DCE)
//===----------------------------------------------------------------------===//

def DeadCodeEliminationPass : Pass<"flash-dce", "func::FuncOp"> {
  let summary = "Remove unused operations";
  let description = [{
    Removes operations whose results are never used.
    
    Example:
      %unused = flash.matmul %a, %b  →  Remove (if %unused is never used)
    
    Benefits:
    - Smaller IR
    - Faster compilation
    - No wasted computation
  }];
  
  let constructor = "mlir::flash::createDeadCodeEliminationPass()";
  
  let dependentDialects = ["flash::FlashDialect"];
}

// #endif // FLASH_TRANSFORMS_PASSES