//===- Passes.td - Flash optimization passes ---------------*- tablegen -*-===//
//
// Defines optimization passes for Flash compiler
//
//===----------------------------------------------------------------------===//

#ifndef FLASH_OPTIMIZATION_PASSES
#define FLASH_OPTIMIZATION_PASSES

include "mlir/Pass/PassBase.td"

//===----------------------------------------------------------------------===//
// Graph Optimizations
//===----------------------------------------------------------------------===//

def FusionPass : Pass<"flash-fusion", "func::FuncOp"> {
  let summary = "Fuse compatible operations to reduce memory traffic";
  let description = [{
    Fuses operations like matmul+relu, matmul+add, add+relu to reduce
    memory bandwidth requirements and improve cache locality.
  }];
  let constructor = "mlir::flash::createFusionPass()";
}

def ConstantFoldingPass : Pass<"flash-constant-fold", "func::FuncOp"> {
  let summary = "Fold constant expressions at compile time";
  let description = [{
    Evaluates operations on constant tensors at compile time, reducing
    runtime computation overhead.
    
    Examples:
      - flash.add constant(...), constant(...) → constant(sum)
      - flash.relu constant(...) → constant(max(0, ...))
      - x + 0 → x
  }];
  let constructor = "mlir::flash::createConstantFoldingPass()";
}

def CSEPass : Pass<"flash-cse", "func::FuncOp"> {
  let summary = "Eliminate common subexpressions";
  let description = [{
    Eliminates redundant computations by reusing previously computed values.
    
    Example:
      %1 = flash.add %a, %b
      %2 = flash.add %a, %b  // Duplicate!
      → Replace all uses of %2 with %1
  }];
  let constructor = "mlir::flash::createCommonSubexpressionEliminationPass()";
}

#endif // FLASH_OPTIMIZATION_PASSES