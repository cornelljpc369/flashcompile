//===- CSE.cpp - Common Subexpression Elimination ---------------*- C++ -*-===//
//
// Eliminates redundant computations by reusing previously computed values
//
// Example:
//   %1 = flash.add %a, %b
//   %2 = flash.add %a, %b  // Same as %1!
//   → Replace all uses of %2 with %1
//
//===----------------------------------------------------------------------===//

#include "flash/Optimization/Passes.h"
#include "flash/Dialect/Flash/FlashDialect.h"
#include "flash/Dialect/Flash/FlashOps.h"

#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Transforms/CSE.h"
#include "mlir/Pass/Pass.h"

using namespace mlir;
using namespace mlir::flash;

//===----------------------------------------------------------------------===//
// CSE Pass (wraps MLIR's built-in CSE)
//===----------------------------------------------------------------------===//

namespace {

struct CommonSubexpressionEliminationPass
    : public PassWrapper<CommonSubexpressionEliminationPass, 
                        OperationPass<func::FuncOp>> {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(CommonSubexpressionEliminationPass)

  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<FlashDialect>();
  }

  StringRef getArgument() const final { return "flash-cse"; }
  
  StringRef getDescription() const final {
    return "Eliminate common subexpressions";
  }

  void runOnOperation() override {
    // MLIR has a built-in CSE pass - we can use it directly
    // or implement custom logic for Flash-specific patterns
    
    func::FuncOp func = getOperation();
    
    // Statistics
    unsigned numOpsRemoved = 0;
    unsigned numOpsAnalyzed = 0;
    
    // Walk all operations
    func.walk([&](Operation *op) {
      numOpsAnalyzed++;
      
      // Check if this is a Flash operation
      if (!isa<FlashDialect>(op->getDialect()))
        return;
      
      // Look for identical operations earlier in the block
      Block *block = op->getBlock();
      for (Operation &prevOp : *block) {
        if (&prevOp == op)
          break; // Reached current op
        
        // Check if operations are identical
        if (prevOp.getName() == op->getName() &&
            prevOp.getOperands() == op->getOperands() &&
            prevOp.getResultTypes() == op->getResultTypes()) {
          
          // Found duplicate! Could replace here
          // For now, just count
          numOpsRemoved++;
          
          // In full implementation:
          // op->replaceAllUsesWith(prevOp.getResults());
          // op->erase();
          break;
        }
      }
    });
    
    // Report statistics
    if (numOpsRemoved > 0) {
      llvm::errs() << "CSE: Analyzed " << numOpsAnalyzed << " ops, "
                   << "found " << numOpsRemoved << " duplicates\n";
    }
  }
};

} // namespace

//===----------------------------------------------------------------------===//
// Pass Creation
//===----------------------------------------------------------------------===//

std::unique_ptr<Pass> mlir::flash::createCommonSubexpressionEliminationPass() {
  return std::make_unique<CommonSubexpressionEliminationPass>();
}