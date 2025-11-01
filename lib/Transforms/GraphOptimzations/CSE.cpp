//===- CSE.cpp - Common Subexpression Elimination ---------------*- C++ -*-===//
//
// Actually eliminates redundant computations
//
//===----------------------------------------------------------------------===//

#include "flash/Transforms/Passes.h"
#include "flash/Dialect/Flash/FlashDialect.h"
#include "flash/Dialect/Flash/FlashOps.h"

#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Pass/Pass.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/Hashing.h"
#include <optional>
#include "mlir/Pass/PassRegistry.h"

using namespace mlir;
using namespace mlir::flash;

//===----------------------------------------------------------------------===//
// Operation Signature (for hashing)
//===----------------------------------------------------------------------===//

namespace {

/// Signature for an operation (for CSE matching)
struct OpSignature {
  // Use optional to allow uninitialized state for sentinels
  std::optional<OperationName> name;
  SmallVector<Value, 4> operands;
  SmallVector<Type, 2> resultTypes;
  
  // Default constructor for sentinels
  OpSignature() = default;
  
  // Constructor for actual operations
  explicit OpSignature(Operation *operation) {
    if (operation) {
      name = operation->getName();
      operands.assign(operation->getOperands().begin(), operation->getOperands().end());
      resultTypes.assign(operation->getResultTypes().begin(), operation->getResultTypes().end());
    }
  }
  
  bool operator==(const OpSignature &other) const {
    // If either doesn't have a name, it's a sentinel - compare by presence
    if (!name.has_value() || !other.name.has_value())
      return name.has_value() == other.name.has_value();
    
    // Otherwise compare operation signature
    return name.value() == other.name.value() &&
           operands == other.operands &&
           resultTypes == other.resultTypes;
  }
};

} // namespace

namespace llvm {
  template<> struct DenseMapInfo<OpSignature> {
    static OpSignature getEmptyKey() {
      // Default constructed = empty
      return OpSignature();
    }
    
    static OpSignature getTombstoneKey() {
      // Create a signature with tombstone marker
      OpSignature sig;
      // Use a dummy value in operands to distinguish from empty
      sig.operands.push_back(Value());
      return sig;
    }
    
    static unsigned getHashValue(const OpSignature &sig) {
      if (!sig.name.has_value())
        return 0; // Sentinel values
      
      return hash_combine(
          sig.name.value().getAsOpaquePointer(),
          hash_combine_range(sig.operands.begin(), sig.operands.end()),
          hash_combine_range(sig.resultTypes.begin(), sig.resultTypes.end())
      );
    }
    
    static bool isEqual(const OpSignature &lhs, const OpSignature &rhs) {
      return lhs == rhs;
    }
  };
}

//===----------------------------------------------------------------------===//
// CSE Pass Implementation
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
    func::FuncOp func = getOperation();
    
    // Statistics
    unsigned numOpsRemoved = 0;
    
    // Walk each block
    func.walk([&](Block *block) {
      eliminateCommonSubexpressionsInBlock(block, numOpsRemoved);
    });
    
    // Report statistics
    if (numOpsRemoved > 0) {
      llvm::outs() << "✓ CSE: Eliminated " << numOpsRemoved 
                   << " redundant operation(s)\n";
    }
  }
  
private:
  void eliminateCommonSubexpressionsInBlock(Block *block, unsigned &numRemoved) {
    // Map from operation signature to first occurrence
    DenseMap<OpSignature, Operation*> seenOps;
    
    // Track operations to erase (can't erase during walk)
    SmallVector<Operation*, 8> toErase;
    
    // Walk operations in block
    for (Operation &op : *block) {
      // Only process Flash operations
      if (!llvm::isa<FlashDialect>(op.getDialect()))
        continue;
      
      // Skip operations with side effects
      // MatMul, Add, ReLU are pure operations
      if (!llvm::isa<MatMulOp, AddOp, ReLUOp>(&op)) {
        continue;
      }
      
      // Create signature
      OpSignature sig(&op);
      
      // Check if we've seen this before
      auto it = seenOps.find(sig);
      if (it != seenOps.end()) {
        // Found duplicate! Replace with first occurrence
        Operation *firstOp = it->second;
        
        // Replace all uses of duplicate with first occurrence
        for (unsigned i = 0; i < op.getNumResults(); ++i) {
          op.getResult(i).replaceAllUsesWith(firstOp->getResult(i));
        }
        
        // Mark for erasure
        toErase.push_back(&op);
        numRemoved++;
        
      } else {
        // First occurrence - remember it
        seenOps[sig] = &op;
      }
    }
    
    // Erase redundant operations
    for (Operation *op : toErase) {
      op->erase();
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

//===----------------------------------------------------------------------===//
// Pass Registration
//===----------------------------------------------------------------------===//

namespace {
// Register the pass when this file is loaded
static PassRegistration<CommonSubexpressionEliminationPass> registration;
} // namespace