//===- AffineToSCF.cpp - Affine to SCF lowering ----------------*- C++ -*-===//
//
// Lowers Affine dialect to SCF using MLIR built-in utilities
//
//===----------------------------------------------------------------------===//

#include "flash/Conversion/Passes.h"

#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Transforms/DialectConversion.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

using namespace mlir;

//===----------------------------------------------------------------------===//
// Helper: Lower Affine For Op to SCF For Op
//===----------------------------------------------------------------------===//

namespace {

// Simple conversion that just converts affine.for with constant bounds
LogicalResult lowerAffineForOp(affine::AffineForOp affineOp, OpBuilder &rewriter) {
  Location loc = affineOp.getLoc();
  
  // Only handle constant bounds for simplicity
  if (!affineOp.hasConstantLowerBound() || !affineOp.hasConstantUpperBound()) {
    return failure();
  }
  
  int64_t lowerBound = affineOp.getConstantLowerBound();
  int64_t upperBound = affineOp.getConstantUpperBound();
  int64_t step = affineOp.getStep().getSExtValue();
  
  // Create SCF bounds
  Value lb = rewriter.create<arith::ConstantIndexOp>(loc, lowerBound);
  Value ub = rewriter.create<arith::ConstantIndexOp>(loc, upperBound);
  Value stepVal = rewriter.create<arith::ConstantIndexOp>(loc, step);
  
  // Create SCF for loop
  auto scfFor = rewriter.create<scf::ForOp>(loc, lb, ub, stepVal);
  
  // Move body operations
//   Block *affineBody = affineOp.getBody();
  Block *scfBody = scfFor.getBody();
  
  // Map induction variables
  rewriter.eraseBlock(&scfFor.getRegion().front());
  rewriter.inlineRegionBefore(affineOp.getRegion(), scfFor.getRegion(), scfFor.getRegion().end());
  scfBody = &scfFor.getRegion().front();
  
  // Replace IV
  scfBody->getArgument(0).replaceAllUsesWith(scfBody->getArgument(0));
  
  // Add yield if needed
  if (scfBody->empty() || !isa<scf::YieldOp>(scfBody->back())) {
    rewriter.setInsertionPointToEnd(scfBody);
    rewriter.create<scf::YieldOp>(loc);
  }
  
  // Replace affine terminator with scf yield
  for (auto &op : llvm::make_early_inc_range(*scfBody)) {
    if (isa<affine::AffineYieldOp>(op)) {
      rewriter.setInsertionPoint(&op);
      rewriter.create<scf::YieldOp>(loc);
      rewriter.eraseOp(&op);
    }
  }
  
  rewriter.replaceOp(affineOp, scfFor.getResults());
  return success();
}

} // namespace

//===----------------------------------------------------------------------===//
// Pass Definition
//===----------------------------------------------------------------------===//

namespace {

struct ConvertAffineToSCFPass
    : public PassWrapper<ConvertAffineToSCFPass,
                        OperationPass<func::FuncOp>> {
  
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(ConvertAffineToSCFPass)
  
  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<scf::SCFDialect,
                    memref::MemRefDialect,
                    arith::ArithDialect>();
  }
  
  StringRef getArgument() const final {
    return "convert-affine-to-scf-custom";
  }
  
  StringRef getDescription() const final {
    return "Convert Affine operations to SCF";
  }
  
  void runOnOperation() override {
    auto func = getOperation();
    
    // Walk and convert affine.for ops
    func.walk([&](affine::AffineForOp affineOp) {
      OpBuilder builder(affineOp);
      if (failed(lowerAffineForOp(affineOp, builder))) {
        signalPassFailure();
      }
    });
    
    // Convert affine.load to memref.load
    func.walk([&](affine::AffineLoadOp loadOp) {
      OpBuilder builder(loadOp);
      auto memrefLoad = builder.create<memref::LoadOp>(
          loadOp.getLoc(), loadOp.getMemRef(), loadOp.getMapOperands());
      loadOp.replaceAllUsesWith(memrefLoad.getResult());
      loadOp.erase();
    });
    
    // Convert affine.store to memref.store
    func.walk([&](affine::AffineStoreOp storeOp) {
      OpBuilder builder(storeOp);
      builder.create<memref::StoreOp>(
          storeOp.getLoc(), storeOp.getValue(), 
          storeOp.getMemRef(), storeOp.getMapOperands());
      storeOp.erase();
    });
  }
};

} // namespace

//===----------------------------------------------------------------------===//
// Pass Creation
//===----------------------------------------------------------------------===//

std::unique_ptr<Pass> mlir::flash::createConvertAffineToSCFPass() {
  return std::make_unique<ConvertAffineToSCFPass>();
}