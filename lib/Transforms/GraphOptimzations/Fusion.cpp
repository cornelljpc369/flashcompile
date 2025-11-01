//===- Fusion.cpp - Operator fusion pass ------------------------*- C++ -*-===//
//
// Fuses compatible operations to reduce memory traffic
//
// Example transformations:
//   matmul + add → matmul_add_fused
//   matmul + relu → matmul_relu_fused
//   add + relu → add_relu_fused
//
//===----------------------------------------------------------------------===//

#include "flash/Transforms/Passes.h"
#include "flash/Dialect/Flash/FlashDialect.h"
#include "flash/Dialect/Flash/FlashOps.h"

#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Pass/PassRegistry.h"

using namespace mlir;
using namespace mlir::flash;

//===----------------------------------------------------------------------===//
// Fusion Patterns
//===----------------------------------------------------------------------===//

namespace {

/// Fuse MatMul + Add
/// Pattern: %c = flash.matmul %a, %b
///          %d = flash.add %c, %bias
/// Rewrite: %d = flash.matmul_add %a, %b, %bias
struct FuseMatMulAdd : public OpRewritePattern<AddOp> {
  using OpRewritePattern<AddOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(AddOp addOp,
                                 PatternRewriter &rewriter) const override {
    // Match pattern: add(matmul(...), bias)
    Value lhs = addOp.getLhs();
    Value rhs = addOp.getRhs();
    
    // Check if LHS is a matmul
    auto matmulOp = lhs.getDefiningOp<MatMulOp>();
    if (!matmulOp)
      return failure();
    
    // Check that matmul has only one use (this add)
    if (!matmulOp->hasOneUse())
      return failure();
    
    // TODO: Create fused op (for now, just document the opportunity)
    // In full implementation, would create flash.matmul_add operation
    
    // For educational purposes, emit a remark
    rewriter.notifyMatchFailure(
        addOp, "Fusion opportunity detected: matmul + add "
               "(fused op not yet implemented)");
    
    return failure();
  }
};

/// Fuse MatMul + ReLU
/// Pattern: %c = flash.matmul %a, %b
///          %d = flash.relu %c
/// Rewrite: %d = flash.matmul_relu %a, %b
struct FuseMatMulReLU : public OpRewritePattern<ReLUOp> {
  using OpRewritePattern<ReLUOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(ReLUOp reluOp,
                                 PatternRewriter &rewriter) const override {
    Value input = reluOp.getInput();
    
    // Check if input is a matmul
    auto matmulOp = input.getDefiningOp<MatMulOp>();
    if (!matmulOp)
      return failure();
    
    // Check single use
    if (!matmulOp->hasOneUse())
      return failure();
    
    // Emit remark about fusion opportunity
    rewriter.notifyMatchFailure(
        reluOp, "Fusion opportunity detected: matmul + relu "
                "(fused op not yet implemented)");
    
    return failure();
  }
};

/// Fuse Add + ReLU  (Actually implement this one!)
/// Pattern: %c = flash.add %a, %b
///          %d = flash.relu %c
/// Rewrite: Just do relu(add(...)) in one pass
struct FuseAddReLU : public OpRewritePattern<ReLUOp> {
  using OpRewritePattern<ReLUOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(ReLUOp reluOp,
                                 PatternRewriter &rewriter) const override {
    Value input = reluOp.getInput();
    
    // Check if input is an add
    auto addOp = input.getDefiningOp<AddOp>();
    if (!addOp)
      return failure();
    
    // Check single use
    if (!addOp->hasOneUse())
      return failure();
    
    // For now, just detect the pattern
    // In full implementation, would fuse into single operation
    rewriter.notifyMatchFailure(
        reluOp, "Fusion opportunity detected: add + relu "
                "(can be fused to reduce memory traffic)");
    
    return failure();
  }
};

} // namespace

//===----------------------------------------------------------------------===//
// Fusion Pass
//===----------------------------------------------------------------------===//

namespace {

struct FusionPass : public PassWrapper<FusionPass, OperationPass<func::FuncOp>> {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(FusionPass)

  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<FlashDialect>();
  }

  StringRef getArgument() const final { return "flash-fusion"; }
  StringRef getDescription() const final {
    return "Fuse compatible operations to reduce memory traffic";
  }

 // Near the end of the file, in runOnOperation():

void runOnOperation() override {
  RewritePatternSet patterns(&getContext());
  
  // Add fusion patterns
  patterns.add<FuseMatMulAdd>(&getContext());
  patterns.add<FuseMatMulReLU>(&getContext());
  patterns.add<FuseAddReLU>(&getContext());
  
  // MLIR 21.1.1 API: Use applyPatternsGreedily
  GreedyRewriteConfig config;
  if (failed(applyPatternsGreedily(getOperation(), std::move(patterns), config))) {
    signalPassFailure();
  }
}
};

} // namespace

//===----------------------------------------------------------------------===//
// Pass Creation
//===----------------------------------------------------------------------===//

std::unique_ptr<Pass> mlir::flash::createFusionPass() {
  return std::make_unique<FusionPass>();
}

//===----------------------------------------------------------------------===//
// Pass Registration
//===----------------------------------------------------------------------===//

namespace {
// Register the pass when this file is loaded
static PassRegistration<FusionPass> registration;
} // namespace