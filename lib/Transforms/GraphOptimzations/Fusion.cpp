//===- Fusion.cpp - Operator fusion pass ------------------------*- C++ -*-===//
//
// ACTUALLY fuses compatible operations (not just detection!)
//
//===----------------------------------------------------------------------===//

#include "flash/Transforms/Passes.h"
#include "flash/Dialect/Flash/FlashDialect.h"
#include "flash/Dialect/Flash/FlashOps.h"

#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "mlir/Pass/Pass.h"

using namespace mlir;
using namespace mlir::flash;

//===----------------------------------------------------------------------===//
// Fusion Patterns - ACTUALLY REWRITE!
//===----------------------------------------------------------------------===//

namespace {

/// Fuse MatMul + ReLU → MatMulReLU
struct FuseMatMulReLU : public OpRewritePattern<ReLUOp> {
  using OpRewritePattern<ReLUOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(ReLUOp reluOp,
                                 PatternRewriter &rewriter) const override {
    Value input = reluOp.getInput();
    
    // Check if input is a matmul
    auto matmulOp = input.getDefiningOp<MatMulOp>();
    if (!matmulOp)
      return failure();
    
    // Check that matmul has only one use (this relu)
    if (!matmulOp->hasOneUse())
      return failure();
    
    // ACTUALLY CREATE FUSED OP!
    rewriter.replaceOpWithNewOp<MatMulReLUOp>(
        reluOp,
        reluOp.getType(),
        matmulOp.getLhs(),
        matmulOp.getRhs()
    );
    
    return success();
  }
};

/// Fuse MatMul + Add → MatMulAdd
struct FuseMatMulAdd : public OpRewritePattern<AddOp> {
  using OpRewritePattern<AddOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(AddOp addOp,
                                 PatternRewriter &rewriter) const override {
    Value lhs = addOp.getLhs();
    Value rhs = addOp.getRhs();
    
    // Check if LHS is a matmul
    auto matmulOp = lhs.getDefiningOp<MatMulOp>();
    if (!matmulOp)
      return failure();
    
    // Check single use
    if (!matmulOp->hasOneUse())
      return failure();
    
    // RHS should be the bias (can be constant or variable)
    // ACTUALLY CREATE FUSED OP!
    rewriter.replaceOpWithNewOp<MatMulAddOp>(
        addOp,
        addOp.getType(),
        matmulOp.getLhs(),
        matmulOp.getRhs(),
        rhs  // bias
    );
    
    return success();
  }
};

/// Fuse Add + ReLU → AddReLU
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
    
    // ACTUALLY CREATE FUSED OP!
    rewriter.replaceOpWithNewOp<AddReLUOp>(
        reluOp,
        reluOp.getType(),
        addOp.getLhs(),
        addOp.getRhs()
    );
    
    return success();
  }
};

/// Fuse MatMul + Add + ReLU → MatMulAdd + ReLU (first pass)
/// Then second pass will fuse the remaining Add+ReLU
/// This demonstrates multi-pass fusion

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

  void runOnOperation() override {
    MLIRContext *context = &getContext();
    RewritePatternSet patterns(context);
    
    // Add fusion patterns
    patterns.add<FuseMatMulReLU>(context);
    patterns.add<FuseMatMulAdd>(context);
    patterns.add<FuseAddReLU>(context);
    
    // Apply patterns greedily
    // This will iterate until no more patterns match
    if (failed(applyPatternsAndFoldGreedily(getOperation(), std::move(patterns)))) {
      signalPassFailure();
    }
    
    // Count fused operations for statistics
    unsigned numFused = 0;
    getOperation().walk([&](Operation *op) {
      if (isa<MatMulReLUOp, MatMulAddOp, AddReLUOp>(op)) {
        numFused++;
      }
    });
    
    if (numFused > 0) {
      llvm::outs() << "✓ Fusion: Created " << numFused 
                   << " fused operation(s)\n";
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