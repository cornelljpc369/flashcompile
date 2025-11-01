//===- LoopTiling.cpp - Loop tiling pass ------------------------*- C++ -*-===//
//
// Tiles loops to improve cache locality
//
// Transformation:
//   for i in [0, N):
//     for j in [0, M):
//       A[i][j] = ...
//
// Becomes:
//   for ii in [0, N, tile):
//     for jj in [0, M, tile):
//       for i in [ii, min(ii+tile, N)):
//         for j in [jj, min(jj+tile, M)):
//           A[i][j] = ...
//
//===----------------------------------------------------------------------===//

#include "flash/Transforms/Passes.h"

#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "mlir/IR/PatternMatch.h"
#include "llvm/Support/Debug.h"

using namespace mlir;

#define DEBUG_TYPE "flash-loop-tiling"

//===----------------------------------------------------------------------===//
// Loop Tiling Implementation
//===----------------------------------------------------------------------===//

namespace {

/// Tile a single scf.for loop
LogicalResult tileSCFForLoop(scf::ForOp forOp, unsigned tileSize,
                             PatternRewriter &rewriter) {
  Location loc = forOp.getLoc();
  
  Value lowerBound = forOp.getLowerBound();
  Value upperBound = forOp.getUpperBound();
  Value step = forOp.getStep();
  
  // Create tile size constant
  Value tileSizeVal = rewriter.create<arith::ConstantIndexOp>(loc, tileSize);
  
  // Calculate new step: step * tileSize
  Value newStep = rewriter.create<arith::MulIOp>(loc, step, tileSizeVal);
  
  // Create outer loop (tile loop)
  auto outerLoop = rewriter.create<scf::ForOp>(
      loc,
      lowerBound,
      upperBound,
      newStep,
      forOp.getInitArgs()
  );
  
  rewriter.setInsertionPointToStart(outerLoop.getBody());
  
  // Inner loop bounds: [outerIV, min(outerIV + tileSize * step, upperBound))
  Value outerIV = outerLoop.getInductionVar();
  
  // innerUpperBound = outerIV + tileSize * step
  Value tileOffset = rewriter.create<arith::MulIOp>(loc, tileSizeVal, step);
  Value innerUpperBoundTemp = rewriter.create<arith::AddIOp>(loc, outerIV, tileOffset);
  
  // innerUpperBound = min(innerUpperBoundTemp, upperBound)
  Value cmp = rewriter.create<arith::CmpIOp>(
      loc, arith::CmpIPredicate::slt, innerUpperBoundTemp, upperBound);
  Value innerUpperBound = rewriter.create<arith::SelectOp>(
      loc, cmp, innerUpperBoundTemp, upperBound);
  
  // Create inner loop (element loop)
  auto innerLoop = rewriter.create<scf::ForOp>(
      loc,
      outerIV,
      innerUpperBound,
      step,
      outerLoop.getRegionIterArgs()
  );
  
  // Move body of original loop to inner loop
  rewriter.mergeBlocks(
      forOp.getBody(),
      innerLoop.getBody(),
      innerLoop.getRegionIterArgs()
  );
  
  // Update inner loop to yield results
  rewriter.setInsertionPointToEnd(innerLoop.getBody());
  auto innerYield = cast<scf::YieldOp>(innerLoop.getBody()->getTerminator());
  
  // Outer loop yields inner loop results
  rewriter.setInsertionPointToEnd(outerLoop.getBody());
  rewriter.create<scf::YieldOp>(loc, innerLoop.getResults());
  
  // Replace original loop with outer loop
  rewriter.replaceOp(forOp, outerLoop.getResults());
  
  return success();
}

/// Pattern to tile scf.for loops
struct TileSCFForPattern : public OpRewritePattern<scf::ForOp> {
  unsigned tileSize;
  
  TileSCFForPattern(MLIRContext *context, unsigned tileSize)
      : OpRewritePattern<scf::ForOp>(context), tileSize(tileSize) {}

  LogicalResult matchAndRewrite(scf::ForOp forOp,
                                 PatternRewriter &rewriter) const override {
    // Only tile loops that haven't been tiled yet
    // (Check if this is already a tile loop by looking at parent)
    if (forOp->getParentOfType<scf::ForOp>())
      return failure(); // Already an inner loop, skip
    
    // Get trip count estimate
    // For now, tile all top-level loops
    
    return tileSCFForLoop(forOp, tileSize, rewriter);
  }
};

} // namespace

//===----------------------------------------------------------------------===//
// Loop Tiling Pass
//===----------------------------------------------------------------------===//

namespace {

struct LoopTilingPass 
    : public PassWrapper<LoopTilingPass, OperationPass<func::FuncOp>> {
  
  LoopTilingPass() = default;
  LoopTilingPass(unsigned tileSize) : tileSize(tileSize) {}
  
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(LoopTilingPass)

  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<scf::SCFDialect, arith::ArithDialect, affine::AffineDialect>();
  }

  StringRef getArgument() const final { return "flash-loop-tiling"; }
  
  StringRef getDescription() const final {
    return "Tile loops for better cache locality";
  }

  void runOnOperation() override {
    MLIRContext *context = &getContext();
    
    // Count loops before
    unsigned loopsBeforeTiling = 0;
    getOperation().walk([&](scf::ForOp) { loopsBeforeTiling++; });
    
    // Apply tiling
    RewritePatternSet patterns(context);
    patterns.add<TileSCFForPattern>(context, tileSize);
    
    if (failed(applyPatternsAndFoldGreedily(getOperation(), std::move(patterns)))) {
      signalPassFailure();
      return;
    }
    
    // Count loops after
    unsigned loopsAfterTiling = 0;
    getOperation().walk([&](scf::ForOp) { loopsAfterTiling++; });
    
    unsigned loopsTiled = (loopsAfterTiling - loopsBeforeTiling) / 2;
    
    if (loopsTiled > 0) {
      llvm::outs() << "✓ Loop Tiling: Tiled " << loopsTiled 
                   << " loop(s) with tile size " << tileSize << "\n";
    }
  }

  unsigned tileSize = 32;
};

} // namespace

//===----------------------------------------------------------------------===//
// Pass Creation
//===----------------------------------------------------------------------===//

std::unique_ptr<Pass> mlir::flash::createLoopTilingPass(unsigned tileSize) {
  return std::make_unique<LoopTilingPass>(tileSize);
}