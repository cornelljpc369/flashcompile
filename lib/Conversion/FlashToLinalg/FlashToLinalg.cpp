//===- FlashToLinalg.cpp - Flash to Linalg lowering ------------*- C++ -*-===//
//
// Lowers Flash dialect operations to Linalg dialect
//
//===----------------------------------------------------------------------===//

#include "flash/Conversion/Passes.h"
#include "flash/Dialect/Flash/FlashDialect.h"
#include "flash/Dialect/Flash/FlashOps.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Transforms/DialectConversion.h"
#include "mlir/Pass/Pass.h"

using namespace mlir;
using namespace mlir::flash;

//===----------------------------------------------------------------------===//
// Lowering Patterns
//===----------------------------------------------------------------------===//

namespace {

/// Lower flash.matmul to linalg.matmul
struct MatMulOpLowering : public OpRewritePattern<MatMulOp> {
  using OpRewritePattern<MatMulOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(MatMulOp op,PatternRewriter &rewriter) const override {
    Location loc = op.getLoc();
    Value lhs = op.getLhs();
    Value rhs = op.getRhs();
    
    // Get result type
    auto resultType = llvm::cast<RankedTensorType>(op.getResult().getType());
    
    // Create empty tensor for output
    // linalg.matmul requires an "outs" operand to write results into
    Value emptyTensor = rewriter.create<tensor::EmptyOp>(
        loc, resultType.getShape(), resultType.getElementType());
    
    // Create linalg.matmul
    // Syntax: linalg.matmul ins(%A, %B) outs(%C_init) -> tensor<...>
    Value result = rewriter.create<linalg::MatmulOp>(
        loc, 
        ValueRange{lhs, rhs},       // inputs
        ValueRange{emptyTensor}     // output (will be filled)
    ).getResult(0);
    
    // Replace flash.matmul with linalg.matmul result
    rewriter.replaceOp(op, result);
    
    return success();
  }
};

} // namespace

/// Lower flash.add to linalg.add

struct AddOpLowering : public OpRewritePattern<AddOp>{
  using OpRewritePattern<AddOp>::OpRewritePattern;
  
  LogicalResult matchAndRewrite(AddOp op,PatternRewriter &rewriter) const override {
    Location loc = op.getLoc();
    Value lhs = op.getLhs();
    Value rhs = op.getRhs();

    // Get result type
    auto resultType = llvm::cast<RankedTensorType>(op.getResult().getType());

    // Create empty tensor for output
    Value emptyTensor = rewriter.create<tensor::EmptyOp>(loc, resultType.getShape(), resultType.getElementType());

    // Create linalg.add
    Value result = rewriter.create<linalg::AddOp>(
        loc, 
        ValueRange{lhs, rhs},       // inputs
        ValueRange{emptyTensor}     // output
    ).getResult(0);
    // Replace flash.add with linalg.add result
    rewriter.replaceOp(op, result);
    
    return success();
  }
};

/// Lower flash.relu to linalg.generic with max(0, x)
struct ReLUOpLowering : public OpRewritePattern<ReLUOp> {
  using OpRewritePattern<ReLUOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(ReLUOp op, PatternRewriter &rewriter) const override {

    Location loc = op.getLoc();
    Value input = op.getInput();

    // Get result type
    auto resultType = llvm::cast<RankedTensorType>(op.getResult().getType());

    // Create empty tensor for output
    Value emptyTensor = rewriter.create<tensor::EmptyOp>(loc, resultType.getShape(), resultType.getElementType());

    // Create affine maps for linalg.generic
    // We want: out[i, j, ...] = max(0, in[i, j, ...])
    // This is an "identity" map - output indices = input indices

    int64_t rank = resultType.getRank();
    SmallVector<AffineMap> indexingMaps;
    indexingMaps.push_back(rewriter.getMultiDimIdentityMap(rank)); // input
    indexingMaps.push_back(rewriter.getMultiDimIdentityMap(rank)); // output

    // Iterator types: all "parallel" (no reductions)
    SmallVector<utils::IteratorType> iteratorTypes(rank, utils::IteratorType::parallel);

    // Create linalg.generic operation
    auto genericOp = rewriter.create<linalg::GenericOp>(
      loc,
      /*resultTypes=*/TypeRange{resultType},
      /*inputs=*/ValueRange{input},
      /*outputs=*/ValueRange{emptyTensor},
      /*indexingMaps=*/indexingMaps,
      /*iteratorTypes=*/iteratorTypes,
      [&](OpBuilder &b, Location loc, ValueRange args) {
        // Body of the operation: compute max(0, x)
        // args[0] = input element
        // args[1] = output element (uninitialized)
        Value inputElem = args[0];
        // Create constant 0.0
        Value zero = b.create<arith::ConstantOp>(loc, b.getFloatAttr(resultType.getElementType(), 0.0));
        // Compute max(0, input)
        Value result = b.create<arith::MaximumFOp>(loc, zero, inputElem);
        // Yield the result
        b.create<linalg::YieldOp>(loc, result);
      }
    );
    // Replace flash.relu with linalg.generic result
    rewriter.replaceOp(op, genericOp.getResult(0));

    return success();
  }
};

//===----------------------------------------------------------------------===//
// Pass Definition
//===----------------------------------------------------------------------===//

namespace {

struct ConvertFlashToLinalgPass: public PassWrapper<ConvertFlashToLinalgPass, OperationPass<func::FuncOp>> {
  
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(ConvertFlashToLinalgPass)
  
  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<linalg::LinalgDialect,
                    tensor::TensorDialect,
                    arith::ArithDialect>();
  }
  
  StringRef getArgument() const final {
    return "convert-flash-to-linalg";
  }
  
  StringRef getDescription() const final {
    return "Convert Flash operations to Linalg operations";
  }
  
  void runOnOperation() override {
    // Get the function we're operating on
    func::FuncOp func = getOperation();
    
    // Set up conversion target
    ConversionTarget target(getContext());
    
    // Mark Flash operations as illegal (must be converted)
    target.addIllegalDialect<FlashDialect>();
    
    // Mark Linalg, Tensor, Arith operations as legal (can remain)
    target.addLegalDialect<linalg::LinalgDialect,
                          tensor::TensorDialect,
                          arith::ArithDialect,
                          func::FuncDialect>();
    
    // Set up rewrite patterns
    RewritePatternSet patterns(&getContext());
    patterns.add<MatMulOpLowering, AddOpLowering, ReLUOpLowering>(&getContext());
    
    // Apply the patterns
    if (failed(applyPartialConversion(func, target, std::move(patterns)))) {
      signalPassFailure();
    }
  }
};

} // namespace

//===----------------------------------------------------------------------===//
// Pass Creation
//===----------------------------------------------------------------------===//

std::unique_ptr<Pass> mlir::flash::createConvertFlashToLinalgPass() {
  return std::make_unique<ConvertFlashToLinalgPass>();
}