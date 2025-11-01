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
#include "mlir/Dialect/Utils/StructuredOpsUtils.h"
#include "llvm/Support/Casting.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h" 

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
// Fused Operations Lowering - FULL IMPLEMENTATION
//===----------------------------------------------------------------------===//

/// Lower flash.matmul_relu to linalg.generic with fused computation
/// Computes: C[i,j] = max(0, sum_k(A[i,k] * B[k,j]))
/// Single loop nest, no intermediate storage!
struct MatMulReLUOpLowering : public OpRewritePattern<MatMulReLUOp> {
  using OpRewritePattern<MatMulReLUOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(MatMulReLUOp op,
                                 PatternRewriter &rewriter) const override {
    Location loc = op.getLoc();
    
    Value lhs = op.getLhs();
    Value rhs = op.getRhs();
    
    auto lhsType = llvm::cast<RankedTensorType>(op.getLhs().getType());
    auto rhsType = llvm::cast<RankedTensorType>(op.getRhs().getType());
    auto resultType = llvm::cast<RankedTensorType>(op.getType());
    
    // Get shapes: A is MxK, B is KxN, C is MxN
    // auto lhsShape = lhsType.getShape();
    // auto rhsShape = rhsType.getShape();
    
    // Create output tensor initialized to zero
    auto zero = rewriter.create<arith::ConstantOp>(
        loc, rewriter.getZeroAttr(resultType.getElementType()));
    
    auto emptyTensor = rewriter.create<tensor::EmptyOp>(
        loc, resultType, ValueRange{});
    
    auto fillOp = rewriter.create<linalg::FillOp>(
    loc, 
    ValueRange{zero.getResult()},           // inputs
    ValueRange{emptyTensor.getResult()}     // outputs
);
    
    // Create indexing maps for matmul
    // A: (i, k) -> affine_map<(i, j, k) -> (i, k)>
    // B: (k, j) -> affine_map<(i, j, k) -> (k, j)>
    // C: (i, j) -> affine_map<(i, j, k) -> (i, j)>
    
    MLIRContext *context = rewriter.getContext();
    
    AffineExpr iExpr = rewriter.getAffineDimExpr(0);
    AffineExpr jExpr = rewriter.getAffineDimExpr(1);
    AffineExpr kExpr = rewriter.getAffineDimExpr(2);
    
    auto mapA = AffineMap::get(3, 0, {iExpr, kExpr}, context);  // (i, k)
    auto mapB = AffineMap::get(3, 0, {kExpr, jExpr}, context);  // (k, j)
    auto mapC = AffineMap::get(3, 0, {iExpr, jExpr}, context);  // (i, j)
    
    SmallVector<AffineMap> indexingMaps = {mapA, mapB, mapC};
    
    // Iterator types: parallel(i), parallel(j), reduction(k)
    SmallVector<utils::IteratorType> iteratorTypes = {
        utils::IteratorType::parallel,
        utils::IteratorType::parallel,
        utils::IteratorType::reduction
    };
    
    // Create linalg.generic that computes matmul+relu in one pass
    auto genericOp = rewriter.create<linalg::GenericOp>(
        loc,
        /*resultTensorTypes=*/resultType,
        /*inputs=*/ValueRange{lhs, rhs},
        /*outputs=*/ValueRange{fillOp.getResult(0)},
        /*indexingMaps=*/indexingMaps,
        /*iteratorTypes=*/iteratorTypes,
        [&](OpBuilder &b, Location loc, ValueRange args) {
          // args[0] = A[i,k], args[1] = B[k,j], args[2] = C[i,j] (accumulator)
          
          // Compute: C[i,j] += A[i,k] * B[k,j]
          Value mul = b.create<arith::MulFOp>(loc, args[0], args[1]);
          Value acc = b.create<arith::AddFOp>(loc, args[2], mul);
          
          // Apply ReLU: max(0, acc)
          Value zero = b.create<arith::ConstantOp>(
              loc, b.getZeroAttr(resultType.getElementType()));
          Value relu = b.create<arith::MaximumFOp>(loc, acc, zero);
          
          b.create<linalg::YieldOp>(loc, relu);
        }
    );
    
    rewriter.replaceOp(op, genericOp.getResults());
    
    return success();
  }
};

/// Lower flash.matmul_add to linalg.generic with fused computation
/// Computes: C[i,j] = sum_k(A[i,k] * B[k,j]) + bias[i,j]
/// Single loop nest!
struct MatMulAddOpLowering : public OpRewritePattern<MatMulAddOp> {
  using OpRewritePattern<MatMulAddOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(MatMulAddOp op,
                                 PatternRewriter &rewriter) const override {
    Location loc = op.getLoc();
    
    Value lhs = op.getLhs();
    Value rhs = op.getRhs();
    Value bias = op.getBias();
    
    auto resultType = llvm::cast<RankedTensorType>(op.getType());
    
    // Initialize output to bias
    auto emptyTensor = rewriter.create<tensor::EmptyOp>(
        loc, resultType.getShape(), resultType.getElementType());
    
    // FIX: linalg.copy also uses ValueRange
    auto copyBias = rewriter.create<linalg::CopyOp>(
        loc,
        bias,
        emptyTensor.getResult()
    );
    
    MLIRContext *context = rewriter.getContext();
    
    AffineExpr iExpr = rewriter.getAffineDimExpr(0);
    AffineExpr jExpr = rewriter.getAffineDimExpr(1);
    AffineExpr kExpr = rewriter.getAffineDimExpr(2);
    
    auto mapA = AffineMap::get(3, 0, {iExpr, kExpr}, context);
    auto mapB = AffineMap::get(3, 0, {kExpr, jExpr}, context);
    auto mapC = AffineMap::get(3, 0, {iExpr, jExpr}, context);
    
    SmallVector<AffineMap> indexingMaps = {mapA, mapB, mapC};
    
    SmallVector<utils::IteratorType> iteratorTypes = {
        utils::IteratorType::parallel,
        utils::IteratorType::parallel,
        utils::IteratorType::reduction
    };
    
    auto genericOp = rewriter.create<linalg::GenericOp>(
        loc,
        resultType,
        ValueRange{lhs, rhs},
        ValueRange{copyBias.getResult(0)},
        indexingMaps,
        iteratorTypes,
        [&](OpBuilder &b, Location loc, ValueRange args) {
          Value mul = b.create<arith::MulFOp>(loc, args[0], args[1]);
          Value result = b.create<arith::AddFOp>(loc, args[2], mul);
          
          b.create<linalg::YieldOp>(loc, result);
        }
    );
    
    rewriter.replaceOp(op, genericOp.getResults());
    
    return success();
  }
};

/// Lower flash.add_relu to linalg.map
/// Computes: C[i,j] = max(0, A[i,j] + B[i,j])
/// Element-wise, easily parallelizable
struct AddReLUOpLowering : public OpRewritePattern<AddReLUOp> {
  using OpRewritePattern<AddReLUOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(AddReLUOp op,
                                 PatternRewriter &rewriter) const override {
    Location loc = op.getLoc();
    
    Value lhs = op.getLhs();
    Value rhs = op.getRhs();
    
    auto resultType = llvm::cast<RankedTensorType>(op.getType());
    
    // Create empty output tensor
    auto emptyTensor = rewriter.create<tensor::EmptyOp>(
        loc, resultType.getShape(), resultType.getElementType());
    
    MLIRContext *context = rewriter.getContext();
    unsigned rank = resultType.getRank();
    
    SmallVector<AffineExpr> exprs;
    for (unsigned i = 0; i < rank; ++i) {
      exprs.push_back(rewriter.getAffineDimExpr(i));
    }
    
    auto identityMap = AffineMap::get(rank, 0, exprs, context);
    SmallVector<AffineMap> indexingMaps = {identityMap, identityMap, identityMap};
    
    SmallVector<utils::IteratorType> iteratorTypes(
        rank, utils::IteratorType::parallel);
    
    auto genericOp = rewriter.create<linalg::GenericOp>(
        loc,
        resultType,
        ValueRange{lhs, rhs},
        ValueRange{emptyTensor.getResult()},
        indexingMaps,
        iteratorTypes,
        [&](OpBuilder &b, Location loc, ValueRange args) {
          Value add = b.create<arith::AddFOp>(loc, args[0], args[1]);
          
          Value zero = b.create<arith::ConstantOp>(
              loc, b.getZeroAttr(resultType.getElementType()));
          Value relu = b.create<arith::MaximumFOp>(loc, add, zero);
          
          b.create<linalg::YieldOp>(loc, relu);
        }
    );
    
    rewriter.replaceOp(op, genericOp.getResults());
    
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
      // FUSED operations - NEW!
    patterns.add<MatMulReLUOpLowering>(&getContext());
    patterns.add<MatMulAddOpLowering>(&getContext());
    patterns.add<AddReLUOpLowering>(&getContext());
    // Apply the patterns
if (failed(applyPatternsAndFoldGreedily(getOperation(), std::move(patterns)))) {
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