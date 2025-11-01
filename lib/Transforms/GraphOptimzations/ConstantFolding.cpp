//===- ConstantFolding.cpp - Constant folding pass --------------*- C++ -*-===//
//
// Actually evaluates constant expressions at compile time
//
//===----------------------------------------------------------------------===//

#include "flash/Transforms/Passes.h"
#include "flash/Dialect/Flash/FlashDialect.h"
#include "flash/Dialect/Flash/FlashOps.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/Attributes.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "mlir/Pass/Pass.h"
#include "llvm/ADT/APFloat.h"


using namespace mlir;
using namespace mlir::flash;

//===----------------------------------------------------------------------===//
// Helper Functions
//===----------------------------------------------------------------------===//

namespace {

/// Get DenseElementsAttr from a value if it's a constant
DenseElementsAttr getConstantAttr(Value val) {
  if (auto defOp = val.getDefiningOp<arith::ConstantOp>()) {
    // MLIR 21.1.1 API: Use dyn_cast on the attribute directly
    if (auto attr = llvm::dyn_cast_if_present<DenseElementsAttr>(defOp.getValue())) {
      return attr;
    }
  }
  return nullptr;
}

/// Check if a value is a constant tensor
bool isConstantTensor(Value val) {
  return getConstantAttr(val) != nullptr;
}

/// Add two dense constant tensors element-wise
DenseElementsAttr addConstants(DenseElementsAttr lhs, DenseElementsAttr rhs) {
  auto lhsValues = lhs.getValues<APFloat>();
  auto rhsValues = rhs.getValues<APFloat>();
  
  SmallVector<APFloat> results;
  results.reserve(lhs.getNumElements());
  
  auto lhsIt = lhsValues.begin();
  auto rhsIt = rhsValues.begin();
  
  // Fix sign comparison warning
  int64_t numElements = lhs.getNumElements();
  for (int64_t i = 0; i < numElements; ++i) {
    APFloat sum = *lhsIt + *rhsIt;
    results.push_back(sum);
    ++lhsIt;
    ++rhsIt;
  }
  
  return DenseElementsAttr::get(lhs.getType(), results);
}

/// Apply ReLU to a dense constant tensor
DenseElementsAttr reluConstants(DenseElementsAttr input) {
  auto values = input.getValues<APFloat>();
  
  SmallVector<APFloat> results;
  results.reserve(input.getNumElements());
  
  APFloat zero(0.0f);
  
  for (const APFloat &val : values) {
    // max(0, val)
    if (val < zero) {
      results.push_back(zero);
    } else {
      results.push_back(val);
    }
  }
  
  return DenseElementsAttr::get(input.getType(), results);
}

} // namespace

//===----------------------------------------------------------------------===//
// Constant Folding Patterns
//===----------------------------------------------------------------------===//

namespace {

/// Fold constant Add operations - ACTUALLY FOLD!
struct FoldConstantAdd : public OpRewritePattern<AddOp> {
  using OpRewritePattern<AddOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(AddOp addOp,
                                 PatternRewriter &rewriter) const override {
    Value lhs = addOp.getLhs();
    Value rhs = addOp.getRhs();
    
    // Get constant attributes
    auto lhsAttr = getConstantAttr(lhs);
    auto rhsAttr = getConstantAttr(rhs);
    
    // Both must be constants
    if (!lhsAttr || !rhsAttr)
      return failure();
    
    // Must have float element type
    auto elemType = lhsAttr.getElementType();
    // MLIR 21.1.1 API: Use llvm::isa instead of isa member function
    if (!llvm::isa<FloatType>(elemType))
      return failure();
    
    // Actually compute the sum!
    auto resultAttr = addConstants(lhsAttr, rhsAttr);
    
    // Create constant op with result
    rewriter.replaceOpWithNewOp<arith::ConstantOp>(addOp, resultAttr);
    
    return success();
  }
};

/// Fold constant ReLU operations - ACTUALLY FOLD!
struct FoldConstantReLU : public OpRewritePattern<ReLUOp> {
  using OpRewritePattern<ReLUOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(ReLUOp reluOp,
                                 PatternRewriter &rewriter) const override {
    Value input = reluOp.getInput();
    
    // Get constant attribute
    auto inputAttr = getConstantAttr(input);
    if (!inputAttr)
      return failure();
    
    // Must have float element type
    auto elemType = inputAttr.getElementType();
    // MLIR 21.1.1 API: Use llvm::isa instead of isa member function
    if (!llvm::isa<FloatType>(elemType))
      return failure();
    
    // Actually apply ReLU!
    auto resultAttr = reluConstants(inputAttr);
    
    // Replace with constant
    rewriter.replaceOpWithNewOp<arith::ConstantOp>(reluOp, resultAttr);
    
    return success();
  }
};

/// Algebraic simplification: x + 0 → x - ACTUALLY SIMPLIFY!
struct SimplifyAddZero : public OpRewritePattern<AddOp> {
  using OpRewritePattern<AddOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(AddOp addOp,
                                 PatternRewriter &rewriter) const override {
    Value lhs = addOp.getLhs();
    Value rhs = addOp.getRhs();
    
    // Check if RHS is constant zero
    auto rhsAttr = getConstantAttr(rhs);
    if (!rhsAttr)
      return failure();
    
    // Check if all elements are zero
    bool allZero = true;
    for (auto val : rhsAttr.getValues<APFloat>()) {
      if (!val.isZero()) {
        allZero = false;
        break;
      }
    }
    
    if (!allZero)
      return failure();
    
    // Replace add with just LHS
    rewriter.replaceOp(addOp, lhs);
    
    return success();
  }
};

/// Algebraic simplification: ReLU(ReLU(x)) → ReLU(x)
struct SimplifyDoubleReLU : public OpRewritePattern<ReLUOp> {
  using OpRewritePattern<ReLUOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(ReLUOp reluOp,
                                 PatternRewriter &rewriter) const override {
    Value input = reluOp.getInput();
    
    // Check if input is also a ReLU
    if (auto innerReLU = input.getDefiningOp<ReLUOp>()) {
      // ReLU(ReLU(x)) = ReLU(x)
      rewriter.replaceOp(reluOp, innerReLU.getResult());
      return success();
    }
    
    return failure();
  }
};

} // namespace

//===----------------------------------------------------------------------===//
// Constant Folding Pass
//===----------------------------------------------------------------------===//

namespace {

struct ConstantFoldingPass 
    : public PassWrapper<ConstantFoldingPass, OperationPass<func::FuncOp>> {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(ConstantFoldingPass)

  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<FlashDialect, arith::ArithDialect>();
  }

  StringRef getArgument() const final { return "flash-constant-fold"; }
  
  StringRef getDescription() const final {
    return "Fold constant expressions at compile time";
  }

  void runOnOperation() override {
    MLIRContext *context = &getContext();
    RewritePatternSet patterns(context);
    
    // Add folding patterns
    patterns.add<FoldConstantAdd>(context);
    patterns.add<FoldConstantReLU>(context);
    patterns.add<SimplifyAddZero>(context);
    patterns.add<SimplifyDoubleReLU>(context);
    
    // MLIR 21.1.1 API: Use applyPatternsGreedily instead of deprecated function
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

std::unique_ptr<Pass> mlir::flash::createConstantFoldingPass() {
  return std::make_unique<ConstantFoldingPass>();
}

//===----------------------------------------------------------------------===//
// Pass Registration
//===----------------------------------------------------------------------===//

namespace {
// Register the pass when this file is loaded
static PassRegistration<ConstantFoldingPass> registration;
} // namespace