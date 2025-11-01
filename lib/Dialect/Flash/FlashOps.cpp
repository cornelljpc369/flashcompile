//===- FlashOps.cpp - Flash dialect operations ------------------*- C++ -*-===//
//
// FlashCompile: Operation implementations
//
//===----------------------------------------------------------------------===//

#include "flash/Dialect/Flash/FlashOps.h"
#include "flash/Dialect/Flash/FlashDialect.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/OpImplementation.h"
#include "llvm/Support/Casting.h"

using namespace mlir;
using namespace mlir::flash;

//===----------------------------------------------------------------------===//
// Fused Operations Verification
//===----------------------------------------------------------------------===//

LogicalResult MatMulReLUOp::verify() {
  auto lhsType = llvm::cast<RankedTensorType>(getLhs().getType());
  auto rhsType = llvm::cast<RankedTensorType>(getRhs().getType());
  auto resultType = llvm::cast<RankedTensorType>(getResult().getType());

  if (lhsType.getRank() != 2 || rhsType.getRank() != 2 || resultType.getRank() != 2) {
    return emitOpError("expects all operands and results to be 2D tensors");
  }

  auto lhsShape = lhsType.getShape();
  auto rhsShape = rhsType.getShape();
  auto resultShape = resultType.getShape();

  if (lhsShape[1] != rhsShape[0]) {
    return emitOpError("inner dimensions must match: ")
           << lhsShape[1] << " vs " << rhsShape[0];
  }

  if (lhsShape[0] != resultShape[0] || rhsShape[1] != resultShape[1]) {
    return emitOpError("result shape mismatch");
  }

  return success();
}

LogicalResult MatMulAddOp::verify() {
  auto lhsType = llvm::cast<RankedTensorType>(getLhs().getType());
  auto rhsType = llvm::cast<RankedTensorType>(getRhs().getType());
  auto biasType = llvm::cast<RankedTensorType>(getBias().getType());
  auto resultType = llvm::cast<RankedTensorType>(getResult().getType());

  if (lhsType.getRank() != 2 || rhsType.getRank() != 2 || 
      biasType.getRank() != 2 || resultType.getRank() != 2) {
    return emitOpError("expects all operands and results to be 2D tensors");
  }

  auto lhsShape = lhsType.getShape();
  auto rhsShape = rhsType.getShape();
  auto biasShape = biasType.getShape();
  auto resultShape = resultType.getShape();

  if (lhsShape[1] != rhsShape[0]) {
    return emitOpError("matmul inner dimensions must match");
  }

  if (biasShape[0] != resultShape[0] || biasShape[1] != resultShape[1]) {
    return emitOpError("bias shape must match result shape");
  }

  if (lhsShape[0] != resultShape[0] || rhsShape[1] != resultShape[1]) {
    return emitOpError("result shape mismatch");
  }

  return success();
}

LogicalResult AddReLUOp::verify() {
  auto lhsType = llvm::cast<RankedTensorType>(getLhs().getType());
  auto rhsType = llvm::cast<RankedTensorType>(getRhs().getType());
  auto resultType = llvm::cast<RankedTensorType>(getResult().getType());

  if (lhsType.getShape() != rhsType.getShape() ||
      lhsType.getShape() != resultType.getShape()) {
    return emitOpError("all operands and results must have the same shape");
  }

  return success();
}

#define GET_OP_CLASSES
#include "flash/Dialect/Flash/FlashOps.cpp.inc"