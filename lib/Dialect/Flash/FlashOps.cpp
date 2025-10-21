//===- FlashOps.cpp - Flash dialect operations ------------------*- C++ -*-===//
//
// FlashCompile: Operation implementations
//
//===----------------------------------------------------------------------===//

#include "flash/Dialect/Flash/FlashOps.h"
#include "flash/Dialect/Flash/FlashDialect.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/OpImplementation.h"

using namespace mlir;
using namespace mlir::flash;

//===----------------------------------------------------------------------===//
// Flash Operations
//===----------------------------------------------------------------------===//

#define GET_OP_CLASSES
#include "flash/Dialect/Flash/FlashOps.cpp.inc"