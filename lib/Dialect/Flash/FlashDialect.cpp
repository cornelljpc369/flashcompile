//===- FlashDialect.cpp - Flash dialect implementation --------------------===//
//
// Implements the Flash dialect
//
//===----------------------------------------------------------------------===//

#include "flash/Dialect/Flash/FlashDialect.h"
#include "flash/Dialect/Flash/FlashOps.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/DialectImplementation.h"

using namespace mlir;
using namespace mlir::flash;

//===----------------------------------------------------------------------===//
// Flash Dialect
//===----------------------------------------------------------------------===//

// Include the generated dialect implementation
#include "flash/Dialect/Flash/FlashDialect.cpp.inc"

void FlashDialect::initialize() {
  // Register all operations
  addOperations<
#define GET_OP_LIST
#include "flash/Dialect/Flash/FlashOps.cpp.inc"
  >();
}