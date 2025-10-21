//===- FlashOps.h - Flash dialect operations --------------------*- C++ -*-===//
//
// Flash operations declaration
//
//===----------------------------------------------------------------------===//

#ifndef FLASH_DIALECT_FLASH_FLASHOPS_H
#define FLASH_DIALECT_FLASH_FLASHOPS_H

#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Dialect.h"
#include "mlir/IR/OpDefinition.h"
#include "mlir/Interfaces/SideEffectInterfaces.h"

// Include our dialect
#include "flash/Dialect/Flash/FlashDialect.h"

// This macro tells MLIR to include operation class definitions
#define GET_OP_CLASSES
#include "flash/Dialect/Flash/FlashOps.h.inc"

#endif // FLASH_DIALECT_FLASH_FLASHOPS_H