//===- Passes.h - Flash conversion passes -----------------------*- C++ -*-===//
//
// Declares Flash conversion passes
//
//===----------------------------------------------------------------------===//

#ifndef FLASH_CONVERSION_PASSES_H
#define FLASH_CONVERSION_PASSES_H

#include "mlir/Pass/Pass.h"
#include <memory>
#include "mlir/Dialect/Bufferization/IR/Bufferization.h"
namespace mlir {
namespace flash {

/// Create a pass to convert Flash operations to Linalg operations
std::unique_ptr<Pass> createConvertFlashToLinalgPass();

/// Create a pass to convert Linalg operations to Affine loops
std::unique_ptr<Pass> createConvertLinalgToAffinePass();

/// Create a pass to convert Affine operations to SCF
std::unique_ptr<Pass> createConvertAffineToSCFPass();

/// Generate pass registration code
#define GEN_PASS_REGISTRATION
#include "flash/Conversion/Passes.h.inc"

} // namespace flash
} // namespace mlir

#endif // FLASH_CONVERSION_PASSES_H