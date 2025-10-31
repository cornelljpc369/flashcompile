//===- Passes.h - Flash Transform Passes ------------------------*- C++ -*-===//
//
// Graph-level optimization passes for Flash dialect
//
//===----------------------------------------------------------------------===//

#ifndef FLASH_TRANSFORMS_PASSES_H
#define FLASH_TRANSFORMS_PASSES_H

#include "mlir/Pass/Pass.h"
#include <memory>

namespace mlir {
namespace flash {

//===----------------------------------------------------------------------===//
// Graph Optimizations
//===----------------------------------------------------------------------===//

/// Fuse compatible operations to reduce memory traffic
/// Examples: matmul+add, matmul+relu, add+relu
std::unique_ptr<Pass> createFusionPass();

/// Fold constant operations at compile time
/// Example: constant + constant → constant
std::unique_ptr<Pass> createConstantFoldingPass();

/// Eliminate common subexpressions
/// Example: x = a+b; y = a+b; → x = a+b; y = x;
std::unique_ptr<Pass> createCommonSubexpressionEliminationPass();

/// Remove unused operations
std::unique_ptr<Pass> createDeadCodeEliminationPass();

//===----------------------------------------------------------------------===//
// Loop Optimizations
//===----------------------------------------------------------------------===//

/// Tile loops for better cache locality
std::unique_ptr<Pass> createLoopTilingPass(unsigned tileSize = 32);

/// Interchange loops for better memory access patterns
std::unique_ptr<Pass> createLoopInterchangePass();

/// Unroll loops for better instruction-level parallelism
std::unique_ptr<Pass> createLoopUnrollingPass(unsigned factor = 4);

//===----------------------------------------------------------------------===//
// Vectorization
//===----------------------------------------------------------------------===//

/// Vectorize operations to use SIMD instructions
std::unique_ptr<Pass> createVectorizationPass();

//===----------------------------------------------------------------------===//
// Registration
//===----------------------------------------------------------------------===//

/// Register all optimization passes
void registerOptimizationPasses();

}// namespace flash
}// namespace mlir

#endif // FLASH_TRANSFORMS_PASSES_H