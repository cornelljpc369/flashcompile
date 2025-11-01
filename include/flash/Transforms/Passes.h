//===- Passes.h - Optimization passes ---------------------------*- C++ -*-===//
//
// Optimization passes for Flash compiler
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
std::unique_ptr<Pass> createFusionPass();

/// Fold constant operations at compile time
std::unique_ptr<Pass> createConstantFoldingPass();

/// Eliminate common subexpressions
std::unique_ptr<Pass> createCommonSubexpressionEliminationPass();

/// Remove unused operations
std::unique_ptr<Pass> createDeadCodeEliminationPass();

//===----------------------------------------------------------------------===//
// Loop Tiling
//===----------------------------------------------------------------------===//

/// Tile loops for better cache locality
/// Splits loops into outer tile loops and inner element loops
/// Example: for i in [0, N) → for ii in [0, N, tile), for i in [ii, ii+tile)
std::unique_ptr<Pass> createLoopTilingPass(unsigned tileSize = 32);

//===----------------------------------------------------------------------===//
// Loop Interchange
//===----------------------------------------------------------------------===//

/// Interchange loops for better memory access patterns
/// Reorders nested loops to improve spatial locality
std::unique_ptr<Pass> createLoopInterchangePass();

//===----------------------------------------------------------------------===//
// Loop Unrolling
//===----------------------------------------------------------------------===//

/// Unroll loops to reduce overhead and improve ILP
/// Duplicates loop body to process multiple iterations together
std::unique_ptr<Pass> createLoopUnrollingPass(unsigned factor = 4);

//===----------------------------------------------------------------------===//
// Vectorization (coming in Hours 37-40)
//===----------------------------------------------------------------------===//

/// Vectorize operations to use SIMD instructions
std::unique_ptr<Pass> createVectorizationPass();

/// Generate declarations for all passes defined in Passes.td
#define GEN_PASS_DECL
#include "flash/Transforms/Passes.h.inc"

/// Register all passes defined in Passes.td
void registerTransformPasses();


} // namespace flash
} // namespace mlir

#endif // FLASH_TRANSFORMS_PASSES_H