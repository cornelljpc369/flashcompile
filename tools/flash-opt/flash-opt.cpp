//===- flash-opt.cpp - FlashCompile optimizer driver ------------*- C++ -*-===//
//
// Tool for testing Flash dialect transformations
//
//===----------------------------------------------------------------------===//

#include "flash/Dialect/Flash/FlashDialect.h"
#include "flash/Conversion/Passes.h"

#include "mlir/IR/MLIRContext.h"
#include "mlir/InitAllDialects.h"
#include "mlir/InitAllPasses.h"
#include "mlir/Tools/mlir-opt/MlirOptMain.h"

int main(int argc, char **argv) {
  // Register all MLIR passes
  mlir::registerAllPasses();
  
  // Register Flash conversion passes
  mlir::flash::registerFlashConversionPasses();
  
  mlir::DialectRegistry registry;
  
  // Register Flash dialect
  registry.insert<mlir::flash::FlashDialect>();
  
  // Register standard MLIR dialects
  mlir::registerAllDialects(registry);
  
  return mlir::asMainReturnCode(
      mlir::MlirOptMain(argc, argv, "FlashCompile optimizer\n", registry));
}