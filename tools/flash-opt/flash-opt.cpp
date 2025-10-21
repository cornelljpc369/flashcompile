//===- flash-opt.cpp - FlashCompile optimizer driver ------------*- C++ -*-===//
//
// Minimal tool to test dialect loading
//
//===----------------------------------------------------------------------===//

#include "flash/Dialect/Flash/FlashDialect.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/InitAllDialects.h"
#include "mlir/Tools/mlir-opt/MlirOptMain.h"

int main(int argc, char **argv) {
  mlir::DialectRegistry registry;
  
  // Register Flash dialect
  registry.insert<mlir::flash::FlashDialect>();
  
  // Register standard MLIR dialects (for testing)
  mlir::registerAllDialects(registry);
  
  return mlir::asMainReturnCode(
      mlir::MlirOptMain(argc, argv, "FlashCompile optimizer\n", registry));
}