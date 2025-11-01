#include "flash/Transforms/Passes.h"

using namespace mlir;
using namespace mlir::flash;

#define GEN_PASS_REGISTRATION
#include "flash/Transforms/Passes.h.inc"

void mlir::flash::registerTransformPasses() {
  registerFlashTransformsPasses();
}