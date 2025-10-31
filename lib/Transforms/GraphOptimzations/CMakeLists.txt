add_mlir_library(FlashOptimizationGraphOpts
  Fusion.cpp
  ConstantFolding.cpp
  CSE.cpp

  LINK_LIBS PUBLIC
  MLIRIR
  MLIRFlashDialect
  MLIRArithDialect
  MLIRFuncDialect
  MLIRTransforms
  MLIRPass
)