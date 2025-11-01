add_mlir_library(FlashOptimizationGraphOpts
  Fusion.cpp
  ConstantFolding.cpp
  CSE.cpp

  DEPENDS
  MLIRFlashOpsIncGen
  MLIRFlashTransformsPassIncGen

  LINK_LIBS PUBLIC
  MLIRIR
  MLIRFlashDialect
  MLIRArithDialect
  MLIRFuncDialect
  MLIRTransforms
  MLIRPass
  MLIRSupport
)