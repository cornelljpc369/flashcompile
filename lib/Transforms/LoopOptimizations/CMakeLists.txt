add_mlir_library(FlashOptimizationLoopOpts
  LoopTiling.cpp

  LINK_LIBS PUBLIC
  MLIRIR
  MLIRSCFDialect
  MLIRAffineDialect
  MLIRArithDialect
  MLIRFuncDialect
  MLIRTransforms
  MLIRPass
  MLIRSupport
)