add_mlir_library(MLIRLinalgToAffine
  LinalgToAffine.cpp

  ADDITIONAL_HEADER_DIRS
  ${PROJECT_SOURCE_DIR}/include/flash/Conversion

  DEPENDS
  MLIRFlashConversionPassIncGen

  LINK_LIBS PUBLIC
  MLIRAffineDialect
  MLIRArithDialect
  MLIRFuncDialect
  MLIRLinalgDialect
  MLIRMemRefDialect
  MLIRTensorDialect
  MLIRIR
  MLIRPass
  MLIRTransforms
  MLIRSupport
)