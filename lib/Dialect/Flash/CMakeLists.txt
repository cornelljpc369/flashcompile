# Flash dialect implementation (empty for now)
# Build the Flash dialect library
add_mlir_dialect_library(MLIRFlashDialect
  FlashDialect.cpp
  FlashOps.cpp

  ADDITIONAL_HEADER_DIRS
  ${PROJECT_SOURCE_DIR}/include/flash/Dialect/Flash

  DEPENDS
  MLIRFlashDialectIncGen
  MLIRFlashOpsIncGen

  LINK_LIBS PUBLIC
  MLIRIR
  MLIRSideEffectInterfaces
)