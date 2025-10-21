# Get MLIR dialect libraries
get_property(dialect_libs GLOBAL PROPERTY MLIR_DIALECT_LIBS)
get_property(conversion_libs GLOBAL PROPERTY MLIR_CONVERSION_LIBS)

# Create test executable
add_executable(test_dialect
  test_dialect.cpp
)

# Include directories
target_include_directories(test_dialect
  PRIVATE
  ${LLVM_INCLUDE_DIRS}
  ${MLIR_INCLUDE_DIRS}
)

# Link libraries - simplified approach for macOS/Homebrew
target_link_libraries(test_dialect
  PRIVATE
  FlashDialect
  ${dialect_libs}
  MLIRIR
  MLIRParser
  MLIRSupport
  MLIRFuncDialect
  LLVMSupport
)

# Try to link GTest if found, otherwise continue without it
if(GTEST_LIBRARY AND GTEST_MAIN_LIBRARY)
  target_link_libraries(test_dialect PRIVATE ${GTEST_LIBRARY} ${GTEST_MAIN_LIBRARY})
endif()

# Register with CTest
add_test(NAME FlashDialectUnitTests COMMAND test_dialect)