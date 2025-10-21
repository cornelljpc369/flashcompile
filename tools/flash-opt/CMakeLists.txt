get_property(dialect_libs GLOBAL PROPERTY MLIR_DIALECT_LIBS)
get_property(conversion_libs GLOBAL PROPERTY MLIR_CONVERSION_LIBS)

add_llvm_executable(flash-opt
  flash-opt.cpp
  )

target_link_libraries(flash-opt
  PRIVATE
  MLIRFlashDialect
  ${dialect_libs}
  ${conversion_libs}
  MLIROptLib
  )