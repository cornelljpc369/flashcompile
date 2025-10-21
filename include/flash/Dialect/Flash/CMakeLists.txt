# Flash dialect TableGen
# Generate dialect definitions from TableGen
set(LLVM_TARGET_DEFINITIONS FlashDialect.td)
mlir_tablegen(FlashDialect.h.inc -gen-dialect-decls)
mlir_tablegen(FlashDialect.cpp.inc -gen-dialect-defs)
add_public_tablegen_target(FlashDialectIncGen)

# Generate operation definitions from TableGen
set(LLVM_TARGET_DEFINITIONS FlashOps.td)
mlir_tablegen(FlashOps.h.inc -gen-op-decls)
mlir_tablegen(FlashOps.cpp.inc -gen-op-defs)
add_public_tablegen_target(FlashOpsIncGen)