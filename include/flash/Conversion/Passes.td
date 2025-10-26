//===- Passes.td - Flash conversion passes ------------------*- tablegen -*-===//
//
// Defines conversion passes for the Flash dialect
//
//===----------------------------------------------------------------------===//

#ifndef FLASH_CONVERSION_PASSES
#define FLASH_CONVERSION_PASSES

include "mlir/Pass/PassBase.td"

//===----------------------------------------------------------------------===//
// ConvertFlashToLinalg Pass
//===----------------------------------------------------------------------===//

def ConvertFlashToLinalg : Pass<"convert-flash-to-linalg", "::mlir::func::FuncOp"> {
  let summary = "Convert Flash operations to Linalg operations";
  
  let description = [{
    This pass lowers Flash dialect operations to equivalent Linalg operations.
    
    Lowering patterns:
    - flash.matmul  → linalg.matmul
    - flash.add     → linalg.add
    - flash.relu    → linalg.generic with max(0, x)
    - flash.conv2d  → linalg.conv_2d_nchw_fchw
    
    This is the first step in compiling Flash IR to executable code.
  }];
  
  let constructor = "mlir::flash::createConvertFlashToLinalgPass()";
  
  let dependentDialects = [
    "mlir::linalg::LinalgDialect",
    "mlir::tensor::TensorDialect",
    "mlir::arith::ArithDialect"
  ];
}

//===----------------------------------------------------------------------===//
// ConvertLinalgToAffine Pass
//===----------------------------------------------------------------------===//

/*def ConvertLinalgToAffine : Pass<"convert-linalg-to-affine", "::mlir::func::FuncOp"> {
  let summary = "Convert Linalg operations to Affine loops";
  
  let description = [{
    This pass lowers Linalg structured operations to explicit Affine loop nests.
    
    Transformations:
    - linalg.matmul → triple nested affine.for (i, j, k loops)
    - linalg.add → double nested affine.for (i, j loops)
    - linalg.generic → affine.for based on iterator types
    
    This exposes explicit loop structure for polyhedral optimization.
  }];
  
  let constructor = "mlir::flash::createConvertLinalgToAffinePass()";
  
  let dependentDialects = [
    "mlir::affine::AffineDialect",
    "mlir::memref::MemRefDialect",
    "mlir::arith::ArithDialect"
  ];
}

//===----------------------------------------------------------------------===//
// ConvertAffineToSCF Pass
//===----------------------------------------------------------------------===//

def ConvertAffineToSCF : Pass<"convert-affine-to-scf-custom", "::mlir::func::FuncOp"> {
  let summary = "Convert Affine operations to SCF (Structured Control Flow)";
  
  let description = [{
    This pass lowers Affine dialect operations to SCF dialect.
    
    Transformations:
    - affine.for → scf.for with explicit bounds
    - affine.load → memref.load
    - affine.store → memref.store
    - affine.if → scf.if
    
    This creates a uniform loop representation before LLVM lowering.
  }];
  
  let constructor = "mlir::flash::createConvertAffineToSCFPass()";
  
  let dependentDialects = [
    "mlir::scf::SCFDialect",
    "mlir::memref::MemRefDialect",
    "mlir::arith::ArithDialect"
  ];
}
*/
#endif // FLASH_CONVERSION_PASSES