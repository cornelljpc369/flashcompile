//===- FlashOps.td - Flash dialect operations ------------------------------===//
//
// Defines operations in the Flash dialect
//
//===----------------------------------------------------------------------===//

//#ifndef FLASH_OPS
//#define FLASH_OPS

include "flash/Dialect/Flash/FlashDialect.td"
include "mlir/Interfaces/SideEffectInterfaces.td"

//===----------------------------------------------------------------------===//
// MatMul Operation
//===----------------------------------------------------------------------===//

def Flash_MatMulOp : Flash_Op<"matmul", [Pure]> {
    let summary = "Matrix Multiplication operation";

    let description = [{
        Performs matrix multiplication: C = A @ B
    
    The operation takes two 2D tensors and produces their matrix product.
    Shapes must be compatible: (M x K) @ (K x N) = (M x N)
    
    Example:
```mlir
    %C = flash.matmul %A, %B : tensor<64x128xf32>, tensor<128x256xf32> 
                                -> tensor<64x256xf32>
```
    }];

    //Input Arguments (operands)
    let arguments = (ins
        AnyTensor:$lhs, // Left-hand side matrix (A)
        AnyTensor:$rhs  // Right-hand side matrix (B)
    );

    //Output results
    let results = (outs
        AnyTensor:$result // Result matrix (C)
    );

    // How to print/parse this operation
    let assemblyFormat = [{
        $lhs `,` $rhs attr-dict `:` type($lhs) `,` type($rhs) `->` type($result)
    }];
}

//#endif //FLASH_OPS