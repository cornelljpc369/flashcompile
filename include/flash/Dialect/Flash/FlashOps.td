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

//===----------------------------------------------------------------------===//
// Add Operation
//===----------------------------------------------------------------------===//

def Flash_AddOp : Flash_Op<"add", [Pure, Commutative]> {
  let summary = "Element-wise addition operation";
  
  let description = [{
    Performs element-wise addition: C = A + B
    
    The operation adds two tensors element-wise. Supports broadcasting
    following NumPy-style broadcasting rules.
    
    Example:
```mlir
    %C = flash.add %A, %B : tensor<4x4xf32>, tensor<4x4xf32> -> tensor<4x4xf32>
    
    // Broadcasting: add bias vector to each row
    %D = flash.add %mat, %bias : tensor<4x8xf32>, tensor<8xf32> -> tensor<4x8xf32>
```
  }];
  
  let arguments = (ins 
    AnyTensor:$lhs,
    AnyTensor:$rhs
  );
  
  let results = (outs 
    AnyTensor:$result
  );
  
  let assemblyFormat = [{
    $lhs `,` $rhs attr-dict `:` type($lhs) `,` type($rhs) `->` type($result)
  }];
}


//===----------------------------------------------------------------------===//
// ReLU Operation
//===----------------------------------------------------------------------===//

def Flash_ReLUOp : Flash_Op<"relu", [Pure]> {
  let summary = "ReLU activation function";
  
  let description = [{
    Applies Rectified Linear Unit (ReLU) activation: f(x) = max(0, x)
    
    Element-wise operation that replaces negative values with zero.
    
    Example:
```mlir
    %activated = flash.relu %input : tensor<4x4xf32> -> tensor<4x4xf32>
```
    
    Mathematical definition:
      relu(x) = x if x > 0
              = 0 if x <= 0
  }];
  
  let arguments = (ins 
    AnyTensor:$input
  );
  
  let results = (outs 
    AnyTensor:$result
  );
  
  let assemblyFormat = [{
    $input attr-dict `:` type($input) `->` type($result)
  }];
}

//===----------------------------------------------------------------------===//
// Conv2D Operation
//===----------------------------------------------------------------------===//

def Flash_Conv2DOp : Flash_Op<"conv2d", [Pure]> {
  let summary = "2D Convolution operation";
  
  let description = [{
    Performs 2D convolution: output = conv2d(input, kernel)
    
    Input format: NCHW (batch, channels, height, width)
    Kernel format: OIHW (out_channels, in_channels, kernel_h, kernel_w)
    
    Example:
```mlir
    %output = flash.conv2d %input, %kernel : tensor<1x3x32x32xf32>, tensor<64x3x3x3xf32> -> tensor<1x64x30x30xf32>
```
    
    Note: This simplified version assumes stride=1, padding=0, dilation=1.
    Full attribute support will be added in later optimization passes.
  }];
  
  let arguments = (ins 
    AnyTensor:$input,
    AnyTensor:$kernel
  );
  
  let results = (outs 
    AnyTensor:$result
  );
  
  let assemblyFormat = [{
    $input `,` $kernel attr-dict `:` type($input) `,` type($kernel) `->` type($result)
  }];
}

//#endif //FLASH_OPS