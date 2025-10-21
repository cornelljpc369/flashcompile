//===- FlashDialect.td - Flash dialect definition--------------------------===//
//
// Defines the Flash dialect
//
//===----------------------------------------------------------------------===//

//#ifndef FLASH_DIALECT
//#define FLASH_DIALECT


//include mlir base op
include "mlir/IR/OpBase.td"


//===----------------------------------------------------------------------===//
// Flash Dialect Definition
//===----------------------------------------------------------------------===//

def Flash_Dialect : Dialect {
    // dialect namespace
    let name = "flash";

    let summary = "Flash ML compiler dialect";
    
    let description = [{
    The Flash dialect contains high-level ML operations like
    matmul, conv2d, and activation functions.
    
    This is the top-level IR in our compiler.
  }];

    // C++ namespace for generated code
    let cppNamespace = "::mlir::flash";

}


//===----------------------------------------------------------------------===//
// Base Operation Definition
//===----------------------------------------------------------------------===//

// Base class for all Flash operations
// Every op we define will inherit from this

class Flash_Op<string mnemonic, list<Trait> traits = []> :
    Op<Flash_Dialect, mnemonic, traits>;

//#endif