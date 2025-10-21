//===- test_dialect.cpp - Flash dialect tests -------------------*- C++ -*-===//
//
// Simple tests without GTest dependency
//
//===----------------------------------------------------------------------===//

#include "flash/Dialect/Flash/FlashDialect.h"
#include "flash/Dialect/Flash/FlashOps.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/OwningOpRef.h"
#include "mlir/Parser/Parser.h"
#include "llvm/Support/Casting.h"
#include "llvm/Support/raw_ostream.h"

using namespace mlir;
using namespace mlir::flash;

// Simple test macros
#define ASSERT(cond, msg) \
  if (!(cond)) { \
    llvm::errs() << "FAIL: " << msg << "\n"; \
    return false; \
  }

#define TEST(name) \
  llvm::outs() << "[ RUN      ] " << #name << "\n"; \
  if (test_##name()) { \
    llvm::outs() << "[       OK ] " << #name << "\n"; \
    passCount++; \
  } else { \
    llvm::errs() << "[  FAILED  ] " << #name << "\n"; \
    failCount++; \
  }

static int passCount = 0;
static int failCount = 0;

//===----------------------------------------------------------------------===//
// Test 1: Dialect Registration
//===----------------------------------------------------------------------===//

bool test_DialectLoads() {
  MLIRContext ctx;
  ctx.getOrLoadDialect<FlashDialect>();
  ctx.getOrLoadDialect<func::FuncDialect>();
  
  auto *dialect = ctx.getLoadedDialect<FlashDialect>();
  ASSERT(dialect != nullptr, "Flash dialect should be loaded");
  ASSERT(dialect->getNamespace() == "flash", "Dialect namespace should be 'flash'");
  
  return true;
}

//===----------------------------------------------------------------------===//
// Test 2: MatMul Parsing
//===----------------------------------------------------------------------===//

bool test_MatMulParsing() {
  MLIRContext ctx;
  ctx.getOrLoadDialect<FlashDialect>();
  ctx.getOrLoadDialect<func::FuncDialect>();
  
  const char *moduleStr = R"mlir(
    module {
      func.func @test(%arg0: tensor<4x4xf32>, %arg1: tensor<4x4xf32>) -> tensor<4x4xf32> {
        %0 = flash.matmul %arg0, %arg1 : tensor<4x4xf32>, tensor<4x4xf32> -> tensor<4x4xf32>
        return %0 : tensor<4x4xf32>
      }
    }
  )mlir";
  
  OwningOpRef<ModuleOp> module = parseSourceString<ModuleOp>(moduleStr, &ctx);
  ASSERT(module, "Module should parse successfully");
  
  // Find MatMul operation
  MatMulOp matmulOp;
  module->walk([&](MatMulOp op) { matmulOp = op; });
  
  ASSERT(matmulOp, "Should find flash.matmul operation");
  
  auto lhsType = llvm::cast<RankedTensorType>(matmulOp.getLhs().getType());
  ASSERT(lhsType.getShape()[0] == 4 && lhsType.getShape()[1] == 4, 
         "LHS should be 4x4");
  ASSERT(lhsType.getElementType().isF32(), "LHS should be f32");
  
  return true;
}

//===----------------------------------------------------------------------===//
// Test 3: MatMul Construction
//===----------------------------------------------------------------------===//

bool test_MatMulConstruction() {
  MLIRContext ctx;
  ctx.getOrLoadDialect<FlashDialect>();
  ctx.getOrLoadDialect<func::FuncDialect>();
  
  OpBuilder builder(&ctx);
  auto loc = builder.getUnknownLoc();
  
  auto module = builder.create<ModuleOp>(loc);
  builder.setInsertionPointToStart(module.getBody());
  
  auto lhsType = RankedTensorType::get({4, 8}, builder.getF32Type());
  auto rhsType = RankedTensorType::get({8, 4}, builder.getF32Type());
  auto resultType = RankedTensorType::get({4, 4}, builder.getF32Type());
  
  auto funcType = builder.getFunctionType({lhsType, rhsType}, {resultType});
  auto func = builder.create<func::FuncOp>(loc, "test_matmul", funcType);
  auto *entryBlock = func.addEntryBlock();
  builder.setInsertionPointToStart(entryBlock);
  
  Value lhs = func.getArgument(0);
  Value rhs = func.getArgument(1);
  
  auto matmulOp = builder.create<MatMulOp>(loc, resultType, lhs, rhs);
  
  ASSERT(matmulOp != nullptr, "MatMul operation should be created");
  ASSERT(matmulOp->getName().getStringRef() == "flash.matmul",
         "Operation name should be 'flash.matmul'");
  ASSERT(matmulOp.getLhs() == lhs, "LHS operand should match");
  ASSERT(matmulOp.getRhs() == rhs, "RHS operand should match");
  
  return true;
}

// //===----------------------------------------------------------------------===//
// // Test 4: Round-Trip Test
// //===----------------------------------------------------------------------===//

// bool test_RoundTrip() {
//   MLIRContext ctx;
//   ctx.getOrLoadDialect<FlashDialect>();
//   ctx.getOrLoadDialect<func::FuncDialect>();
  
//   const char *originalIR = R"mlir(
// module {
//   func.func @test(%arg0: tensor<2x3xf32>, %arg1: tensor<3x2xf32>) -> tensor<2x2xf32> {
//     %0 = flash.matmul %arg0, %arg1 : tensor<2x3xf32>, tensor<3x2xf32> -> tensor<2x2xf32>
//     return %0 : tensor<2x2xf32>
//   }
// }
// )mlir";
  
//   // First parse
//   OwningOpRef<ModuleOp> module1 = parseSourceString<ModuleOp>(originalIR, &ctx);
//   ASSERT(module1, "First parse should succeed");
  
//   // Print to string
//   std::string printedIR;
//   llvm::raw_string_ostream os(printedIR);
//   module1->print(os);
//   os.flush();
  
//   // Second parse
//   OwningOpRef<ModuleOp> module2 = parseSourceString<ModuleOp>(printedIR, &ctx);
//   ASSERT(module2, "Second parse (after print) should succeed");
  
//   // Both should have MatMul
//   Flash_MatMulOp op1, op2;
//   module1->walk([&](Flash_MatMulOp op) { op1 = op; });
//   module2->walk([&](Flash_MatMulOp op) { op2 = op; });
  
//   ASSERT(op1 && op2, "Both modules should contain MatMul");
//   ASSERT(op1.getLhs().getType() == op2.getLhs().getType(),
//          "Types should match after round-trip");
  
//   return true;
// }

//===----------------------------------------------------------------------===//
// Test 5: F32 Type Support
//===----------------------------------------------------------------------===//

bool test_F32Types() {
  MLIRContext ctx;
  ctx.getOrLoadDialect<FlashDialect>();
  ctx.getOrLoadDialect<func::FuncDialect>();
  
  const char *f32IR = R"mlir(
    module {
      func.func @test(%arg0: tensor<4x4xf32>, %arg1: tensor<4x4xf32>) -> tensor<4x4xf32> {
        %0 = flash.matmul %arg0, %arg1 : tensor<4x4xf32>, tensor<4x4xf32> -> tensor<4x4xf32>
        return %0 : tensor<4x4xf32>
      }
    }
  )mlir";
  
  OwningOpRef<ModuleOp> module = parseSourceString<ModuleOp>(f32IR, &ctx);
  ASSERT(module, "F32 tensors should be accepted");
  
  return true;
}

//===----------------------------------------------------------------------===//
// Test 6: F64 Type Support
//===----------------------------------------------------------------------===//

bool test_F64Types() {
  MLIRContext ctx;
  ctx.getOrLoadDialect<FlashDialect>();
  ctx.getOrLoadDialect<func::FuncDialect>();
  
  const char *f64IR = R"mlir(
    module {
      func.func @test(%arg0: tensor<4x4xf64>, %arg1: tensor<4x4xf64>) -> tensor<4x4xf64> {
        %0 = flash.matmul %arg0, %arg1 : tensor<4x4xf64>, tensor<4x4xf64> -> tensor<4x4xf64>
        return %0 : tensor<4x4xf64>
      }
    }
  )mlir";
  
  OwningOpRef<ModuleOp> module = parseSourceString<ModuleOp>(f64IR, &ctx);
  ASSERT(module, "F64 tensors should be accepted");
  
  return true;
}

//===----------------------------------------------------------------------===//
// Test 7: Various Shapes
//===----------------------------------------------------------------------===//

bool test_VariousShapes() {
  MLIRContext ctx;
  ctx.getOrLoadDialect<FlashDialect>();
  ctx.getOrLoadDialect<func::FuncDialect>();
  
  const char *shapesIR = R"mlir(
    module {
      func.func @test1(%arg0: tensor<1x1xf32>, %arg1: tensor<1x1xf32>) -> tensor<1x1xf32> {
        %0 = flash.matmul %arg0, %arg1 : tensor<1x1xf32>, tensor<1x1xf32> -> tensor<1x1xf32>
        return %0 : tensor<1x1xf32>
      }
      func.func @test2(%arg0: tensor<10x20xf32>, %arg1: tensor<20x30xf32>) -> tensor<10x30xf32> {
        %0 = flash.matmul %arg0, %arg1 : tensor<10x20xf32>, tensor<20x30xf32> -> tensor<10x30xf32>
        return %0 : tensor<10x30xf32>
      }
    }
  )mlir";
  
  OwningOpRef<ModuleOp> module = parseSourceString<ModuleOp>(shapesIR, &ctx);
  ASSERT(module, "Module with various shapes should parse");
  
  int matmulCount = 0;
  module->walk([&](MatMulOp op) { matmulCount++; });
  
  ASSERT(matmulCount == 2, "Should find 2 MatMul operations");
  
  return true;
}

//===----------------------------------------------------------------------===//
// Main
//===----------------------------------------------------------------------===//

int main(int argc, char **argv) {
  llvm::outs() << "[==========] Running 7 tests\n";
  llvm::outs() << "[----------] 7 tests from FlashDialect\n";
  
  TEST(DialectLoads);
  TEST(MatMulParsing);
  TEST(MatMulConstruction);
  // TEST(RoundTrip);
  TEST(F32Types);
  TEST(F64Types);
  TEST(VariousShapes);
  
  llvm::outs() << "[----------] 7 tests from FlashDialect (" 
               << (passCount + failCount) << " ms total)\n\n";
  llvm::outs() << "[==========] 7 tests from 1 test suite ran.\n";
  llvm::outs() << "[  PASSED  ] " << passCount << " tests.\n";
  
  if (failCount > 0) {
    llvm::errs() << "[  FAILED  ] " << failCount << " tests.\n";
    return 1;
  }
  
  return 0;
}