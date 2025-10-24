#!/bin/bash
#
# flash-compile-and-run.sh - Compile and execute Flash IR in one command
#
# Usage: ./tools/flash-compile-and-run.sh <input.mlir>
#

set -e  # Exit on any error

# Parse arguments
INPUT_FILE=$1
VERBOSE=${VERBOSE:-0}

if [ -z "$INPUT_FILE" ]; then
  echo "Usage: $0 <input.mlir>"
  echo ""
  echo "Example:"
  echo "  $0 /tmp/test.mlir"
  echo "  VERBOSE=1 $0 /tmp/test.mlir    # Show LLVM IR"
  exit 1
fi

# Check if input file exists
if [ ! -f "$INPUT_FILE" ]; then
  echo "Error: File '$INPUT_FILE' not found"
  exit 1
fi

# Create temporary directory
TMP_DIR=$(mktemp -d)

# Cleanup on exit
cleanup() {
  rm -rf "$TMP_DIR"
}
trap cleanup EXIT

# Output files
LLVM_DIALECT="$TMP_DIR/lowered.mlir"
LLVM_IR="$TMP_DIR/output.ll"

# Colors
GREEN='\033[0;32m'
BLUE='\033[0;34m'
YELLOW='\033[1;33m'
NC='\033[0m'

echo -e "${BLUE}════════════════════════════════════════${NC}"
echo -e "${BLUE}   FlashCompile: Compile & Execute      ${NC}"
echo -e "${BLUE}════════════════════════════════════════${NC}"
echo ""

# Get script directory (where this script is)
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
PROJECT_ROOT="$( cd "$SCRIPT_DIR/.." && pwd )"

# Paths to tools
FLASH_OPT="$PROJECT_ROOT/build/tools/flash-opt/flash-opt"
MLIR_TRANSLATE="mlir-translate"
LLI="lli"

# Check if tools exist
if [ ! -f "$FLASH_OPT" ]; then
  echo "Error: flash-opt not found at: $FLASH_OPT"
  echo "Please build the project first: cd build && cmake --build ."
  exit 1
fi

if ! command -v mlir-translate &> /dev/null; then
  echo "Error: mlir-translate not found in PATH"
  echo "Make sure LLVM is installed: brew install llvm"
  exit 1
fi

if ! command -v lli &> /dev/null; then
  echo "Error: lli not found in PATH"
  exit 1
fi

# Step 1: Lower to LLVM Dialect
echo -e "${YELLOW}[1/3]${NC} Lowering to LLVM dialect..."

"$FLASH_OPT" "$INPUT_FILE" \
  --convert-flash-to-linalg \
  --one-shot-bufferize \
  --convert-linalg-to-loops \
  --lower-affine \
  --convert-scf-to-cf \
  --convert-cf-to-llvm \
  --convert-arith-to-llvm \
  --convert-func-to-llvm \
  --finalize-memref-to-llvm \
  --reconcile-unrealized-casts \
  -o "$LLVM_DIALECT" 2>&1

if [ $? -ne 0 ]; then
  echo -e "${RED}❌ Lowering failed${NC}"
  exit 1
fi

echo -e "${GREEN}✅ Lowered to LLVM dialect${NC}"

# Step 2: Translate to LLVM IR
echo -e "${YELLOW}[2/3]${NC} Translating to LLVM IR..."

"$MLIR_TRANSLATE" --mlir-to-llvmir "$LLVM_DIALECT" -o "$LLVM_IR" 2>&1

if [ $? -ne 0 ]; then
  echo -e "${RED}❌ Translation failed${NC}"
  exit 1
fi

echo -e "${GREEN}✅ Generated LLVM IR${NC}"

# Show LLVM IR if verbose
if [ "$VERBOSE" = "1" ]; then
  echo ""
  echo -e "${BLUE}════════ LLVM IR ════════${NC}"
  cat "$LLVM_IR"
  echo -e "${BLUE}═════════════════════════${NC}"
  echo ""
fi

# Step 3: Execute
echo -e "${YELLOW}[3/3]${NC} Executing..."
echo ""
echo -e "${BLUE}────────── OUTPUT ──────────${NC}"

# Try to find runtime library
RUNTIME_LIB="/opt/homebrew/Cellar/llvm/21.1.1/lib/libmlir_runner_utils.dylib"

if [ -f "$RUNTIME_LIB" ]; then
  "$LLI" -load="$RUNTIME_LIB" "$LLVM_IR"
  EXIT_CODE=$?
else
  echo -e "${YELLOW}Warning: Runtime library not found, running without print support${NC}"
  "$LLI" "$LLVM_IR"
  EXIT_CODE=$?
fi

echo -e "${BLUE}────────────────────────────${NC}"
echo ""

if [ $EXIT_CODE -eq 0 ]; then
  echo -e "${GREEN}✅ Execution completed successfully!${NC}"
else
  echo -e "${YELLOW}⚠️  Execution completed with exit code: $EXIT_CODE${NC}"
fi

echo ""
echo -e "${BLUE}════════════════════════════════════════${NC}"

exit $EXIT_CODE
