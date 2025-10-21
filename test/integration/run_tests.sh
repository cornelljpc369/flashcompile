#!/bin/bash

# Simple integration test runner without lit

FLASH_OPT="$1"
TEST_DIR="$2"

if [ -z "$FLASH_OPT" ] || [ -z "$TEST_DIR" ]; then
  echo "Usage: $0 <flash-opt-path> <test-dir>"
  exit 1
fi

echo "Running integration tests..."
echo "Tool: $FLASH_OPT"
echo "Tests: $TEST_DIR"
echo ""

passed=0
failed=0

for test in "$TEST_DIR"/*.mlir; do
  testname=$(basename "$test")
  echo -n "Testing $testname... "
  
  if "$FLASH_OPT" "$test" > /dev/null 2>&1; then
    echo "PASSED"
    ((passed++))
  else
    echo "FAILED"
    ((failed++))
  fi
done

echo ""
echo "Results: $passed passed, $failed failed"

if [ $failed -gt 0 ]; then
  exit 1
fi