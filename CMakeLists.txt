cmake_minimum_required(VERSION 3.20)
project(flashcompile LANGUAGES CXX C)

# C++ standard
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED YES)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

# Find LLVM and MLIR
find_package(MLIR REQUIRED CONFIG)
find_package(LLVM REQUIRED CONFIG)

message(STATUS "Using MLIRConfig.cmake in: ${MLIR_DIR}")
message(STATUS "Using LLVMConfig.cmake in: ${LLVM_DIR}")
message(STATUS "Using LLVM version: ${LLVM_PACKAGE_VERSION}")

# Add LLVM/MLIR modules
list(APPEND CMAKE_MODULE_PATH "${MLIR_CMAKE_DIR}")
list(APPEND CMAKE_MODULE_PATH "${LLVM_CMAKE_DIR}")

include(TableGen)
include(AddLLVM)
include(AddMLIR)
include(HandleLLVMOptions)

# Include directories
include_directories(${LLVM_INCLUDE_DIRS})
include_directories(${MLIR_INCLUDE_DIRS})
include_directories(${PROJECT_SOURCE_DIR}/include)
include_directories(${PROJECT_BINARY_DIR}/include)

# Link directories
link_directories(${LLVM_BUILD_LIBRARY_DIR})
add_definitions(${LLVM_DEFINITIONS})

# Enable testing
enable_testing()
include(CTest)

# Find GTest from LLVM installation
# LLVM bundles its own GTest - use that instead of system GTest
set(LLVM_GTEST_INCLUDE_DIR "${LLVM_INCLUDE_DIRS}/../../src/utils/unittest/googletest/include")
set(LLVM_GTEST_LIB_DIR "${LLVM_LIBRARY_DIRS}")

# Try to find GTest libraries in LLVM's lib directory
find_library(GTEST_LIBRARY 
  NAMES gtest libgtest
  PATHS ${LLVM_LIBRARY_DIRS}
  NO_DEFAULT_PATH
)

find_library(GTEST_MAIN_LIBRARY 
  NAMES gtest_main libgtest_main
  PATHS ${LLVM_LIBRARY_DIRS}
  NO_DEFAULT_PATH
)

# Fallback: Use LLVM's bundled gtest if separate libraries not found
if(NOT GTEST_LIBRARY OR NOT GTEST_MAIN_LIBRARY)
  message(STATUS "Using LLVM's bundled gtest")
  # On macOS with Homebrew LLVM, gtest is typically linked into LLVM libraries
  # We'll link against LLVMSupport which includes gtest symbols
  set(GTEST_LIBRARY "")
  set(GTEST_MAIN_LIBRARY "")
  set(LLVM_HAS_GTEST TRUE)
else()
  message(STATUS "Found GTest: ${GTEST_LIBRARY}")
  set(LLVM_HAS_GTEST FALSE)
endif()

# Subdirectories
add_subdirectory(include)
add_subdirectory(lib)
add_subdirectory(tools)
add_subdirectory(test)