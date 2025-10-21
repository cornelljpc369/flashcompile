# Simple integration test runner (no lit dependency)
add_custom_target(check-flash-integration
  COMMAND ${CMAKE_CURRENT_SOURCE_DIR}/run_tests.sh 
          ${CMAKE_BINARY_DIR}/tools/flash-opt/flash-opt
          ${CMAKE_CURRENT_SOURCE_DIR}
  DEPENDS flash-opt
  WORKING_DIRECTORY ${CMAKE_BINARY_DIR}
  COMMENT "Running Flash integration tests"
)