add_subdirectory(unit)
add_subdirectory(integration)
# Optional: add a target to run all tests
add_custom_target(check-flash
  COMMAND ${CMAKE_CTEST_COMMAND} --output-on-failure
  DEPENDS test_dialect check-flash-integration
  COMMENT "Running all Flash tests"
)