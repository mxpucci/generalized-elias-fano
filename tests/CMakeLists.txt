file(GLOB TEST_SOURCES "*.cpp")

# Remove the test that requires custom CLI arguments and has its own main
list(REMOVE_ITEM TEST_SOURCES "${CMAKE_CURRENT_SOURCE_DIR}/split_point_strategy_dataset_test.cpp")

# Consolidate standard tests into a single executable
add_executable(gef_tests ${TEST_SOURCES})
target_link_libraries(gef_tests PRIVATE gef::gef gtest_main)
# Reuse PCH from gef to speed up compilation
target_precompile_headers(gef_tests REUSE_FROM gef)
# Enable Unity Build for tests (massive speedup for many small test files)
set_target_properties(gef_tests PROPERTIES UNITY_BUILD ON)
add_test(NAME gef_tests COMMAND gef_tests)

# Build the dataset strategy test separately
add_executable(split_point_strategy_dataset_test split_point_strategy_dataset_test.cpp)
target_link_libraries(split_point_strategy_dataset_test PRIVATE gef::gef gtest_main)
target_precompile_headers(split_point_strategy_dataset_test REUSE_FROM gef)
# No add_test because it requires --dataset arguments
