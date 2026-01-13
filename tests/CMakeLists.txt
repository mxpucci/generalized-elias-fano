file(GLOB TEST_FILES CONFIGURE_DEPENDS "*.cpp")

# Build a separate executable for each test file to avoid ODR violations
# caused by the external 'pasta' library (which has non-inline definitions in headers).
foreach(test_file ${TEST_FILES})
    get_filename_component(test_name ${test_file} NAME_WE)
    
    add_executable(${test_name} ${test_file})
    target_link_libraries(${test_name} PRIVATE gef::gef gtest_main)
    
    # The dataset test requires CLI args and shouldn't be run by CTest automatically
    if(NOT test_name STREQUAL "split_point_strategy_dataset_test")
        add_test(NAME ${test_name} COMMAND ${test_name})
    endif()
endforeach()
