file(GLOB TEST_FILES "*.cpp")

foreach(test_file ${TEST_FILES})
    get_filename_component(test_name ${test_file} NAME_WE)
    add_executable(${test_name} ${test_file})
    target_link_libraries(${test_name} PRIVATE gef::gef gtest_main)
    add_test(NAME ${test_name} COMMAND ${test_name})
endforeach() 