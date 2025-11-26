cmake_minimum_required(VERSION 3.14)
project(gef_examples)

set(CMAKE_CXX_STANDARD 17)

# Example 1: Using installed library
find_package(gef QUIET)
if(gef_FOUND)
    add_executable(basic_usage_installed basic_usage.cpp)
    target_link_libraries(basic_usage_installed PRIVATE gef::gef)
endif()

# Example 2: Using as subdirectory (assuming parent project includes gef)
if(TARGET gef::gef)
    add_executable(basic_usage_subdir basic_usage.cpp)
    target_link_libraries(basic_usage_subdir PRIVATE gef::gef)

    # Graph compression example
    add_executable(graph_example graph_example.cpp)
    target_link_libraries(graph_example PRIVATE gef::gef)
endif() 