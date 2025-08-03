cmake_minimum_required(VERSION 3.14)
project(gef 
    VERSION 1.0.0
    DESCRIPTION "GEF - A C++ library"
    LANGUAGES CXX
)

# Set a default build type if none was specified
if(NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
  message(STATUS "Setting build type to 'Release' as none was specified.")
  set(CMAKE_BUILD_TYPE Release CACHE STRING "Choose the type of build, options are: Debug Release RelWithDebInfo MinSizeRel." FORCE)
endif()

# Add high-performance compiler flags for Release builds
set(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE} -O3 -march=native")

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

include(FetchContent)

# Determine if this is the main project or a subproject
set(GEF_IS_MAIN_PROJECT OFF)
if(CMAKE_CURRENT_SOURCE_DIR STREQUAL CMAKE_SOURCE_DIR)
    set(GEF_IS_MAIN_PROJECT ON)
endif()

# --- Primary Dependency: sdsl-lite ---
# Set sdsl-lite's test option to OFF before we do anything with it.
set(SDSL_BUILD_TESTS OFF CACHE BOOL "Build tests for sdsl-lite" FORCE)

FetchContent_Declare(
    sdsl-lite
    GIT_REPOSITORY https://github.com/simongog/sdsl-lite.git
    GIT_TAG        master # It is recommended to use a specific tag/commit
)
FetchContent_MakeAvailable(sdsl-lite)

# --- Project Sources ---
add_subdirectory(src)

# --- Tests Configuration ---
option(GEF_BUILD_TESTS "Build tests for gef" ${GEF_IS_MAIN_PROJECT})
if(GEF_BUILD_TESTS)
    enable_testing()
    add_subdirectory(tests)
endif()

# --- Benchmarks Configuration ---
option(GEF_BUILD_BENCHMARKS "Build benchmarks for gef" ${GEF_IS_MAIN_PROJECT})
if(GEF_BUILD_BENCHMARKS)
    FetchContent_Declare(
        googlebenchmark
        URL https://github.com/google/benchmark/archive/refs/tags/v1.8.3.zip
    )
    set(BENCHMARK_ENABLE_INSTALL OFF CACHE BOOL "Disable benchmark installation" FORCE)
    set(BENCHMARK_ENABLE_GTEST_TESTS OFF CACHE BOOL "Disable Google Test in benchmark" FORCE)
    set(BENCHMARK_ENABLE_TESTING OFF CACHE BOOL "Disable benchmark testing" FORCE)
    FetchContent_MakeAvailable(googlebenchmark)
    add_subdirectory(benchmarks)
endif() 