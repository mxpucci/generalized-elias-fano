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
set(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE} -O3")

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

include(FetchContent)

# Determine if this is the main project or a subproject
set(GEF_IS_MAIN_PROJECT OFF)
if(CMAKE_CURRENT_SOURCE_DIR STREQUAL CMAKE_SOURCE_DIR)
    set(GEF_IS_MAIN_PROJECT ON)
endif()

# --- Primary Dependency: sdsl-lite ---
set(SDSL_BUILD_TESTS OFF CACHE BOOL "Build tests for sdsl-lite" FORCE)
set(SDSL_BUILD_DIVSUFSORT ON  CACHE BOOL "Build bundled libdivsufsort" FORCE)

FetchContent_Declare(
        sdsl-lite
        GIT_REPOSITORY https://github.com/simongog/sdsl-lite.git
        GIT_TAG        master
)
FetchContent_MakeAvailable(sdsl-lite)

# Remove any brittle hard-coded includes you had for sdsl/libdivsufsort
# target_include_directories(sdsl INTERFACE ${sdsl-lite_SOURCE_DIR}/external/libdivsufsort/include)

# Discover divsufsort.h location regardless of how sdsl-lite brought it in
set(_divsufsort_include "")

# 1) If divsufsort targets exist and expose include dirs, use them
foreach(tgt divsufsort divsufsort64)
    if(TARGET ${tgt})
        get_target_property(_inc ${tgt} INTERFACE_INCLUDE_DIRECTORIES)
        if(_inc)
            list(GET _inc 0 _divsufsort_include)
            break()
        endif()
    endif()
endforeach()

# 2) Try common sdsl-lite layouts (submodule or ExternalProject)
if(NOT _divsufsort_include)
    foreach(dir
            ${sdsl-lite_SOURCE_DIR}/external/libdivsufsort/include
            ${sdsl-lite_BINARY_DIR}/external/libdivsufsort/include
            ${sdsl-lite_BINARY_DIR}/external/libdivsufsort/src/libdivsufsort/include
    )
        if(EXISTS "${dir}/divsufsort.h")
            set(_divsufsort_include "${dir}")
            break()
        endif()
    endforeach()
endif()

# 3) Fallback to Homebrew on Apple Silicon (M1/M2)
if(APPLE AND NOT _divsufsort_include)
    foreach(dir /opt/homebrew/include /usr/local/include)
        if(EXISTS "${dir}/divsufsort.h")
            set(_divsufsort_include "${dir}")
            break()
        endif()
    endforeach()
endif()

if(NOT _divsufsort_include)
    message(FATAL_ERROR "Could not find divsufsort.h. Either let sdsl-lite fetch libdivsufsort, or `brew install libdivsufsort`.")
endif()

# Propagate the header path via sdsl so all consumers inherit it
target_include_directories(sdsl INTERFACE "${_divsufsort_include}")

# Also ensure link libraries propagate (harmless if already linked)
if(TARGET divsufsort)
    target_link_libraries(sdsl INTERFACE divsufsort)
endif()
if(TARGET divsufsort64)
    target_link_libraries(sdsl INTERFACE divsufsort64)
endif()
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