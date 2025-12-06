cmake_minimum_required(VERSION 3.16)
project(gef 
    VERSION 1.0.0
    DESCRIPTION "GEF - A C++ library"
    LANGUAGES CXX
)

include(GNUInstallDirs)
if(NOT CMAKE_INSTALL_INCLUDEDIR)
    set(CMAKE_INSTALL_INCLUDEDIR "include" CACHE PATH "Installation directory for header files" FORCE)
endif()
set(TLX_INSTALL_INCLUDE_DIR ${CMAKE_INSTALL_INCLUDEDIR} CACHE PATH "Installation directory for TLX headers" FORCE)

# Set a default build type if none was specified
if(NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
  message(STATUS "Setting build type to 'Release' as none was specified.")
  set(CMAKE_BUILD_TYPE Release CACHE STRING "Choose the type of build, options are: Debug Release RelWithDebInfo MinSizeRel." FORCE)
endif()

# Allow override of target architecture for CI builds
# Usage: cmake -DGEF_TARGET_ARCH=skylake-avx512 to target specific CPU
set(GEF_TARGET_ARCH "native" CACHE STRING "Target CPU architecture (native, skylake-avx512, etc.)")

# Option for static linking (useful for CI builds targeting older systems)
option(GEF_STATIC_LINK "Build with static linking" OFF)
option(GEF_DISABLE_SIMD "Force-disable SIMD-specific intrinsics and use scalar fallbacks" OFF)

if(GEF_DISABLE_SIMD)
    message(STATUS "GEF SIMD optimizations disabled (GEF_DISABLE_SIMD=ON)")
endif()

# Add high-performance compiler flags for Release builds
set(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE} -O3 -march=${GEF_TARGET_ARCH} -ffast-math -funroll-loops")

# Additional optimization flags for better vectorization
if(CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
    set(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE} -ftree-vectorize -fvect-cost-model=dynamic")
endif()

# Enable Link-Time Optimization for Release builds (massive performance gain)
if(CMAKE_BUILD_TYPE STREQUAL "Release")
    include(CheckIPOSupported)
    check_ipo_supported(RESULT ipo_supported OUTPUT ipo_error)
    if(ipo_supported)
        message(STATUS "Link-Time Optimization (LTO) enabled")
        set(CMAKE_INTERPROCEDURAL_OPTIMIZATION_RELEASE TRUE)
    else()
        message(WARNING "LTO not supported: ${ipo_error}")
    endif()
endif()

# Apply static linking flags if requested
if(GEF_STATIC_LINK)
    message(STATUS "Static linking enabled")
    set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} -static -static-libstdc++ -static-libgcc")
endif()

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Find and enable OpenMP for parallel processing when available
find_package(OpenMP)
if(OpenMP_CXX_FOUND)
    message(STATUS "OpenMP found - parallel processing enabled")
else()
    message(WARNING "OpenMP not found - UniformedPartitioner will run sequentially")
endif()

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

if(TARGET sdsl)
    # GCC 15 introduces -Wtemplate-body which catches latent bugs in uninstantiated templates.
    # sdsl-lite has a bug in louds_tree::swap (referencing m_select1 instead of m_bv_select1).
    # Since we don't use louds_tree::swap, we can silence this error.
    include(CheckCXXCompilerFlag)
    check_cxx_compiler_flag("-Wno-template-body" CXX_HAS_WNO_TEMPLATE_BODY)
    if(CXX_HAS_WNO_TEMPLATE_BODY)
        target_compile_options(sdsl PRIVATE "-Wno-template-body")
    endif()
endif()

# Use ccache if available to speed up repeated builds
find_program(CCACHE_PROGRAM ccache)
if(CCACHE_PROGRAM)
    message(STATUS "ccache found: ${CCACHE_PROGRAM}")
    set(CMAKE_CXX_COMPILER_LAUNCHER "${CCACHE_PROGRAM}")
    set(CMAKE_C_COMPILER_LAUNCHER "${CCACHE_PROGRAM}")
else()
    message(STATUS "ccache not found. Install it for faster builds.")
endif()

# --- Secondary Dependency: SUX (Header-only library) ---
FetchContent_Declare(
    sux
    GIT_REPOSITORY https://github.com/vigna/sux.git
    GIT_TAG        master
)
FetchContent_MakeAvailable(sux)

# SUX is header-only, so we just need to expose its include directory
if(TARGET sux)
    # If sux creates a target, use it
    set(SUX_INCLUDE_DIR ${sux_SOURCE_DIR})
else()
    # Otherwise, just get the source directory
    FetchContent_GetProperties(sux)
    if(NOT sux_POPULATED)
        FetchContent_Populate(sux)
    endif()
    set(SUX_INCLUDE_DIR ${sux_SOURCE_DIR})
endif()


# --- Pasta (rank/select bitvectors) ---
set(TLX_INSTALL OFF CACHE BOOL "" FORCE)
set(TLX_BUILD_TESTS OFF CACHE BOOL "" FORCE)
set(TLX_BUILD_BENCHMARKS OFF CACHE BOOL "" FORCE)
FetchContent_Declare(
    pasta
    GIT_REPOSITORY https://github.com/pasta-toolbox/bit_vector.git
    GIT_TAG        main
)

FetchContent_MakeAvailable(pasta)
set(PASTA_INCLUDE_DIR ${pasta_SOURCE_DIR}/include)


# --- Project Sources ---
add_subdirectory(src)

# Link Pasta bit-vector implementation
if(TARGET pasta_bit_vector)
    target_link_libraries(gef PUBLIC pasta_bit_vector)
elseif(EXISTS "${PASTA_INCLUDE_DIR}")
    target_include_directories(gef PUBLIC ${PASTA_INCLUDE_DIR})
endif()

# Propagate OpenMP usage through the gef target so dependants can opt-in
if(OpenMP_CXX_FOUND)
    target_link_libraries(gef PUBLIC OpenMP::OpenMP_CXX)
endif()

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

