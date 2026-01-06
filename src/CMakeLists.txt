add_library(gef gef.cpp)
add_library(gef::gef ALIAS gef)

# Set library properties
set_target_properties(gef PROPERTIES
    VERSION ${PROJECT_VERSION}
    SOVERSION ${PROJECT_VERSION_MAJOR}
    EXPORT_NAME gef
)

target_include_directories(gef
    PUBLIC
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/../include>
        $<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}>
)

# Link against sdsl publicly so consumers get access to sdsl functionality
# This is appropriate for a library that exposes sdsl types in its interface
target_link_libraries(gef PUBLIC sdsl)

# SUX is header-only, so we just need to add its include directory
target_include_directories(gef PUBLIC ${SUX_INCLUDE_DIR})

# Precompile common headers to speed up compilation
target_precompile_headers(gef PUBLIC
    <algorithm>
    <cmath>
    <filesystem>
    <fstream>
    <iostream>
    <memory>
    <string>
    <vector>
    <utility>
    <type_traits>
    <sdsl/int_vector.hpp>
)

# Apply maximum optimization to the library itself
# This is crucial since most GEF code is header-only templates
target_compile_options(gef PUBLIC 
    $<$<CONFIG:Release>:-O3 -march=native -ffast-math -funroll-loops>
    $<$<AND:$<CONFIG:Release>,$<CXX_COMPILER_ID:GNU,Clang>>:-ftree-vectorize -fvect-cost-model=dynamic>
)
# Enable Link-Time Optimization (LTO) in Release builds
set_target_properties(gef PROPERTIES
    INTERPROCEDURAL_OPTIMIZATION_RELEASE TRUE
)

if(GEF_DISABLE_SIMD)
    target_compile_definitions(gef PUBLIC GEF_DISABLE_SIMD)
endif()

# For Debug builds, ensure reasonable performance while keeping debuggability
target_compile_options(gef PUBLIC 
    $<$<CONFIG:Debug>:-O1 -g>
)

add_executable(ugef ugef.cpp)
target_compile_options(ugef PRIVATE -D_FILE_OFFSET_BITS=64)
target_link_libraries(ugef PRIVATE gef)

add_executable(rle_gef rle_gef.cpp)
target_compile_options(rle_gef PRIVATE -D_FILE_OFFSET_BITS=64)
target_link_libraries(rle_gef PRIVATE gef)

add_executable(bgef bgef.cpp)
target_compile_options(bgef PRIVATE -D_FILE_OFFSET_BITS=64)
target_link_libraries(bgef PRIVATE gef)

add_executable(bgef_no_rle bgef_no_rle.cpp)
target_compile_options(bgef_no_rle PRIVATE -D_FILE_OFFSET_BITS=64)
target_link_libraries(bgef_no_rle PRIVATE gef)

