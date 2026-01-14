cmake_minimum_required(VERSION 3.14)

# Ensure Release build type if not specified
if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release)
endif()

add_executable(compression_benchmark bench_compression.cpp)
target_link_libraries(compression_benchmark PRIVATE gef benchmark::benchmark OpenMP::OpenMP_CXX)
# Explicitly add optimization flags to be sure, in case they are not propagated or overridden
target_compile_options(compression_benchmark PRIVATE -O3 -march=native)

add_executable(random_access_benchmark bench_random_access.cpp)
target_link_libraries(random_access_benchmark PRIVATE gef benchmark::benchmark OpenMP::OpenMP_CXX)
target_compile_options(random_access_benchmark PRIVATE -O3 -march=native)

add_executable(decompression_benchmark bench_decompression.cpp)
target_link_libraries(decompression_benchmark PRIVATE gef benchmark::benchmark OpenMP::OpenMP_CXX)
target_compile_options(decompression_benchmark PRIVATE -O3 -march=native)
