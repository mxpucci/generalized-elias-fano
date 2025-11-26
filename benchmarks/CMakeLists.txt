file(GLOB BENCHMARK_FILES "*.cpp")

foreach(benchmark_file ${BENCHMARK_FILES})
    get_filename_component(benchmark_name ${benchmark_file} NAME_WE)
    
    # Skip compression_benchmark as we handle it separately below
    if(benchmark_name STREQUAL "compression_benchmark" OR
       benchmark_name STREQUAL "single_thread_compression_throughput")
        continue()
    endif()
    
    add_executable(${benchmark_name} ${benchmark_file})
    
    # Check if it's learned_gef_benchmark to NOT link benchmark::benchmark
    if(benchmark_name STREQUAL "learned_gef_benchmark")
        target_link_libraries(${benchmark_name} PRIVATE gef::gef)
    else()
        target_link_libraries(${benchmark_name} PRIVATE gef::gef benchmark::benchmark)
    endif()
    
    # Ensure benchmarks are always compiled with maximum optimization
    target_compile_options(${benchmark_name} PRIVATE 
        -O3 -march=native -ffast-math -funroll-loops
        $<$<CXX_COMPILER_ID:GNU,Clang>:-ftree-vectorize -fvect-cost-model=dynamic>
    )
    
    # Enable Link-Time Optimization (LTO) for maximum performance
    set_target_properties(${benchmark_name} PROPERTIES
        INTERPROCEDURAL_OPTIMIZATION TRUE
    )
    
    # Disable debug symbols for benchmarks to reduce binary size
    target_compile_options(${benchmark_name} PRIVATE -g0)

    if(OpenMP_CXX_FOUND)
        target_link_libraries(${benchmark_name} PRIVATE OpenMP::OpenMP_CXX)
    endif()
endforeach()

# Special handling for compression_benchmark: create two versions
# 1. compression_benchmark - WITH OpenMP (for parallel decompression)
# 2. compression_benchmark_no_omp - WITHOUT OpenMP (for sequential comparison)

# Version 1: WITH OpenMP (default)
add_executable(compression_benchmark compression_benchmark.cpp)
target_link_libraries(compression_benchmark PRIVATE gef::gef benchmark::benchmark)
if(OpenMP_CXX_FOUND)
    target_link_libraries(compression_benchmark PRIVATE OpenMP::OpenMP_CXX)
endif()
target_compile_options(compression_benchmark PRIVATE 
    -O3 -march=native -ffast-math -funroll-loops
    $<$<CXX_COMPILER_ID:GNU,Clang>:-ftree-vectorize -fvect-cost-model=dynamic>
)
set_target_properties(compression_benchmark PROPERTIES
    INTERPROCEDURAL_OPTIMIZATION TRUE
)
target_compile_options(compression_benchmark PRIVATE -g0)

# If OpenMP was found, add a compile definition to make it clear this version has OpenMP
if(OpenMP_CXX_FOUND)
    target_compile_definitions(compression_benchmark PRIVATE GEF_BENCHMARK_WITH_OPENMP=1)
endif()

# Version 2: WITHOUT OpenMP (sequential)
# NOTE: Since gef links OpenMP as PUBLIC, we must explicitly disable OpenMP at compile time
# to prevent the parallel code paths from being taken.
add_executable(compression_benchmark_no_omp compression_benchmark.cpp)
target_link_libraries(compression_benchmark_no_omp PRIVATE gef::gef benchmark::benchmark)
target_compile_options(compression_benchmark_no_omp PRIVATE 
    -O3 -march=native -ffast-math -funroll-loops
    $<$<CXX_COMPILER_ID:GNU,Clang>:-ftree-vectorize -fvect-cost-model=dynamic>
)
set_target_properties(compression_benchmark_no_omp PROPERTIES
    INTERPROCEDURAL_OPTIMIZATION TRUE
)
target_compile_options(compression_benchmark_no_omp PRIVATE -g0)
target_compile_definitions(compression_benchmark_no_omp PRIVATE GEF_BENCHMARK_WITH_OPENMP=0)

# CRITICAL: Explicitly disable OpenMP compilation to override the PUBLIC linkage from gef
# This undefines _OPENMP and disables OpenMP pragmas at compile time
if(OpenMP_CXX_FOUND)
    target_compile_definitions(compression_benchmark_no_omp PRIVATE _OPENMP=0)
    if(CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
        target_compile_options(compression_benchmark_no_omp PRIVATE -fno-openmp -Wno-unknown-pragmas)
    endif()
endif()

if(OpenMP_CXX_FOUND)
    message(STATUS "Building two compression_benchmark versions:")
    message(STATUS "  - compression_benchmark (WITH OpenMP)")
    message(STATUS "  - compression_benchmark_no_omp (WITHOUT OpenMP)")
else()
    message(STATUS "OpenMP not found - both benchmark versions will be sequential")
endif() 

# ---------------------------------------------------------------------------
# Single-thread compression throughput measurement executables
# ---------------------------------------------------------------------------

function(configure_single_thread_target target_name)
    target_link_libraries(${target_name} PRIVATE gef::gef)
    target_compile_options(${target_name} PRIVATE
        -O3 -march=native -ffast-math -funroll-loops
        $<$<CXX_COMPILER_ID:GNU,Clang>:-ftree-vectorize -fvect-cost-model=dynamic>
        -g0
    )
    set_target_properties(${target_name} PROPERTIES
        INTERPROCEDURAL_OPTIMIZATION TRUE
    )
    if(OpenMP_CXX_FOUND)
        target_compile_definitions(${target_name} PRIVATE _OPENMP=0)
        if(CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
            target_compile_options(${target_name} PRIVATE -fno-openmp -Wno-unknown-pragmas)
        endif()
    endif()
endfunction()

add_executable(single_thread_compression_throughput_simd single_thread_compression_throughput.cpp)
configure_single_thread_target(single_thread_compression_throughput_simd)

add_executable(single_thread_compression_throughput single_thread_compression_throughput.cpp)
configure_single_thread_target(single_thread_compression_throughput)
target_compile_definitions(single_thread_compression_throughput PRIVATE GEF_DISABLE_SIMD)
