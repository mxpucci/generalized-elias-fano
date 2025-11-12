file(GLOB BENCHMARK_FILES "*.cpp")

foreach(benchmark_file ${BENCHMARK_FILES})
    get_filename_component(benchmark_name ${benchmark_file} NAME_WE)
    
    # Skip compression_benchmark as we handle it separately below
    if(benchmark_name STREQUAL "compression_benchmark" OR
       benchmark_name STREQUAL "single_thread_compression_throughput")
        continue()
    endif()
    
    add_executable(${benchmark_name} ${benchmark_file})
    target_link_libraries(${benchmark_name} PRIVATE gef::gef benchmark::benchmark)
    
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
endforeach()

# Special handling for compression_benchmark: create two versions
# 1. compression_benchmark - WITH OpenMP (for parallel decompression)
# 2. compression_benchmark_no_omp - WITHOUT OpenMP (for sequential comparison)

# Version 1: WITH OpenMP (default)
add_executable(compression_benchmark compression_benchmark.cpp)
target_link_libraries(compression_benchmark PRIVATE gef::gef benchmark::benchmark)
target_compile_options(compression_benchmark PRIVATE 
    -O3 -march=native -ffast-math -funroll-loops
    $<$<CXX_COMPILER_ID:GNU,Clang>:-ftree-vectorize -fvect-cost-model=dynamic>
)
set_target_properties(compression_benchmark PROPERTIES
    INTERPROCEDURAL_OPTIMIZATION TRUE
)
target_compile_options(compression_benchmark PRIVATE -g0)

# If OpenMP was found, it's already linked via CMAKE_CXX_FLAGS
# Add a compile definition to make it clear this version has OpenMP
if(OpenMP_CXX_FOUND)
    target_compile_definitions(compression_benchmark PRIVATE GEF_BENCHMARK_WITH_OPENMP=1)
endif()

# Version 2: WITHOUT OpenMP (sequential)
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

# Explicitly remove OpenMP flags and undefine _OPENMP for the no-omp version
if(OpenMP_CXX_FOUND)
    # Remove OpenMP compile flags
    target_compile_options(compression_benchmark_no_omp PRIVATE ${OpenMP_CXX_FLAGS})
    # The above adds flags, we need to remove them. Use a different approach:
    # Undefine _OPENMP to disable OpenMP code paths
    target_compile_definitions(compression_benchmark_no_omp PRIVATE _OPENMP=0)
    target_compile_options(compression_benchmark_no_omp PRIVATE -Wno-unknown-pragmas)
    
    # For GCC/Clang: explicitly disable OpenMP
    if(CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
        target_compile_options(compression_benchmark_no_omp PRIVATE -fno-openmp)
        # Remove OpenMP from link flags by using a custom link options
        get_target_property(LINK_FLAGS compression_benchmark_no_omp LINK_FLAGS)
        if(LINK_FLAGS)
            string(REPLACE "${OpenMP_CXX_FLAGS}" "" LINK_FLAGS "${LINK_FLAGS}")
            set_target_properties(compression_benchmark_no_omp PROPERTIES LINK_FLAGS "${LINK_FLAGS}")
        endif()
    endif()
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