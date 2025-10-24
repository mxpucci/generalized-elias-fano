file(GLOB BENCHMARK_FILES "*.cpp" "compression_benchmark.cpp")

foreach(benchmark_file ${BENCHMARK_FILES})
    get_filename_component(benchmark_name ${benchmark_file} NAME_WE)
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