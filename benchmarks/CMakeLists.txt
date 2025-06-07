file(GLOB BENCHMARK_FILES "*.cpp")

foreach(benchmark_file ${BENCHMARK_FILES})
    get_filename_component(benchmark_name ${benchmark_file} NAME_WE)
    add_executable(${benchmark_name} ${benchmark_file})
    target_link_libraries(${benchmark_name} PRIVATE gef::gef benchmark::benchmark)
endforeach() 