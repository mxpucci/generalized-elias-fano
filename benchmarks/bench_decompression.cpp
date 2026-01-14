#include "benchmark_utils.hpp"
#include "gef/gef.hpp"

template <typename GefType>
void RegisterDecompression(const std::string& name) {
    for (size_t i = 0; i < g_input_files.size(); ++i) {
        std::string fname = std::filesystem::path(g_input_files[i]).filename().string();
        std::string bench_name = fname + "/" + name;
        
        benchmark::RegisterBenchmark(bench_name.c_str(), [](benchmark::State& state, int file_idx) {
             size_t f_idx = static_cast<size_t>(file_idx);
             const auto& path = g_input_files[f_idx];
             
             // Load dataset and build GEF outside the benchmark loop
             auto dataset = load_custom_dataset(path);
             auto data = std::move(dataset.data);
             if (data.empty()) {
                 state.SkipWithError("Empty data");
                 return;
             }
             
             GefType gef_struct(data);
             std::vector<uint64_t> output(data.size());
             
             for (auto _ : state) {
                 gef_struct.get_elements(0, data.size(), output);
                 benchmark::DoNotOptimize(output.data());
                 benchmark::ClobberMemory();
             }
             
             state.SetBytesProcessed(state.iterations() * data.size() * sizeof(uint64_t));
             
        }, i)->Unit(benchmark::kMillisecond);
    }
}

int main(int argc, char** argv) {
    RegisterInputFiles(argc, argv);
    benchmark::Initialize(&argc, argv);
    
    if (g_input_files.empty()) {
        std::cerr << "No input files found.\n";
        return 0;
    }

    RegisterDecompression<gef::B_GEF<uint64_t>>("B_GEF");
    RegisterDecompression<gef::B_GEF_APPROXIMATE<uint64_t>>("B_GEF_APPROXIMATE");
    RegisterDecompression<gef::B_STAR_GEF<uint64_t>>("B_STAR_GEF");
    RegisterDecompression<gef::B_STAR_GEF_APPROXIMATE<uint64_t>>("B_STAR_GEF_APPROXIMATE");
    RegisterDecompression<gef::RLE_GEF<uint64_t>>("RLE_GEF");
    RegisterDecompression<gef::U_GEF<uint64_t>>("U_GEF");
    RegisterDecompression<gef::U_GEF_APPROXIMATE<uint64_t>>("U_GEF_APPROXIMATE");

    benchmark::RunSpecifiedBenchmarks();
    return 0;
}
