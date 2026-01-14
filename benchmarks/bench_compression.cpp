#include "benchmark_utils.hpp"
#include "gef/gef.hpp"

// Helper to register benchmarks
template <typename GefType>
void RegisterCompression(const std::string& name) {
    for (size_t i = 0; i < g_input_files.size(); ++i) {
        std::string fname = std::filesystem::path(g_input_files[i]).filename().string();
        std::string bench_name = fname + "/" + name;
        
        benchmark::RegisterBenchmark(bench_name.c_str(), [](benchmark::State& state, int file_idx) {
             size_t f_idx = static_cast<size_t>(file_idx);
             const auto& path = g_input_files[f_idx];
             
             state.PauseTiming();
             // Use custom loader
             auto dataset = load_custom_dataset(path);
             auto data = std::move(dataset.data);
             state.ResumeTiming();
             
             for (auto _ : state) {
                GefType gef_struct(data);                
                benchmark::DoNotOptimize(gef_struct.size_in_bytes());
                 
                 // Compute ratio as requested: size_in_bytes() / N * sizeof(uint64_t)
                 // This interprets to (size / N) * 8, which is Bits Per Integer (BPI).
                 double bpi = 0.0;
                 if (!data.empty()) {
                    bpi = (static_cast<double>(gef_struct.size_in_bytes()) / data.size()) * sizeof(uint64_t);
                 }
                 state.counters["Ratio"] = bpi;
                 
                 // Also add standard compression ratio (compressed / raw) for clarity
                 state.counters["CompRatio"] = !data.empty() ? 
                    (static_cast<double>(gef_struct.size_in_bytes()) / (data.size() * sizeof(uint64_t))) : 0.0;
             }
             
             if (!data.empty()) {
                 state.SetBytesProcessed(state.iterations() * data.size() * sizeof(uint64_t));
             }
             
        }, i)->Unit(benchmark::kMillisecond)->UseManualTime();
    }
}

int main(int argc, char** argv) {
    RegisterInputFiles(argc, argv);
    benchmark::Initialize(&argc, argv);
    
    if (g_input_files.empty()) {
        std::cerr << "No input files found.\n";
        return 0;
    }

    RegisterCompression<gef::B_GEF<uint64_t>>("B_GEF");
    RegisterCompression<gef::B_GEF_APPROXIMATE<uint64_t>>("B_GEF_APPROXIMATE");
    RegisterCompression<gef::B_STAR_GEF<uint64_t>>("B_STAR_GEF");
    RegisterCompression<gef::B_STAR_GEF_APPROXIMATE<uint64_t>>("B_STAR_GEF_APPROXIMATE");
    RegisterCompression<gef::RLE_GEF<uint64_t>>("RLE_GEF");
    RegisterCompression<gef::U_GEF<uint64_t>>("U_GEF");
    RegisterCompression<gef::U_GEF_APPROXIMATE<uint64_t>>("U_GEF_APPROXIMATE");

    benchmark::RunSpecifiedBenchmarks();
    return 0;
}
