#include "benchmark_utils.hpp"
#include "gef/gef.hpp"
#include <random>

template <typename GefType>
void RegisterRandomAccess(const std::string& name) {
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
             
             // Build GEF
             GefType gef_struct(data);
             
             // Generate random indices
             // Use 1M queries to be safe and avoid small sample noise
             size_t num_queries = 1000000;
             std::vector<size_t> indices(num_queries);
             std::mt19937 gen(42);
             std::uniform_int_distribution<size_t> dist(0, data.size() - 1);
             for(auto& idx : indices) idx = dist(gen);
             
             size_t query_idx = 0;
             for (auto _ : state) {
                 // Perform one access per iteration
                 // This ensures the timing overhead is accounted for by the framework if it loops internally,
                 // but here we are in the loop.
                 // The 'state' iterator defines the loop.
                 
                 // To reduce loop overhead relative to work, we could unroll or batch.
                 // But Google Benchmark handles high-frequency loops well.
                 
                 auto val = gef_struct[indices[query_idx++ % num_queries]];
                 benchmark::DoNotOptimize(val);
             }
             
             state.SetBytesProcessed(state.iterations() * sizeof(uint64_t));
             
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

    RegisterRandomAccess<gef::B_GEF<int64_t>>("B_GEF");
    RegisterRandomAccess<gef::B_GEF_APPROXIMATE<int64_t>>("B_GEF_APPROXIMATE");
    RegisterRandomAccess<gef::B_STAR_GEF<int64_t>>("B_STAR_GEF");
    RegisterRandomAccess<gef::B_STAR_GEF_APPROXIMATE<int64_t>>("B_STAR_GEF_APPROXIMATE");
    RegisterRandomAccess<gef::RLE_GEF<int64_t>>("RLE_GEF");
    RegisterRandomAccess<gef::U_GEF<int64_t>>("U_GEF");
    RegisterRandomAccess<gef::U_GEF_APPROXIMATE<int64_t>>("U_GEF_APPROXIMATE");

    benchmark::RunSpecifiedBenchmarks();
    return 0;
}
