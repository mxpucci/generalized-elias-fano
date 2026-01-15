#include "benchmark_utils.hpp"
#include "gef/gef.hpp"
#include <omp.h>
#include <iostream>
#include <vector>
#include <string>
#include <map>
#include <functional>
#include <iomanip>
#include <filesystem>
#include <numeric>

// Configuration globals
std::string g_compressor_filter = "RLE_GEF";
gef::SplitPointStrategy g_strategy = gef::OPTIMAL_SPLIT_POINT;
int g_threads = 1;

// ==========================================
// Meta-programming Helpers
// ==========================================

template <typename T, typename Data, typename Strat>
using supports_strategy_t = decltype(T(std::declval<Data>(), std::declval<Strat>()));

template <typename T, typename Data, typename Strat, typename = void>
struct has_strategy_ctor : std::false_type {};

template <typename T, typename Data, typename Strat>
struct has_strategy_ctor<T, Data, Strat, std::void_t<supports_strategy_t<T, Data, Strat>>> : std::true_type {};

// ==========================================
// Benchmark Logic
// ==========================================

template<typename GefType, size_t Size, bool RandomAccess>
void RunBenchmark(benchmark::State& state, const std::string& path) {
    constexpr bool uses_strategy = has_strategy_ctor<GefType, std::vector<uint64_t>, gef::SplitPointStrategy>::value;

    // Load dataset once
    std::vector<uint64_t> data;
    try {
        auto dataset = load_custom_dataset(path);
        data = std::move(dataset.data);
    } catch (const std::exception& e) {
        state.SkipWithError(("Failed to load: " + path + " " + e.what()).c_str());
        return;
    }

    if (data.empty()) {
        state.SkipWithError("Empty dataset");
        return;
    }

    // Main Benchmark Loop
    for (auto _ : state) {
        // Measure compression time
        // We include destruction time as it's part of the lifecycle, 
        // but typically construction is the heavy part.
        
        if constexpr (uses_strategy) {
            GefType gef(data, g_strategy);
            benchmark::DoNotOptimize(gef.size_in_bytes());
            
            // Record metrics (only needed once really, but fine to overwrite)
            double size_bytes = static_cast<double>(gef.size_in_bytes());
            double bpe = (size_bytes * 8.0) / data.size();
            state.counters["BPE"] = bpe;
            state.counters["CompressionRatio"] = size_bytes / (data.size() * sizeof(uint64_t));
        } else {
            GefType gef(data);
            benchmark::DoNotOptimize(gef.size_in_bytes());
            
            double size_bytes = static_cast<double>(gef.size_in_bytes());
            double bpe = (size_bytes * 8.0) / data.size();
            state.counters["BPE"] = bpe;
            state.counters["CompressionRatio"] = size_bytes / (data.size() * sizeof(uint64_t));
        }
    }

    state.SetBytesProcessed(state.iterations() * data.size() * sizeof(uint64_t));
}

// ==========================================
// Benchmark Registry
// ==========================================

template<template<typename, size_t, bool> class CompressorT, size_t Size>
void register_variant(const std::string& name) {
    if (name != g_compressor_filter) return;

    for (size_t i = 0; i < g_input_files.size(); ++i) {
        const auto& path = g_input_files[i];
        std::string filename = std::filesystem::path(path).filename().string();

        // 1. Random Access Enabled
        {
            std::string bench_name = filename + "/" + name + "/" + std::to_string(Size) + "/RA_Enabled";
            benchmark::RegisterBenchmark(bench_name.c_str(), [path](benchmark::State& state) {
                RunBenchmark<CompressorT<uint64_t, Size, true>, Size, true>(state, path);
            })->Unit(benchmark::kMillisecond);
        }

        // 2. Random Access Disabled
        {
            std::string bench_name = filename + "/" + name + "/" + std::to_string(Size) + "/RA_Disabled";
            benchmark::RegisterBenchmark(bench_name.c_str(), [path](benchmark::State& state) {
                RunBenchmark<CompressorT<uint64_t, Size, false>, Size, false>(state, path);
            })->Unit(benchmark::kMillisecond);
        }
    }
}

#define REGISTER_VARIANT_SIZE(COMP_NAME, CLASS_NAME, SIZE) \
    register_variant<gef::CLASS_NAME, SIZE>(COMP_NAME);

#define REGISTER_ALL_SIZES(COMP_NAME, CLASS_NAME) \
    REGISTER_VARIANT_SIZE(COMP_NAME, CLASS_NAME, 8000) \
    REGISTER_VARIANT_SIZE(COMP_NAME, CLASS_NAME, 10000) \
    REGISTER_VARIANT_SIZE(COMP_NAME, CLASS_NAME, 16000) \
    REGISTER_VARIANT_SIZE(COMP_NAME, CLASS_NAME, 24000) \
    REGISTER_VARIANT_SIZE(COMP_NAME, CLASS_NAME, 32000) \
    REGISTER_VARIANT_SIZE(COMP_NAME, CLASS_NAME, 40000) \
    REGISTER_VARIANT_SIZE(COMP_NAME, CLASS_NAME, 48000) \
    REGISTER_VARIANT_SIZE(COMP_NAME, CLASS_NAME, 56000) \
    REGISTER_VARIANT_SIZE(COMP_NAME, CLASS_NAME, 64000) \
    REGISTER_VARIANT_SIZE(COMP_NAME, CLASS_NAME, 100000) \
    REGISTER_VARIANT_SIZE(COMP_NAME, CLASS_NAME, 250000) \
    REGISTER_VARIANT_SIZE(COMP_NAME, CLASS_NAME, 500000)

void initialize_registry() {
    REGISTER_ALL_SIZES("RLE_GEF", RLE_GEF);
    REGISTER_ALL_SIZES("B_GEF", B_GEF);
    REGISTER_ALL_SIZES("B_GEF_APPROXIMATE", B_GEF_APPROXIMATE);
    REGISTER_ALL_SIZES("B_STAR_GEF", B_STAR_GEF);
    REGISTER_ALL_SIZES("B_STAR_GEF_APPROXIMATE", B_STAR_GEF_APPROXIMATE);
    REGISTER_ALL_SIZES("U_GEF", U_GEF);
    REGISTER_ALL_SIZES("U_GEF_APPROXIMATE", U_GEF_APPROXIMATE);
}

// ==========================================
// Main
// ==========================================

void print_usage(const char* prog) {
    std::cerr << "Usage: " << prog << " [options] <dataset.bin> [more_datasets...]\n\n";
    std::cerr << "Options:\n";
    std::cerr << "  --compressor NAME       Compressor to benchmark (default: RLE_GEF)\n";
    std::cerr << "  --strategy STRAT        OPTIMAL or APPROXIMATE (default: OPTIMAL)\n";
    std::cerr << "  --threads N             Number of threads for parallel compression (default: 1)\n";
    std::cerr << "  [Google Benchmark Flags] (e.g., --benchmark_out=res.json --benchmark_format=json)\n\n";
    std::cerr << "Available compressors: RLE_GEF, B_GEF, B_GEF_APPROXIMATE, B_STAR_GEF, B_STAR_GEF_APPROXIMATE, U_GEF, U_GEF_APPROXIMATE\n";
}

int main(int argc, char** argv) {
    // 1. Parse custom args and filter them out for Google Benchmark
    std::vector<char*> bm_argv;
    bm_argv.push_back(argv[0]);

    for (int i = 1; i < argc; ++i) {
        std::string arg = argv[i];
        if (arg == "--compressor" && i + 1 < argc) {
            g_compressor_filter = argv[++i];
        } else if (arg == "--strategy" && i + 1 < argc) {
            std::string s = argv[++i];
            g_strategy = (s == "APPROXIMATE") ? gef::APPROXIMATE_SPLIT_POINT : gef::OPTIMAL_SPLIT_POINT;
        } else if (arg == "--threads" && i + 1 < argc) {
            g_threads = std::stoi(argv[++i]);
        } else if (arg == "--help" || arg == "-h") {
            print_usage(argv[0]);
            return 0;
        } else {
            // Check if it looks like a flag
            if (arg.rfind("-", 0) == 0) {
                // Pass through benchmark flags
                bm_argv.push_back(argv[i]);
            } else {
                // Assume it's a file
                if (std::filesystem::is_directory(arg)) {
                    for (const auto& e : std::filesystem::directory_iterator(arg)) {
                        if (e.path().extension() == ".bin") g_input_files.push_back(e.path());
                    }
                } else {
                    g_input_files.push_back(argv[i]);
                }
            }
        }
    }
    
    // Sort files for consistent order
    std::sort(g_input_files.begin(), g_input_files.end());

    if (g_input_files.empty()) {
        std::cerr << "Error: No datasets provided\n";
        print_usage(argv[0]);
        return 1;
    }

    // Set thread count
    omp_set_num_threads(g_threads);

    // Initialize Benchmark with filtered args
    int bm_argc = static_cast<int>(bm_argv.size());
    benchmark::Initialize(&bm_argc, bm_argv.data());

    // Register Benchmarks
    initialize_registry();

    std::cout << "Configuration:\n";
    std::cout << "  Compressor: " << g_compressor_filter << "\n";
    std::cout << "  Strategy: " << (g_strategy == gef::OPTIMAL_SPLIT_POINT ? "OPTIMAL" : "APPROXIMATE") << "\n";
    std::cout << "  Threads: " << g_threads << "\n";
    std::cout << "  Datasets: " << g_input_files.size() << "\n\n";

    benchmark::RunSpecifiedBenchmarks();
    
    return 0;
}
