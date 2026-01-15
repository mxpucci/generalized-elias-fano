#include "gef/gef.hpp"
#include <omp.h>
#include <chrono>
#include <iomanip>
#include <iostream>
#include <fstream>
#include <vector>
#include <string>
#include <filesystem>
#include <algorithm>
#include <numeric>
#include <functional>
#include <map>
#include <tuple>
#include <type_traits>

// ==========================================
// Constants & Helpers
// ==========================================

const std::vector<int> THREAD_COUNTS = {1, 2, 4, 8, 12, 16, 20, 24, 28, 32, 64};

std::vector<uint64_t> load_dataset(const std::string& filename) {
    std::ifstream in(filename, std::ios::binary);
    if (!in) throw std::runtime_error("Cannot open file: " + filename);
    
    uint64_t n;
    in.read(reinterpret_cast<char*>(&n), 8);
    
    in.seekg(0, std::ios::end);
    size_t file_size = in.tellg();
    in.seekg(8, std::ios::beg);
    
    if (file_size == 8 + 8 + n * 8) {
        uint64_t x;
        in.read(reinterpret_cast<char*>(&x), 8);
    }
    
    std::vector<uint64_t> data(n);
    in.read(reinterpret_cast<char*>(data.data()), n * 8);
    return data;
}

double calculate_throughput(size_t num_elements, double time_ms) {
    double bytes = num_elements * sizeof(uint64_t);
    double megabytes = bytes / (1024.0 * 1024.0);
    double seconds = time_ms / 1000.0;
    return megabytes / seconds;
}

// ==========================================
// Meta-programming Helpers
// ==========================================

// Helper to check if a class can be constructed with (data, strategy)
template <typename T, typename Data, typename Strat>
using supports_strategy_t = decltype(T(std::declval<Data>(), std::declval<Strat>()));

template <typename T, typename Data, typename Strat, typename = void>
struct has_strategy_ctor : std::false_type {};

template <typename T, typename Data, typename Strat>
struct has_strategy_ctor<T, Data, Strat, std::void_t<supports_strategy_t<T, Data, Strat>>> : std::true_type {};

// ==========================================
// Core Benchmark Logic
// ==========================================

template<typename GefType>
void run_benchmark(const std::string& name, size_t partition_size,
                   const std::vector<std::string>& paths, gef::SplitPointStrategy strategy) {
    
    constexpr bool uses_strategy = has_strategy_ctor<GefType, std::vector<uint64_t>, gef::SplitPointStrategy>::value;

    std::cout << "\n========================================\n";
    std::cout << "Compressor: " << name << "\n";
    std::cout << "Partition Size: " << partition_size << "\n";
    if constexpr (uses_strategy) {
        std::cout << "Strategy: " << (strategy == gef::OPTIMAL_SPLIT_POINT ? "OPTIMAL" : "APPROXIMATE") << "\n";
    } else {
        std::cout << "Strategy: N/A (Compressor ignores strategy)\n";
    }
    std::cout << "========================================\n\n";
    
    std::cout << std::setw(8) << "Threads" << " | "
              << std::setw(12) << "Time (ms)" << " | "
              << std::setw(14) << "Throughput" << " | "
              << std::setw(8) << "Speedup" << " | "
              << std::setw(10) << "Efficiency\n";
    std::cout << std::string(65, '-') << "\n";
    
    double baseline = 0;
    
    for (int threads : THREAD_COUNTS) {
        std::vector<double> throughputs;
        std::vector<double> times;
        
        for (const auto& path : paths) {
            try {
                auto data = load_dataset(path);
                
                omp_set_num_threads(threads);
                auto start = std::chrono::high_resolution_clock::now();
                
                // Compile-time check: Initialize with or without strategy based on the type
                if constexpr (uses_strategy) {
                    GefType gef(data, strategy);
                    // Ensure code isn't optimized away
                    volatile auto sz = gef.size_in_bytes(); (void)sz;
                } else {
                    GefType gef(data); 
                    volatile auto sz = gef.size_in_bytes(); (void)sz;
                }
                
                auto end = std::chrono::high_resolution_clock::now();
                
                double ms = std::chrono::duration<double, std::milli>(end - start).count();
                double tp = calculate_throughput(data.size(), ms);
                throughputs.push_back(tp);
                times.push_back(ms);
            } catch (const std::exception& e) {
                std::cerr << "Error processing " << path << ": " << e.what() << "\n";
            }
        }
        
        if (throughputs.empty()) continue;
        
        double avg_tp = std::accumulate(throughputs.begin(), throughputs.end(), 0.0) / throughputs.size();
        double avg_time = std::accumulate(times.begin(), times.end(), 0.0) / times.size();
        
        if (threads == 1) baseline = avg_tp;
        double speedup = (baseline > 0) ? avg_tp / baseline : 0.0;
        double efficiency = (threads > 0) ? (speedup / threads) * 100.0 : 0.0;
        
        std::cout << std::setw(8) << threads << " | "
                  << std::setw(12) << std::fixed << std::setprecision(2) << avg_time << " | "
                  << std::setw(11) << avg_tp << " MB/s | "
                  << std::setw(7) << std::setprecision(2) << speedup << "x | "
                  << std::setw(9) << std::setprecision(1) << efficiency << "%\n";
    }
}

// ==========================================
// Benchmark Registry (Dispatcher)
// ==========================================

// Function signature for the type-erased benchmark runner
using BenchmarkRunner = std::function<void(const std::vector<std::string>&, gef::SplitPointStrategy)>;

// The Registry Map: Key = {CompressorName, PartitionSize}
static std::map<std::pair<std::string, size_t>, BenchmarkRunner> benchmark_registry;

// Helper to register a specific template instantiation
template<template<typename, size_t, bool> class CompressorT, size_t Size, bool random_access = true>
void register_variant(const std::string& name) {
    benchmark_registry[{name, Size}] = [name](const std::vector<std::string>& paths, gef::SplitPointStrategy strategy) {
        run_benchmark<CompressorT<uint64_t, Size, true>>(name, Size, paths, strategy);
    };
}

// Helper macro to register a compressor for multiple standard sizes
#define REGISTER_SIZES(COMP_NAME, CLASS_NAME) \
    register_variant<gef::CLASS_NAME, 64000>(COMP_NAME); \
    register_variant<gef::CLASS_NAME, 65536>(COMP_NAME); \
    register_variant<gef::CLASS_NAME, 131072>(COMP_NAME);

// Initialize all valid combinations
void initialize_registry() {
    REGISTER_SIZES("RLE_GEF", RLE_GEF);
    REGISTER_SIZES("B_GEF", B_GEF);
    REGISTER_SIZES("B_GEF_APPROXIMATE", B_GEF_APPROXIMATE);
    REGISTER_SIZES("B_STAR_GEF", B_STAR_GEF);
    REGISTER_SIZES("B_STAR_GEF_APPROXIMATE", B_STAR_GEF_APPROXIMATE);
    REGISTER_SIZES("U_GEF", U_GEF);
    REGISTER_SIZES("U_GEF_APPROXIMATE", U_GEF_APPROXIMATE);
}

// ==========================================
// Main
// ==========================================

void print_usage(const char* prog) {
    std::cerr << "Usage: " << prog << " [options] <dataset.bin> [more_datasets...]\n\n";
    std::cerr << "Options:\n";
    std::cerr << "  --compressor NAME       Compressor (default: RLE_GEF)\n";
    std::cerr << "  --partition-size SIZE   Partition size (default: 64000)\n";
    std::cerr << "  --strategy STRAT        OPTIMAL or APPROXIMATE (default: OPTIMAL)\n\n";
    std::cerr << "Available variants:\n";
    for (const auto& [key, val] : benchmark_registry) {
        std::cerr << "  " << key.first << " @ " << key.second << "\n";
    }
}

int main(int argc, char** argv) {
    initialize_registry();

    std::string compressor = "RLE_GEF";
    size_t partition_size = 64000;
    gef::SplitPointStrategy strategy = gef::OPTIMAL_SPLIT_POINT;
    std::vector<std::string> paths;
    
    for (int i = 1; i < argc; ++i) {
        std::string arg = argv[i];
        if (arg == "--compressor" && i+1 < argc) {
            compressor = argv[++i];
        } else if (arg == "--partition-size" && i+1 < argc) {
            partition_size = std::stoul(argv[++i]);
        } else if (arg == "--strategy" && i+1 < argc) {
            std::string s = argv[++i];
            strategy = (s == "APPROXIMATE") ? gef::APPROXIMATE_SPLIT_POINT : gef::OPTIMAL_SPLIT_POINT;
        } else if (arg == "--help" || arg == "-h") {
            print_usage(argv[0]);
            return 0;
        } else if (arg[0] != '-') {
            if (std::filesystem::is_directory(arg)) {
                for (const auto& e : std::filesystem::directory_iterator(arg)) {
                    if (e.path().extension() == ".bin") paths.push_back(e.path());
                }
            } else {
                paths.push_back(arg);
            }
        }
    }
    
    if (paths.empty()) {
        std::cerr << "Error: No datasets provided\n";
        print_usage(argv[0]);
        return 1;
    }
    
    std::sort(paths.begin(), paths.end());
    
    // Look up the benchmark in the registry
    auto it = benchmark_registry.find({compressor, partition_size});
    if (it != benchmark_registry.end()) {
        std::cout << "Starting Benchmark for " << compressor << " (Size: " << partition_size << ")\n";
        // Execute the function stored in the map
        it->second(paths, strategy);
    } else {
        std::cerr << "Error: The combination of Compressor '" << compressor 
                  << "' and Partition Size " << partition_size << " is not pre-compiled.\n";
        std::cerr << "Please use one of the sizes listed in --help or add the size to 'initialize_registry()'.\n";
        return 1;
    }
    
    return 0;
}