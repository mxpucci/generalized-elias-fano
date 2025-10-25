#include <benchmark/benchmark.h>
#include "gef/B_GEF.hpp"
#include "gef/U_GEF.hpp"
#include "gef/RLE_GEF.hpp"
#include "gef/B_GEF_STAR.hpp"
#include "gef/UniformedPartitioner.hpp"
#include "datastructures/SDSLBitVectorFactory.hpp"
#include "datastructures/SUXBitVectorFactory.hpp"
#include "gef/utils.hpp"
#include <vector>
#include <random>
#include <algorithm>
#include <memory>
#include <string>
#include <iostream>
#include <chrono>
#include <filesystem>
#include <sstream>

// (Compressor Wrappers and Global Data are unchanged)
#pragma region Compressor Wrappers and Globals
template<typename T>
struct U_GEF_Wrapper : public gef::U_GEF<T> {
    // This constructor now matches B_GEF's, taking a strategy.
    // It is assumed the base class gef::U_GEF has been updated to accept this parameter.
    U_GEF_Wrapper(const std::vector<T>& data, std::shared_ptr<IBitVectorFactory> factory, gef::SplitPointStrategy strategy)
        : gef::U_GEF<T>(factory, data, strategy) {}
    U_GEF_Wrapper() : gef::U_GEF<T>() {}
};

template<typename T>
struct RLE_GEF_Wrapper : public gef::RLE_GEF<T> {
    RLE_GEF_Wrapper(const std::vector<T>& data, std::shared_ptr<IBitVectorFactory> factory)
        : gef::RLE_GEF<T>(factory, data) {}
    RLE_GEF_Wrapper() : gef::RLE_GEF<T>() {}
};

template<typename T>
struct B_GEF_Wrapper : public gef::B_GEF<T> {
    B_GEF_Wrapper(const std::vector<T>& data, std::shared_ptr<IBitVectorFactory> factory, gef::SplitPointStrategy strategy)
        : gef::B_GEF<T>(factory, data, strategy) {}
    B_GEF_Wrapper() : gef::B_GEF<T>() {}
};

template<typename T>
struct B_GEF_NO_RLE_Wrapper : public gef::B_GEF_STAR<T> {
    B_GEF_NO_RLE_Wrapper(const std::vector<T>& data, std::shared_ptr<IBitVectorFactory> factory, gef::SplitPointStrategy strategy)
        : gef::B_GEF_STAR<T>(factory, data, strategy) {}
    B_GEF_NO_RLE_Wrapper() : gef::B_GEF_STAR<T>() {}
};

std::vector<std::string> g_input_files;
std::shared_ptr<IBitVectorFactory> g_factory = std::make_shared<SDSLBitVectorFactory>();
std::string g_factory_name = "SDSL";

enum class BitVectorImplementation {
    SDSL,
    SUX
};

std::string strategyToString(gef::SplitPointStrategy strategy) {
    switch (strategy) {
        case gef::SplitPointStrategy::APPROXIMATE_SPLIT_POINT: return "APPROXIMATE";
        case gef::SplitPointStrategy::OPTIMAL_SPLIT_POINT: return "BRUTE_FORCE";
        default: return "UNKNOWN";
    }
}

std::string implementationToString(BitVectorImplementation impl) {
    switch (impl) {
        case BitVectorImplementation::SDSL: return "SDSL";
        case BitVectorImplementation::SUX: return "SUX";
        default: return "UNKNOWN";
    }
}

void setFactory(BitVectorImplementation impl) {
    switch (impl) {
        case BitVectorImplementation::SDSL:
            g_factory = std::make_shared<SDSLBitVectorFactory>();
            g_factory_name = "SDSL";
            break;
        case BitVectorImplementation::SUX:
            g_factory = std::make_shared<SUXBitVectorFactory>();
            g_factory_name = "SUX";
            break;
    }
}
#pragma endregion

#pragma region Benchmark Fixture
class FileBasedCompressionBenchmark : public benchmark::Fixture {
public:
    std::vector<int64_t> input_data;
    std::string current_file_path;
    std::string current_basename;

    void SetUp(::benchmark::State& state) override {
        size_t file_idx = state.range(0);
        if (file_idx >= g_input_files.size()) {
            state.SkipWithError("File index out of bounds.");
            return;
        }
        current_file_path = g_input_files[file_idx];
        current_basename = std::filesystem::path(current_file_path).filename().string();

        input_data = read_data_binary<int64_t, int64_t>(current_file_path, true);
        if (input_data.empty()) {
            state.SkipWithError("Input data is empty.");
        }
        state.counters["num_integers"] = input_data.size();
    }

    void TearDown(::benchmark::State& state) override {}
};
#pragma endregion


// ============================================================================
// Benchmark Definitions - Type 1: Compression Efficiency (Time & Space)
// ============================================================================

const std::vector<size_t> PARTITION_SIZES = {512, 1024, 2048, 4096, 8192, 16384, 32768};

BENCHMARK_DEFINE_F(FileBasedCompressionBenchmark, B_GEF_Compression)(benchmark::State& state) {
    gef::SplitPointStrategy strategy = static_cast<gef::SplitPointStrategy>(state.range(1));
    size_t partition_size = state.range(2);
    state.SetLabel(current_basename + "/" + g_factory_name + "/" + strategyToString(strategy) + "/" + std::to_string(partition_size));

    for (auto _ : state) {
        state.PauseTiming();
        std::vector<int64_t> data_copy = input_data;
        state.ResumeTiming();
        gef::UniformedPartitioner<int64_t, B_GEF_Wrapper<int64_t>, std::shared_ptr<IBitVectorFactory>, gef::SplitPointStrategy> compressor(
            data_copy, partition_size, g_factory, strategy);
        benchmark::DoNotOptimize(compressor);
    }

    gef::UniformedPartitioner<int64_t, B_GEF_Wrapper<int64_t>, std::shared_ptr<IBitVectorFactory>, gef::SplitPointStrategy> final_compressor(
        input_data, partition_size, g_factory, strategy);
    state.counters["size_in_bytes"] = final_compressor.size_in_bytes();
    state.counters["bpi"] = static_cast<double>(final_compressor.size_in_bytes() * 8) / input_data.size();

    double bytes_processed = static_cast<double>(state.iterations() * input_data.size() * sizeof(int64_t));
    state.counters["compression_throughput_MBs"] = benchmark::Counter(bytes_processed, benchmark::Counter::kIsRate, benchmark::Counter::kIs1024);
}

BENCHMARK_DEFINE_F(FileBasedCompressionBenchmark, B_GEF_NO_RLE_Compression)(benchmark::State& state) {
    gef::SplitPointStrategy strategy = static_cast<gef::SplitPointStrategy>(state.range(1));
    size_t partition_size = state.range(2);
    state.SetLabel(current_basename + "/" + g_factory_name + "/" + strategyToString(strategy) + "/" + std::to_string(partition_size));

    for (auto _ : state) {
        state.PauseTiming();
        std::vector<int64_t> data_copy = input_data;
        state.ResumeTiming();
        gef::UniformedPartitioner<int64_t, B_GEF_NO_RLE_Wrapper<int64_t>, std::shared_ptr<IBitVectorFactory>, gef::SplitPointStrategy> compressor(
            data_copy, partition_size, g_factory, strategy);
        benchmark::DoNotOptimize(compressor);
    }

    gef::UniformedPartitioner<int64_t, B_GEF_NO_RLE_Wrapper<int64_t>, std::shared_ptr<IBitVectorFactory>, gef::SplitPointStrategy> final_compressor(
        input_data, partition_size, g_factory, strategy);
    state.counters["size_in_bytes"] = final_compressor.size_in_bytes();
    state.counters["bpi"] = static_cast<double>(final_compressor.size_in_bytes() * 8) / input_data.size();


    double bytes_processed = static_cast<double>(state.iterations() * input_data.size() * sizeof(int64_t));
    state.counters["compression_throughput_MBs"] = benchmark::Counter(bytes_processed, benchmark::Counter::kIsRate, benchmark::Counter::kIs1024);
}

BENCHMARK_DEFINE_F(FileBasedCompressionBenchmark, U_GEF_Compression)(benchmark::State& state) {
    gef::SplitPointStrategy strategy = static_cast<gef::SplitPointStrategy>(state.range(1));
    size_t partition_size = state.range(2);
    state.SetLabel(current_basename + "/" + g_factory_name + "/" + strategyToString(strategy) + "/" + std::to_string(partition_size));

    for (auto _ : state) {
        state.PauseTiming();
        std::vector<int64_t> data_copy = input_data;
        state.ResumeTiming();
        gef::UniformedPartitioner<int64_t, U_GEF_Wrapper<int64_t>, std::shared_ptr<IBitVectorFactory>, gef::SplitPointStrategy> compressor(
            data_copy, partition_size, g_factory, strategy);
        benchmark::DoNotOptimize(compressor);
    }
    gef::UniformedPartitioner<int64_t, U_GEF_Wrapper<int64_t>, std::shared_ptr<IBitVectorFactory>, gef::SplitPointStrategy> final_compressor(
        input_data, partition_size, g_factory, strategy);
    state.counters["size_in_bytes"] = final_compressor.size_in_bytes();
    state.counters["bpi"] = static_cast<double>(final_compressor.size_in_bytes() * 8) / input_data.size();


    double bytes_processed = static_cast<double>(state.iterations() * input_data.size() * sizeof(int64_t));
    state.counters["compression_throughput_MBs"] = benchmark::Counter(bytes_processed, benchmark::Counter::kIsRate, benchmark::Counter::kIs1024);
}

BENCHMARK_DEFINE_F(FileBasedCompressionBenchmark, RLE_GEF_Compression)(benchmark::State& state) {
    size_t partition_size = state.range(1);
    state.SetLabel(current_basename + "/" + g_factory_name + "/" + std::to_string(partition_size));
    for (auto _ : state) {
        state.PauseTiming();
        std::vector<int64_t> data_copy = input_data;
        state.ResumeTiming();
        gef::UniformedPartitioner<int64_t, RLE_GEF_Wrapper<int64_t>, std::shared_ptr<IBitVectorFactory>> compressor(
            data_copy, partition_size, g_factory);
        benchmark::DoNotOptimize(compressor);
    }
    gef::UniformedPartitioner<int64_t, RLE_GEF_Wrapper<int64_t>, std::shared_ptr<IBitVectorFactory>> final_compressor(
        input_data, partition_size, g_factory);
    state.counters["size_in_bytes"] = final_compressor.size_in_bytes();
    state.counters["bpi"] = static_cast<double>(final_compressor.size_in_bytes() * 8) / input_data.size();

    double bytes_processed = static_cast<double>(state.iterations() * input_data.size() * sizeof(int64_t));
    state.counters["compression_throughput_MBs"] = benchmark::Counter(bytes_processed, benchmark::Counter::kIsRate, benchmark::Counter::kIs1024);
}

// ============================================================================
// Benchmark Definitions - Type 3: Lookup Efficiency
// ============================================================================

const size_t NUM_LOOKUPS = 100000;

template<template<typename> class CompressorWrapper, typename... Args>
void LookupBenchmark(benchmark::State& state, const std::vector<int64_t>& data, size_t partition_size, Args... args) {
    if (data.empty()) {
        state.SkipWithError("Input data is empty for lookup benchmark.");
        return;
    }
    gef::UniformedPartitioner<int64_t, CompressorWrapper<int64_t>, Args...> compressor(data, partition_size, args...);
    std::vector<size_t> random_indices(NUM_LOOKUPS);
    std::mt19937 gen(1337);
    std::uniform_int_distribution<size_t> distrib(0, data.size() - 1);
    for (size_t i = 0; i < NUM_LOOKUPS; ++i) {
        random_indices[i] = distrib(gen);
    }

    for (auto _ : state) {
        for (size_t i = 0; i < NUM_LOOKUPS; ++i) {
            benchmark::DoNotOptimize(compressor[random_indices[i]]);
        }
    }

    state.SetItemsProcessed(NUM_LOOKUPS * state.iterations());
    double bytes_processed = static_cast<double>(state.items_processed() * sizeof(int64_t));
    state.counters["lookup_throughput_MBs"] = benchmark::Counter(bytes_processed, benchmark::Counter::kIsRate, benchmark::Counter::kIs1024);
}


// ============================================================================
// Benchmark Definitions - Type 4: Serialization Space
// ============================================================================
template<template<typename> class CompressorWrapper, typename... Args>
void SerializationSpaceBenchmark(benchmark::State& state, const std::vector<int64_t>& data, size_t partition_size, Args... args) {
    if (data.empty()) {
        state.SkipWithError("Input data is empty for serialization space benchmark.");
        return;
    }

    // Only run once - serialization benchmarks measure space, not time
    for (auto _ : state) {
        gef::UniformedPartitioner<int64_t, CompressorWrapper<int64_t>, Args...> compressor(data, partition_size, args...);

        // Using a unique path for each run to be safe
        std::string temp_filename = "benchmark_serialization_" + std::to_string(std::chrono::high_resolution_clock::now().time_since_epoch().count()) + ".tmp";
        std::filesystem::path temp_path = std::filesystem::temp_directory_path() / temp_filename;

        compressor.serialize(temp_path);

        try {
            size_t file_size = std::filesystem::file_size(temp_path);
            state.counters["serialized_size_in_bytes"] = file_size;
            state.counters["serialized_bpi"] = static_cast<double>(file_size * 8) / data.size();
        } catch (const std::filesystem::filesystem_error& e) {
            state.SkipWithError(("Failed to get file size: " + std::string(e.what())).c_str());
        }

        std::filesystem::remove(temp_path);
    }
}

// ============================================================================
// Benchmark Definitions - Type 5: Decompression Throughput (Full)
// ============================================================================
template<template<typename> class CompressorWrapper, typename... Args>
void DecompressionThroughputBenchmark(benchmark::State& state, const std::vector<int64_t>& data, size_t partition_size, Args... args) {
    if (data.empty()) {
        state.SkipWithError("Input data is empty for decompression throughput benchmark.");
        return;
    }
    
    // Build the compressor once outside the timing loop
    gef::UniformedPartitioner<int64_t, CompressorWrapper<int64_t>, Args...> compressor(data, partition_size, args...);
    
    for (auto _ : state) {
        // Decompress ALL elements using optimized get_elements
        auto decompressed = compressor.get_elements(0, data.size());
        benchmark::DoNotOptimize(decompressed);
    }
    
    // Report throughput
    state.SetItemsProcessed(data.size() * state.iterations());
    double bytes_processed = static_cast<double>(state.items_processed() * sizeof(int64_t));
    state.counters["decompression_throughput_MBs"] = benchmark::Counter(bytes_processed, benchmark::Counter::kIsRate, benchmark::Counter::kIs1024);
}


#pragma region Lookup Definitions and Registration
BENCHMARK_DEFINE_F(FileBasedCompressionBenchmark, B_GEF_Lookup)(benchmark::State& state) {
    gef::SplitPointStrategy strategy = static_cast<gef::SplitPointStrategy>(state.range(1));
    size_t partition_size = state.range(2);
    state.SetLabel(current_basename + "/" + g_factory_name + "/" + strategyToString(strategy) + "/" + std::to_string(partition_size));
    LookupBenchmark<B_GEF_Wrapper>(state, input_data, partition_size, g_factory, strategy);
}

BENCHMARK_DEFINE_F(FileBasedCompressionBenchmark, B_GEF_NO_RLE_Lookup)(benchmark::State& state) {
    gef::SplitPointStrategy strategy = static_cast<gef::SplitPointStrategy>(state.range(1));
    size_t partition_size = state.range(2);
    state.SetLabel(current_basename + "/" + g_factory_name + "/" + strategyToString(strategy) + "/" + std::to_string(partition_size));
    LookupBenchmark<B_GEF_NO_RLE_Wrapper>(state, input_data, partition_size, g_factory, strategy);
}

BENCHMARK_DEFINE_F(FileBasedCompressionBenchmark, U_GEF_Lookup)(benchmark::State& state) {
    gef::SplitPointStrategy strategy = static_cast<gef::SplitPointStrategy>(state.range(1));
    size_t partition_size = state.range(2);
    state.SetLabel(current_basename + "/" + g_factory_name + "/" + strategyToString(strategy) + "/" + std::to_string(partition_size));
    LookupBenchmark<U_GEF_Wrapper>(state, input_data, partition_size, g_factory, strategy);
}

BENCHMARK_DEFINE_F(FileBasedCompressionBenchmark, RLE_GEF_Lookup)(benchmark::State& state) {
    size_t partition_size = state.range(1);
    state.SetLabel(current_basename + "/" + g_factory_name + "/" + std::to_string(partition_size));
    LookupBenchmark<RLE_GEF_Wrapper>(state, input_data, partition_size, g_factory);
}

BENCHMARK_DEFINE_F(FileBasedCompressionBenchmark, B_GEF_Serialization_Space)(benchmark::State& state) {
    gef::SplitPointStrategy strategy = static_cast<gef::SplitPointStrategy>(state.range(1));
    size_t partition_size = state.range(2);
    state.SetLabel(current_basename + "/" + g_factory_name + "/" + strategyToString(strategy) + "/" + std::to_string(partition_size));
    SerializationSpaceBenchmark<B_GEF_Wrapper>(state, input_data, partition_size, g_factory, strategy);
}

BENCHMARK_DEFINE_F(FileBasedCompressionBenchmark, B_GEF_NO_RLE_Serialization_Space)(benchmark::State& state) {
    gef::SplitPointStrategy strategy = static_cast<gef::SplitPointStrategy>(state.range(1));
    size_t partition_size = state.range(2);
    state.SetLabel(current_basename + "/" + g_factory_name + "/" + strategyToString(strategy) + "/" + std::to_string(partition_size));
    SerializationSpaceBenchmark<B_GEF_NO_RLE_Wrapper>(state, input_data, partition_size, g_factory, strategy);
}

BENCHMARK_DEFINE_F(FileBasedCompressionBenchmark, U_GEF_Serialization_Space)(benchmark::State& state) {
    gef::SplitPointStrategy strategy = static_cast<gef::SplitPointStrategy>(state.range(1));
    size_t partition_size = state.range(2);
    state.SetLabel(current_basename + "/" + g_factory_name + "/" + strategyToString(strategy) + "/" + std::to_string(partition_size));
    SerializationSpaceBenchmark<U_GEF_Wrapper>(state, input_data, partition_size, g_factory, strategy);
}

BENCHMARK_DEFINE_F(FileBasedCompressionBenchmark, RLE_GEF_Serialization_Space)(benchmark::State& state) {
    size_t partition_size = state.range(1);
    state.SetLabel(current_basename + "/" + g_factory_name + "/" + std::to_string(partition_size));
    SerializationSpaceBenchmark<RLE_GEF_Wrapper>(state, input_data, partition_size, g_factory);
}

// Decompression Throughput Benchmark Definitions
BENCHMARK_DEFINE_F(FileBasedCompressionBenchmark, B_GEF_Decompression)(benchmark::State& state) {
    gef::SplitPointStrategy strategy = static_cast<gef::SplitPointStrategy>(state.range(1));
    size_t partition_size = state.range(2);
    state.SetLabel(current_basename + "/" + g_factory_name + "/" + strategyToString(strategy) + "/" + std::to_string(partition_size));
    DecompressionThroughputBenchmark<B_GEF_Wrapper>(state, input_data, partition_size, g_factory, strategy);
}

BENCHMARK_DEFINE_F(FileBasedCompressionBenchmark, B_GEF_NO_RLE_Decompression)(benchmark::State& state) {
    gef::SplitPointStrategy strategy = static_cast<gef::SplitPointStrategy>(state.range(1));
    size_t partition_size = state.range(2);
    state.SetLabel(current_basename + "/" + g_factory_name + "/" + strategyToString(strategy) + "/" + std::to_string(partition_size));
    DecompressionThroughputBenchmark<B_GEF_NO_RLE_Wrapper>(state, input_data, partition_size, g_factory, strategy);
}

BENCHMARK_DEFINE_F(FileBasedCompressionBenchmark, U_GEF_Decompression)(benchmark::State& state) {
    gef::SplitPointStrategy strategy = static_cast<gef::SplitPointStrategy>(state.range(1));
    size_t partition_size = state.range(2);
    state.SetLabel(current_basename + "/" + g_factory_name + "/" + strategyToString(strategy) + "/" + std::to_string(partition_size));
    DecompressionThroughputBenchmark<U_GEF_Wrapper>(state, input_data, partition_size, g_factory, strategy);
}

BENCHMARK_DEFINE_F(FileBasedCompressionBenchmark, RLE_GEF_Decompression)(benchmark::State& state) {
    size_t partition_size = state.range(1);
    state.SetLabel(current_basename + "/" + g_factory_name + "/" + std::to_string(partition_size));
    DecompressionThroughputBenchmark<RLE_GEF_Wrapper>(state, input_data, partition_size, g_factory);
}

void RegisterBenchmarksForFile(size_t file_idx) {
    // Register all partition sizes by chaining Args() calls
    // Compression
    BENCHMARK_REGISTER_F(FileBasedCompressionBenchmark, B_GEF_Compression)
        ->Args({(long)file_idx, (long)gef::SplitPointStrategy::APPROXIMATE_SPLIT_POINT, 512})
        ->Args({(long)file_idx, (long)gef::SplitPointStrategy::OPTIMAL_SPLIT_POINT, 512})
        ->Args({(long)file_idx, (long)gef::SplitPointStrategy::APPROXIMATE_SPLIT_POINT, 1024})
        ->Args({(long)file_idx, (long)gef::SplitPointStrategy::OPTIMAL_SPLIT_POINT, 1024})
        ->Args({(long)file_idx, (long)gef::SplitPointStrategy::APPROXIMATE_SPLIT_POINT, 2048})
        ->Args({(long)file_idx, (long)gef::SplitPointStrategy::OPTIMAL_SPLIT_POINT, 2048})
        ->Args({(long)file_idx, (long)gef::SplitPointStrategy::APPROXIMATE_SPLIT_POINT, 4096})
        ->Args({(long)file_idx, (long)gef::SplitPointStrategy::OPTIMAL_SPLIT_POINT, 4096})
        ->Args({(long)file_idx, (long)gef::SplitPointStrategy::APPROXIMATE_SPLIT_POINT, 8192})
        ->Args({(long)file_idx, (long)gef::SplitPointStrategy::OPTIMAL_SPLIT_POINT, 8192})
        ->Args({(long)file_idx, (long)gef::SplitPointStrategy::APPROXIMATE_SPLIT_POINT, 16384})
        ->Args({(long)file_idx, (long)gef::SplitPointStrategy::OPTIMAL_SPLIT_POINT, 16384})
        ->Args({(long)file_idx, (long)gef::SplitPointStrategy::APPROXIMATE_SPLIT_POINT, 32768})
        ->Args({(long)file_idx, (long)gef::SplitPointStrategy::OPTIMAL_SPLIT_POINT, 32768})
        ->ArgNames({"file_idx", "strategy", "partition_size"});
    
    BENCHMARK_REGISTER_F(FileBasedCompressionBenchmark, B_GEF_NO_RLE_Compression)
        ->Args({(long)file_idx, (long)gef::SplitPointStrategy::APPROXIMATE_SPLIT_POINT, 512})
        ->Args({(long)file_idx, (long)gef::SplitPointStrategy::OPTIMAL_SPLIT_POINT, 512})
        ->Args({(long)file_idx, (long)gef::SplitPointStrategy::APPROXIMATE_SPLIT_POINT, 1024})
        ->Args({(long)file_idx, (long)gef::SplitPointStrategy::OPTIMAL_SPLIT_POINT, 1024})
        ->Args({(long)file_idx, (long)gef::SplitPointStrategy::APPROXIMATE_SPLIT_POINT, 2048})
        ->Args({(long)file_idx, (long)gef::SplitPointStrategy::OPTIMAL_SPLIT_POINT, 2048})
        ->Args({(long)file_idx, (long)gef::SplitPointStrategy::APPROXIMATE_SPLIT_POINT, 4096})
        ->Args({(long)file_idx, (long)gef::SplitPointStrategy::OPTIMAL_SPLIT_POINT, 4096})
        ->Args({(long)file_idx, (long)gef::SplitPointStrategy::APPROXIMATE_SPLIT_POINT, 8192})
        ->Args({(long)file_idx, (long)gef::SplitPointStrategy::OPTIMAL_SPLIT_POINT, 8192})
        ->Args({(long)file_idx, (long)gef::SplitPointStrategy::APPROXIMATE_SPLIT_POINT, 16384})
        ->Args({(long)file_idx, (long)gef::SplitPointStrategy::OPTIMAL_SPLIT_POINT, 16384})
        ->Args({(long)file_idx, (long)gef::SplitPointStrategy::APPROXIMATE_SPLIT_POINT, 32768})
        ->Args({(long)file_idx, (long)gef::SplitPointStrategy::OPTIMAL_SPLIT_POINT, 32768})
        ->ArgNames({"file_idx", "strategy", "partition_size"});
    
    BENCHMARK_REGISTER_F(FileBasedCompressionBenchmark, U_GEF_Compression)
        ->Args({(long)file_idx, (long)gef::SplitPointStrategy::APPROXIMATE_SPLIT_POINT, 512})
        ->Args({(long)file_idx, (long)gef::SplitPointStrategy::OPTIMAL_SPLIT_POINT, 512})
        ->Args({(long)file_idx, (long)gef::SplitPointStrategy::APPROXIMATE_SPLIT_POINT, 1024})
        ->Args({(long)file_idx, (long)gef::SplitPointStrategy::OPTIMAL_SPLIT_POINT, 1024})
        ->Args({(long)file_idx, (long)gef::SplitPointStrategy::APPROXIMATE_SPLIT_POINT, 2048})
        ->Args({(long)file_idx, (long)gef::SplitPointStrategy::OPTIMAL_SPLIT_POINT, 2048})
        ->Args({(long)file_idx, (long)gef::SplitPointStrategy::APPROXIMATE_SPLIT_POINT, 4096})
        ->Args({(long)file_idx, (long)gef::SplitPointStrategy::OPTIMAL_SPLIT_POINT, 4096})
        ->Args({(long)file_idx, (long)gef::SplitPointStrategy::APPROXIMATE_SPLIT_POINT, 8192})
        ->Args({(long)file_idx, (long)gef::SplitPointStrategy::OPTIMAL_SPLIT_POINT, 8192})
        ->Args({(long)file_idx, (long)gef::SplitPointStrategy::APPROXIMATE_SPLIT_POINT, 16384})
        ->Args({(long)file_idx, (long)gef::SplitPointStrategy::OPTIMAL_SPLIT_POINT, 16384})
        ->Args({(long)file_idx, (long)gef::SplitPointStrategy::APPROXIMATE_SPLIT_POINT, 32768})
        ->Args({(long)file_idx, (long)gef::SplitPointStrategy::OPTIMAL_SPLIT_POINT, 32768})
        ->ArgNames({"file_idx", "strategy", "partition_size"});
    
    BENCHMARK_REGISTER_F(FileBasedCompressionBenchmark, RLE_GEF_Compression)
        ->Args({(long)file_idx, 512})
        ->Args({(long)file_idx, 1024})
        ->Args({(long)file_idx, 2048})
        ->Args({(long)file_idx, 4096})
        ->Args({(long)file_idx, 8192})
        ->Args({(long)file_idx, 16384})
        ->Args({(long)file_idx, 32768})
        ->ArgNames({"file_idx", "partition_size"});

    // Lookup
    BENCHMARK_REGISTER_F(FileBasedCompressionBenchmark, B_GEF_Lookup)
        ->Args({(long)file_idx, (long)gef::SplitPointStrategy::APPROXIMATE_SPLIT_POINT, 512})
        ->Args({(long)file_idx, (long)gef::SplitPointStrategy::OPTIMAL_SPLIT_POINT, 512})
        ->Args({(long)file_idx, (long)gef::SplitPointStrategy::APPROXIMATE_SPLIT_POINT, 1024})
        ->Args({(long)file_idx, (long)gef::SplitPointStrategy::OPTIMAL_SPLIT_POINT, 1024})
        ->Args({(long)file_idx, (long)gef::SplitPointStrategy::APPROXIMATE_SPLIT_POINT, 2048})
        ->Args({(long)file_idx, (long)gef::SplitPointStrategy::OPTIMAL_SPLIT_POINT, 2048})
        ->Args({(long)file_idx, (long)gef::SplitPointStrategy::APPROXIMATE_SPLIT_POINT, 4096})
        ->Args({(long)file_idx, (long)gef::SplitPointStrategy::OPTIMAL_SPLIT_POINT, 4096})
        ->Args({(long)file_idx, (long)gef::SplitPointStrategy::APPROXIMATE_SPLIT_POINT, 8192})
        ->Args({(long)file_idx, (long)gef::SplitPointStrategy::OPTIMAL_SPLIT_POINT, 8192})
        ->Args({(long)file_idx, (long)gef::SplitPointStrategy::APPROXIMATE_SPLIT_POINT, 16384})
        ->Args({(long)file_idx, (long)gef::SplitPointStrategy::OPTIMAL_SPLIT_POINT, 16384})
        ->Args({(long)file_idx, (long)gef::SplitPointStrategy::APPROXIMATE_SPLIT_POINT, 32768})
        ->Args({(long)file_idx, (long)gef::SplitPointStrategy::OPTIMAL_SPLIT_POINT, 32768})
        ->ArgNames({"file_idx", "strategy", "partition_size"});
    
    BENCHMARK_REGISTER_F(FileBasedCompressionBenchmark, B_GEF_NO_RLE_Lookup)
        ->Args({(long)file_idx, (long)gef::SplitPointStrategy::APPROXIMATE_SPLIT_POINT, 512})
        ->Args({(long)file_idx, (long)gef::SplitPointStrategy::OPTIMAL_SPLIT_POINT, 512})
        ->Args({(long)file_idx, (long)gef::SplitPointStrategy::APPROXIMATE_SPLIT_POINT, 1024})
        ->Args({(long)file_idx, (long)gef::SplitPointStrategy::OPTIMAL_SPLIT_POINT, 1024})
        ->Args({(long)file_idx, (long)gef::SplitPointStrategy::APPROXIMATE_SPLIT_POINT, 2048})
        ->Args({(long)file_idx, (long)gef::SplitPointStrategy::OPTIMAL_SPLIT_POINT, 2048})
        ->Args({(long)file_idx, (long)gef::SplitPointStrategy::APPROXIMATE_SPLIT_POINT, 4096})
        ->Args({(long)file_idx, (long)gef::SplitPointStrategy::OPTIMAL_SPLIT_POINT, 4096})
        ->Args({(long)file_idx, (long)gef::SplitPointStrategy::APPROXIMATE_SPLIT_POINT, 8192})
        ->Args({(long)file_idx, (long)gef::SplitPointStrategy::OPTIMAL_SPLIT_POINT, 8192})
        ->Args({(long)file_idx, (long)gef::SplitPointStrategy::APPROXIMATE_SPLIT_POINT, 16384})
        ->Args({(long)file_idx, (long)gef::SplitPointStrategy::OPTIMAL_SPLIT_POINT, 16384})
        ->Args({(long)file_idx, (long)gef::SplitPointStrategy::APPROXIMATE_SPLIT_POINT, 32768})
        ->Args({(long)file_idx, (long)gef::SplitPointStrategy::OPTIMAL_SPLIT_POINT, 32768})
        ->ArgNames({"file_idx", "strategy", "partition_size"});
    
    BENCHMARK_REGISTER_F(FileBasedCompressionBenchmark, U_GEF_Lookup)
        ->Args({(long)file_idx, (long)gef::SplitPointStrategy::APPROXIMATE_SPLIT_POINT, 512})
        ->Args({(long)file_idx, (long)gef::SplitPointStrategy::OPTIMAL_SPLIT_POINT, 512})
        ->Args({(long)file_idx, (long)gef::SplitPointStrategy::APPROXIMATE_SPLIT_POINT, 1024})
        ->Args({(long)file_idx, (long)gef::SplitPointStrategy::OPTIMAL_SPLIT_POINT, 1024})
        ->Args({(long)file_idx, (long)gef::SplitPointStrategy::APPROXIMATE_SPLIT_POINT, 2048})
        ->Args({(long)file_idx, (long)gef::SplitPointStrategy::OPTIMAL_SPLIT_POINT, 2048})
        ->Args({(long)file_idx, (long)gef::SplitPointStrategy::APPROXIMATE_SPLIT_POINT, 4096})
        ->Args({(long)file_idx, (long)gef::SplitPointStrategy::OPTIMAL_SPLIT_POINT, 4096})
        ->Args({(long)file_idx, (long)gef::SplitPointStrategy::APPROXIMATE_SPLIT_POINT, 8192})
        ->Args({(long)file_idx, (long)gef::SplitPointStrategy::OPTIMAL_SPLIT_POINT, 8192})
        ->Args({(long)file_idx, (long)gef::SplitPointStrategy::APPROXIMATE_SPLIT_POINT, 16384})
        ->Args({(long)file_idx, (long)gef::SplitPointStrategy::OPTIMAL_SPLIT_POINT, 16384})
        ->Args({(long)file_idx, (long)gef::SplitPointStrategy::APPROXIMATE_SPLIT_POINT, 32768})
        ->Args({(long)file_idx, (long)gef::SplitPointStrategy::OPTIMAL_SPLIT_POINT, 32768})
        ->ArgNames({"file_idx", "strategy", "partition_size"});
    
    BENCHMARK_REGISTER_F(FileBasedCompressionBenchmark, RLE_GEF_Lookup)
        ->Args({(long)file_idx, 512})
        ->Args({(long)file_idx, 1024})
        ->Args({(long)file_idx, 2048})
        ->Args({(long)file_idx, 4096})
        ->Args({(long)file_idx, 8192})
        ->Args({(long)file_idx, 16384})
        ->Args({(long)file_idx, 32768})
        ->ArgNames({"file_idx", "partition_size"});

    // Serialization Space (only need 1 iteration to measure size)
    BENCHMARK_REGISTER_F(FileBasedCompressionBenchmark, B_GEF_Serialization_Space)
        ->Args({(long)file_idx, (long)gef::SplitPointStrategy::APPROXIMATE_SPLIT_POINT, 512})
        ->Args({(long)file_idx, (long)gef::SplitPointStrategy::OPTIMAL_SPLIT_POINT, 512})
        ->Args({(long)file_idx, (long)gef::SplitPointStrategy::APPROXIMATE_SPLIT_POINT, 1024})
        ->Args({(long)file_idx, (long)gef::SplitPointStrategy::OPTIMAL_SPLIT_POINT, 1024})
        ->Args({(long)file_idx, (long)gef::SplitPointStrategy::APPROXIMATE_SPLIT_POINT, 2048})
        ->Args({(long)file_idx, (long)gef::SplitPointStrategy::OPTIMAL_SPLIT_POINT, 2048})
        ->Args({(long)file_idx, (long)gef::SplitPointStrategy::APPROXIMATE_SPLIT_POINT, 4096})
        ->Args({(long)file_idx, (long)gef::SplitPointStrategy::OPTIMAL_SPLIT_POINT, 4096})
        ->Args({(long)file_idx, (long)gef::SplitPointStrategy::APPROXIMATE_SPLIT_POINT, 8192})
        ->Args({(long)file_idx, (long)gef::SplitPointStrategy::OPTIMAL_SPLIT_POINT, 8192})
        ->Args({(long)file_idx, (long)gef::SplitPointStrategy::APPROXIMATE_SPLIT_POINT, 16384})
        ->Args({(long)file_idx, (long)gef::SplitPointStrategy::OPTIMAL_SPLIT_POINT, 16384})
        ->Args({(long)file_idx, (long)gef::SplitPointStrategy::APPROXIMATE_SPLIT_POINT, 32768})
        ->Args({(long)file_idx, (long)gef::SplitPointStrategy::OPTIMAL_SPLIT_POINT, 32768})
        ->ArgNames({"file_idx", "strategy", "partition_size"})
        ->Iterations(1);
    
    BENCHMARK_REGISTER_F(FileBasedCompressionBenchmark, B_GEF_NO_RLE_Serialization_Space)
        ->Args({(long)file_idx, (long)gef::SplitPointStrategy::APPROXIMATE_SPLIT_POINT, 512})
        ->Args({(long)file_idx, (long)gef::SplitPointStrategy::OPTIMAL_SPLIT_POINT, 512})
        ->Args({(long)file_idx, (long)gef::SplitPointStrategy::APPROXIMATE_SPLIT_POINT, 1024})
        ->Args({(long)file_idx, (long)gef::SplitPointStrategy::OPTIMAL_SPLIT_POINT, 1024})
        ->Args({(long)file_idx, (long)gef::SplitPointStrategy::APPROXIMATE_SPLIT_POINT, 2048})
        ->Args({(long)file_idx, (long)gef::SplitPointStrategy::OPTIMAL_SPLIT_POINT, 2048})
        ->Args({(long)file_idx, (long)gef::SplitPointStrategy::APPROXIMATE_SPLIT_POINT, 4096})
        ->Args({(long)file_idx, (long)gef::SplitPointStrategy::OPTIMAL_SPLIT_POINT, 4096})
        ->Args({(long)file_idx, (long)gef::SplitPointStrategy::APPROXIMATE_SPLIT_POINT, 8192})
        ->Args({(long)file_idx, (long)gef::SplitPointStrategy::OPTIMAL_SPLIT_POINT, 8192})
        ->Args({(long)file_idx, (long)gef::SplitPointStrategy::APPROXIMATE_SPLIT_POINT, 16384})
        ->Args({(long)file_idx, (long)gef::SplitPointStrategy::OPTIMAL_SPLIT_POINT, 16384})
        ->Args({(long)file_idx, (long)gef::SplitPointStrategy::APPROXIMATE_SPLIT_POINT, 32768})
        ->Args({(long)file_idx, (long)gef::SplitPointStrategy::OPTIMAL_SPLIT_POINT, 32768})
        ->ArgNames({"file_idx", "strategy", "partition_size"})
        ->Iterations(1);
    
    BENCHMARK_REGISTER_F(FileBasedCompressionBenchmark, U_GEF_Serialization_Space)
        ->Args({(long)file_idx, (long)gef::SplitPointStrategy::APPROXIMATE_SPLIT_POINT, 512})
        ->Args({(long)file_idx, (long)gef::SplitPointStrategy::OPTIMAL_SPLIT_POINT, 512})
        ->Args({(long)file_idx, (long)gef::SplitPointStrategy::APPROXIMATE_SPLIT_POINT, 1024})
        ->Args({(long)file_idx, (long)gef::SplitPointStrategy::OPTIMAL_SPLIT_POINT, 1024})
        ->Args({(long)file_idx, (long)gef::SplitPointStrategy::APPROXIMATE_SPLIT_POINT, 2048})
        ->Args({(long)file_idx, (long)gef::SplitPointStrategy::OPTIMAL_SPLIT_POINT, 2048})
        ->Args({(long)file_idx, (long)gef::SplitPointStrategy::APPROXIMATE_SPLIT_POINT, 4096})
        ->Args({(long)file_idx, (long)gef::SplitPointStrategy::OPTIMAL_SPLIT_POINT, 4096})
        ->Args({(long)file_idx, (long)gef::SplitPointStrategy::APPROXIMATE_SPLIT_POINT, 8192})
        ->Args({(long)file_idx, (long)gef::SplitPointStrategy::OPTIMAL_SPLIT_POINT, 8192})
        ->Args({(long)file_idx, (long)gef::SplitPointStrategy::APPROXIMATE_SPLIT_POINT, 16384})
        ->Args({(long)file_idx, (long)gef::SplitPointStrategy::OPTIMAL_SPLIT_POINT, 16384})
        ->Args({(long)file_idx, (long)gef::SplitPointStrategy::APPROXIMATE_SPLIT_POINT, 32768})
        ->Args({(long)file_idx, (long)gef::SplitPointStrategy::OPTIMAL_SPLIT_POINT, 32768})
        ->ArgNames({"file_idx", "strategy", "partition_size"})
        ->Iterations(1);
    
    BENCHMARK_REGISTER_F(FileBasedCompressionBenchmark, RLE_GEF_Serialization_Space)
        ->Args({(long)file_idx, 512})
        ->Args({(long)file_idx, 1024})
        ->Args({(long)file_idx, 2048})
        ->Args({(long)file_idx, 4096})
        ->Args({(long)file_idx, 8192})
        ->Args({(long)file_idx, 16384})
        ->Args({(long)file_idx, 32768})
        ->ArgNames({"file_idx", "partition_size"})
        ->Iterations(1);

    // Decompression Throughput
    BENCHMARK_REGISTER_F(FileBasedCompressionBenchmark, B_GEF_Decompression)
        ->Args({(long)file_idx, (long)gef::SplitPointStrategy::APPROXIMATE_SPLIT_POINT, 512})
        ->Args({(long)file_idx, (long)gef::SplitPointStrategy::OPTIMAL_SPLIT_POINT, 512})
        ->Args({(long)file_idx, (long)gef::SplitPointStrategy::APPROXIMATE_SPLIT_POINT, 1024})
        ->Args({(long)file_idx, (long)gef::SplitPointStrategy::OPTIMAL_SPLIT_POINT, 1024})
        ->Args({(long)file_idx, (long)gef::SplitPointStrategy::APPROXIMATE_SPLIT_POINT, 2048})
        ->Args({(long)file_idx, (long)gef::SplitPointStrategy::OPTIMAL_SPLIT_POINT, 2048})
        ->Args({(long)file_idx, (long)gef::SplitPointStrategy::APPROXIMATE_SPLIT_POINT, 4096})
        ->Args({(long)file_idx, (long)gef::SplitPointStrategy::OPTIMAL_SPLIT_POINT, 4096})
        ->Args({(long)file_idx, (long)gef::SplitPointStrategy::APPROXIMATE_SPLIT_POINT, 8192})
        ->Args({(long)file_idx, (long)gef::SplitPointStrategy::OPTIMAL_SPLIT_POINT, 8192})
        ->Args({(long)file_idx, (long)gef::SplitPointStrategy::APPROXIMATE_SPLIT_POINT, 16384})
        ->Args({(long)file_idx, (long)gef::SplitPointStrategy::OPTIMAL_SPLIT_POINT, 16384})
        ->Args({(long)file_idx, (long)gef::SplitPointStrategy::APPROXIMATE_SPLIT_POINT, 32768})
        ->Args({(long)file_idx, (long)gef::SplitPointStrategy::OPTIMAL_SPLIT_POINT, 32768})
        ->ArgNames({"file_idx", "strategy", "partition_size"});
    
    BENCHMARK_REGISTER_F(FileBasedCompressionBenchmark, B_GEF_NO_RLE_Decompression)
        ->Args({(long)file_idx, (long)gef::SplitPointStrategy::APPROXIMATE_SPLIT_POINT, 512})
        ->Args({(long)file_idx, (long)gef::SplitPointStrategy::OPTIMAL_SPLIT_POINT, 512})
        ->Args({(long)file_idx, (long)gef::SplitPointStrategy::APPROXIMATE_SPLIT_POINT, 1024})
        ->Args({(long)file_idx, (long)gef::SplitPointStrategy::OPTIMAL_SPLIT_POINT, 1024})
        ->Args({(long)file_idx, (long)gef::SplitPointStrategy::APPROXIMATE_SPLIT_POINT, 2048})
        ->Args({(long)file_idx, (long)gef::SplitPointStrategy::OPTIMAL_SPLIT_POINT, 2048})
        ->Args({(long)file_idx, (long)gef::SplitPointStrategy::APPROXIMATE_SPLIT_POINT, 4096})
        ->Args({(long)file_idx, (long)gef::SplitPointStrategy::OPTIMAL_SPLIT_POINT, 4096})
        ->Args({(long)file_idx, (long)gef::SplitPointStrategy::APPROXIMATE_SPLIT_POINT, 8192})
        ->Args({(long)file_idx, (long)gef::SplitPointStrategy::OPTIMAL_SPLIT_POINT, 8192})
        ->Args({(long)file_idx, (long)gef::SplitPointStrategy::APPROXIMATE_SPLIT_POINT, 16384})
        ->Args({(long)file_idx, (long)gef::SplitPointStrategy::OPTIMAL_SPLIT_POINT, 16384})
        ->Args({(long)file_idx, (long)gef::SplitPointStrategy::APPROXIMATE_SPLIT_POINT, 32768})
        ->Args({(long)file_idx, (long)gef::SplitPointStrategy::OPTIMAL_SPLIT_POINT, 32768})
        ->ArgNames({"file_idx", "strategy", "partition_size"});
    
    BENCHMARK_REGISTER_F(FileBasedCompressionBenchmark, U_GEF_Decompression)
        ->Args({(long)file_idx, (long)gef::SplitPointStrategy::APPROXIMATE_SPLIT_POINT, 512})
        ->Args({(long)file_idx, (long)gef::SplitPointStrategy::OPTIMAL_SPLIT_POINT, 512})
        ->Args({(long)file_idx, (long)gef::SplitPointStrategy::APPROXIMATE_SPLIT_POINT, 1024})
        ->Args({(long)file_idx, (long)gef::SplitPointStrategy::OPTIMAL_SPLIT_POINT, 1024})
        ->Args({(long)file_idx, (long)gef::SplitPointStrategy::APPROXIMATE_SPLIT_POINT, 2048})
        ->Args({(long)file_idx, (long)gef::SplitPointStrategy::OPTIMAL_SPLIT_POINT, 2048})
        ->Args({(long)file_idx, (long)gef::SplitPointStrategy::APPROXIMATE_SPLIT_POINT, 4096})
        ->Args({(long)file_idx, (long)gef::SplitPointStrategy::OPTIMAL_SPLIT_POINT, 4096})
        ->Args({(long)file_idx, (long)gef::SplitPointStrategy::APPROXIMATE_SPLIT_POINT, 8192})
        ->Args({(long)file_idx, (long)gef::SplitPointStrategy::OPTIMAL_SPLIT_POINT, 8192})
        ->Args({(long)file_idx, (long)gef::SplitPointStrategy::APPROXIMATE_SPLIT_POINT, 16384})
        ->Args({(long)file_idx, (long)gef::SplitPointStrategy::OPTIMAL_SPLIT_POINT, 16384})
        ->Args({(long)file_idx, (long)gef::SplitPointStrategy::APPROXIMATE_SPLIT_POINT, 32768})
        ->Args({(long)file_idx, (long)gef::SplitPointStrategy::OPTIMAL_SPLIT_POINT, 32768})
        ->ArgNames({"file_idx", "strategy", "partition_size"});
    
    BENCHMARK_REGISTER_F(FileBasedCompressionBenchmark, RLE_GEF_Decompression)
        ->Args({(long)file_idx, 512})
        ->Args({(long)file_idx, 1024})
        ->Args({(long)file_idx, 2048})
        ->Args({(long)file_idx, 4096})
        ->Args({(long)file_idx, 8192})
        ->Args({(long)file_idx, 16384})
        ->Args({(long)file_idx, 32768})
        ->ArgNames({"file_idx", "partition_size"});
}
#pragma endregion


#pragma region Main
int main(int argc, char** argv) {
    benchmark::Initialize(&argc, argv);

    // Parse custom arguments
    BitVectorImplementation selected_impl = BitVectorImplementation::SDSL;
    
    for (int i = 1; i < argc; ++i) {
        std::string arg = argv[i];
        if (arg == "--bitvector=sux") {
            selected_impl = BitVectorImplementation::SUX;
        } else if (arg == "--bitvector=sdsl") {
            selected_impl = BitVectorImplementation::SDSL;
        } else if (arg == "--bitvector=both") {
            std::cerr << "Error: --bitvector=both is not supported." << std::endl;
            std::cerr << "Please run the benchmark twice, once with --bitvector=sdsl and once with --bitvector=sux" << std::endl;
            return 1;
        } else if (arg.rfind("--", 0) != 0) {
            // Not a flag, assume it's an input file
            g_input_files.push_back(arg);
        }
    }

    if (g_input_files.empty()) {
        std::cerr << "Usage: " << argv[0] << " [--bitvector=sdsl|sux] <input_file1> [input_file2 ...] [benchmark_flags]" << std::endl;
        std::cerr << "Example: " << argv[0] << " --bitvector=sux data/file1.bin --benchmark_format=json" << std::endl;
        std::cerr << "\nBitVector Implementation Options:" << std::endl;
        std::cerr << "  --bitvector=sdsl   Use SDSL bitvectors (default)" << std::endl;
        std::cerr << "  --bitvector=sux    Use SUX bitvectors" << std::endl;
        std::cerr << "\nTo compare both implementations, run the benchmark twice with different flags." << std::endl;
        return 1;
    }

    // Set the factory once before registering benchmarks
    setFactory(selected_impl);
    std::cout << "Running benchmarks with " << implementationToString(selected_impl) << " bitvector implementation..." << std::endl;
    
    // Register benchmarks for all input files
    for (size_t i = 0; i < g_input_files.size(); ++i) {
        RegisterBenchmarksForFile(i);
    }

    benchmark::RunSpecifiedBenchmarks();

    return 0;
}
#pragma endregion