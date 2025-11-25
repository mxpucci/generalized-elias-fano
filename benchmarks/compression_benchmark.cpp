#include <benchmark/benchmark.h>
#include "gef/B_GEF.hpp"
#include "gef/U_GEF.hpp"
#include "gef/RLE_GEF.hpp"
#include "gef/B_GEF_STAR.hpp"
#include "gef/UniformedPartitioner.hpp"
#include "datastructures/SDSLBitVectorFactory.hpp"
#include "datastructures/SUXBitVectorFactory.hpp"
#include "datastructures/PastaBitVectorFactory.hpp"
#include "gef/CompressionProfile.hpp"
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
    // Primary constructor matching gef::U_GEF expectations
    U_GEF_Wrapper(const std::vector<T>& data,
                  const std::shared_ptr<IBitVectorFactory>& factory,
                  gef::SplitPointStrategy strategy)
        : gef::U_GEF<T>(factory, data, strategy) {}

    // Convenience overload accepting a lightweight span-view
    U_GEF_Wrapper(gef::Span<const T> data,
                  const std::shared_ptr<IBitVectorFactory>& factory,
                  gef::SplitPointStrategy strategy)
        : gef::U_GEF<T>(factory, data, strategy) {}

    U_GEF_Wrapper() : gef::U_GEF<T>() {}
};

template<typename T>
struct RLE_GEF_Wrapper : public gef::RLE_GEF<T> {
    RLE_GEF_Wrapper(const std::vector<T>& data,
                    const std::shared_ptr<IBitVectorFactory>& factory)
        : gef::RLE_GEF<T>(factory, data) {}

    RLE_GEF_Wrapper(gef::Span<const T> data,
                    const std::shared_ptr<IBitVectorFactory>& factory)
        : gef::RLE_GEF<T>(factory, data) {}

    RLE_GEF_Wrapper() : gef::RLE_GEF<T>() {}
};

template<typename T>
struct B_GEF_Wrapper : public gef::B_GEF<T> {
    B_GEF_Wrapper(gef::Span<const T> data,
                  const std::shared_ptr<IBitVectorFactory>& factory,
                  gef::SplitPointStrategy strategy,
                  CompressionBuildMetrics* metrics = nullptr)
        : gef::B_GEF<T>(factory, data, strategy, metrics) {}

    B_GEF_Wrapper(const std::vector<T>& data,
                  const std::shared_ptr<IBitVectorFactory>& factory,
                  gef::SplitPointStrategy strategy,
                  CompressionBuildMetrics* metrics = nullptr)
        : gef::B_GEF<T>(factory, data, strategy, metrics) {}

    B_GEF_Wrapper() : gef::B_GEF<T>() {}
};

template<typename T>
struct B_GEF_NO_RLE_Wrapper : public gef::B_GEF_STAR<T> {
    B_GEF_NO_RLE_Wrapper(const std::vector<T>& data,
                         const std::shared_ptr<IBitVectorFactory>& factory,
                         gef::SplitPointStrategy strategy)
        : gef::B_GEF_STAR<T>(factory, data, strategy) {}

    B_GEF_NO_RLE_Wrapper(gef::Span<const T> data,
                         const std::shared_ptr<IBitVectorFactory>& factory,
                         gef::SplitPointStrategy strategy)
        : gef::B_GEF_STAR<T>(factory, data, strategy) {}

    B_GEF_NO_RLE_Wrapper() : gef::B_GEF_STAR<T>() {}
};

std::vector<std::string> g_input_files;
std::shared_ptr<IBitVectorFactory> g_factory = std::make_shared<PastaBitVectorFactory>();
std::string g_factory_name = "PASTA";

enum class BitVectorImplementation {
    PASTA,
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
        case BitVectorImplementation::PASTA: return "PASTA";
        case BitVectorImplementation::SDSL: return "SDSL";
        case BitVectorImplementation::SUX: return "SUX";
        default: return "UNKNOWN";
    }
}

void setFactory(BitVectorImplementation impl) {
    switch (impl) {
        case BitVectorImplementation::PASTA:
            g_factory = std::make_shared<PastaBitVectorFactory>();
            g_factory_name = "PASTA";
            break;
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
class UniformedPartitionerBenchmark : public benchmark::Fixture {
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

// Reduced set for faster execution
// const std::vector<size_t> PARTITION_SIZES = {
//     65536, 1048576, 8388608
// };
// Full set:
const std::vector<size_t> PARTITION_SIZES = {
    8192, 16384, 32768, 65536, 131072, 262144, 524288, 1048576, 2097152, 4194304, 8388608
};

// Partition size for restricted benchmarks
const size_t FIXED_PARTITION_SIZE = 1048576;

const size_t DEFAULT_PARTITION_SIZE = FIXED_PARTITION_SIZE; // Updated to match the restriction
const gef::SplitPointStrategy DEFAULT_LOOKUP_STRATEGY = gef::SplitPointStrategy::APPROXIMATE_SPLIT_POINT;

// Reduced set for faster execution
const std::vector<size_t> GET_ELEMENTS_RANGES = {
    10, 10240, 1000000
};
// Full set:
// const std::vector<size_t> GET_ELEMENTS_RANGES = {
//     10, 20, 40, 80, 160, 320, 640, 1280, 2560, 5120, 10240,
//     20480, 40960, 81920, 163840, 327680, 655360, 1000000
// };

BENCHMARK_DEFINE_F(UniformedPartitionerBenchmark, B_GEF_Compression)(benchmark::State& state) {
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

BENCHMARK_DEFINE_F(UniformedPartitionerBenchmark, B_GEF_NO_RLE_Compression)(benchmark::State& state) {
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

BENCHMARK_DEFINE_F(UniformedPartitionerBenchmark, U_GEF_Compression)(benchmark::State& state) {
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

BENCHMARK_DEFINE_F(UniformedPartitionerBenchmark, RLE_GEF_Compression)(benchmark::State& state) {
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

const size_t NUM_LOOKUPS = 1e6;

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
        int64_t accumulator = 0;
        for (size_t i = 0; i < NUM_LOOKUPS; ++i) {
            accumulator += compressor[random_indices[i]];
        }
        benchmark::DoNotOptimize(accumulator);
    }

    state.SetItemsProcessed(NUM_LOOKUPS * state.iterations());
    double bytes_processed = static_cast<double>(state.items_processed() * sizeof(int64_t));
    state.counters["lookup_throughput_MBs"] = benchmark::Counter(bytes_processed, benchmark::Counter::kIsRate, benchmark::Counter::kIs1024);
}


// ============================================================================
// Benchmark Definitions - Type 4: Size-In-Bytes Measurement
// ============================================================================
template<template<typename> class CompressorWrapper, typename... Args>
void SizeInBytesBenchmark(benchmark::State& state, const std::vector<int64_t>& data, size_t partition_size, Args... args) {
    if (data.empty()) {
        state.SkipWithError("Input data is empty for size benchmark.");
        return;
    }

    gef::UniformedPartitioner<int64_t, CompressorWrapper<int64_t>, Args...> compressor(data, partition_size, args...);
    size_t bytes_used = compressor.size_in_bytes();
    for (auto _ : state) {
        benchmark::DoNotOptimize(bytes_used);
    }

    state.counters["size_in_bytes"] = bytes_used;
    state.counters["bpi"] = static_cast<double>(bytes_used * 8) / data.size();
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
    const size_t total = data.size();
    std::vector<int64_t> decompressed(total);
    for (auto _ : state) {
        // Decompress ALL elements using optimized get_elements
        const size_t written = compressor.get_elements(0, total, decompressed);
        if (written != total) {
            state.SkipWithError("Decompression returned fewer elements than expected.");
            return;
        }
        benchmark::DoNotOptimize(decompressed.data());
    }
    
    // Report throughput
    state.SetItemsProcessed(data.size() * state.iterations());
    double bytes_processed = static_cast<double>(state.items_processed() * sizeof(int64_t));
    state.counters["decompression_throughput_MBs"] = benchmark::Counter(bytes_processed, benchmark::Counter::kIsRate, benchmark::Counter::kIs1024);
}


// ============================================================================
// Benchmark Definitions - Type 6: Partial get_elements Throughput
// ============================================================================
template<template<typename> class CompressorWrapper, typename... Args>
void GetElementsRangeBenchmark(benchmark::State& state,
                               const std::vector<int64_t>& data,
                               size_t partition_size,
                               size_t range,
                               Args... args) {
    if (data.empty()) {
        state.SkipWithError("Input data is empty for get_elements benchmark.");
        return;
    }
    if (range == 0) {
        state.SkipWithError("Range must be greater than zero for get_elements benchmark.");
        return;
    }
    if (range > data.size()) {
        state.SkipWithMessage("Range exceeds input size for get_elements benchmark; skipping.");
        return;
    }

    gef::UniformedPartitioner<int64_t, CompressorWrapper<int64_t>, Args...> compressor(data, partition_size, args...);
    std::vector<int64_t> buffer(range);
    
    // Precompute start indices to avoid RNG overhead during measurement
    std::mt19937_64 gen(1337);
    std::uniform_int_distribution<size_t> distrib(0, data.size() - range);
    
    // Use a buffer of random indices to cycle through
    // Google Benchmark runs loop many times, so we need enough indices
    // or just cycle through a moderate size buffer.
    const size_t NUM_INDICES = 10000;
    std::vector<size_t> start_indices(NUM_INDICES);
    for(size_t i=0; i<NUM_INDICES; ++i) {
        start_indices[i] = distrib(gen);
    }
    
    size_t idx = 0;
    for (auto _ : state) {
        const size_t start = start_indices[idx];
        const size_t written = compressor.get_elements(start, range, buffer);
        if (written != range) {
            state.SkipWithError("get_elements returned fewer elements than requested.");
            return;
        }
        benchmark::DoNotOptimize(buffer.data());
        
        idx++;
        if (idx >= NUM_INDICES) idx = 0;
    }

    state.SetItemsProcessed(range * state.iterations());
    double bytes_processed = static_cast<double>(state.items_processed() * sizeof(int64_t));
    state.counters["get_elements_throughput_MBs"] = benchmark::Counter(bytes_processed, benchmark::Counter::kIsRate, benchmark::Counter::kIs1024);
}

#pragma region Lookup Definitions and Registration
// ... [Rest of registration code remains same] ...
BENCHMARK_DEFINE_F(UniformedPartitionerBenchmark, B_GEF_Lookup)(benchmark::State& state) {
    gef::SplitPointStrategy strategy = static_cast<gef::SplitPointStrategy>(state.range(1));
    size_t partition_size = state.range(2);
    state.SetLabel(current_basename + "/" + g_factory_name + "/" + strategyToString(strategy) + "/" + std::to_string(partition_size));
    LookupBenchmark<B_GEF_Wrapper>(state, input_data, partition_size, g_factory, strategy);
}

BENCHMARK_DEFINE_F(UniformedPartitionerBenchmark, B_GEF_NO_RLE_Lookup)(benchmark::State& state) {
    gef::SplitPointStrategy strategy = static_cast<gef::SplitPointStrategy>(state.range(1));
    size_t partition_size = state.range(2);
    state.SetLabel(current_basename + "/" + g_factory_name + "/" + strategyToString(strategy) + "/" + std::to_string(partition_size));
    LookupBenchmark<B_GEF_NO_RLE_Wrapper>(state, input_data, partition_size, g_factory, strategy);
}

BENCHMARK_DEFINE_F(UniformedPartitionerBenchmark, U_GEF_Lookup)(benchmark::State& state) {
    gef::SplitPointStrategy strategy = static_cast<gef::SplitPointStrategy>(state.range(1));
    size_t partition_size = state.range(2);
    state.SetLabel(current_basename + "/" + g_factory_name + "/" + strategyToString(strategy) + "/" + std::to_string(partition_size));
    LookupBenchmark<U_GEF_Wrapper>(state, input_data, partition_size, g_factory, strategy);
}

BENCHMARK_DEFINE_F(UniformedPartitionerBenchmark, RLE_GEF_Lookup)(benchmark::State& state) {
    size_t partition_size = state.range(1);
    state.SetLabel(current_basename + "/" + g_factory_name + "/" + std::to_string(partition_size));
    LookupBenchmark<RLE_GEF_Wrapper>(state, input_data, partition_size, g_factory);
}

BENCHMARK_DEFINE_F(UniformedPartitionerBenchmark, B_GEF_SizeInBytes)(benchmark::State& state) {
    gef::SplitPointStrategy strategy = static_cast<gef::SplitPointStrategy>(state.range(1));
    size_t partition_size = state.range(2);
    state.SetLabel(current_basename + "/" + g_factory_name + "/" + strategyToString(strategy) + "/" + std::to_string(partition_size));
    SizeInBytesBenchmark<B_GEF_Wrapper>(state, input_data, partition_size, g_factory, strategy);
}

BENCHMARK_DEFINE_F(UniformedPartitionerBenchmark, B_GEF_NO_RLE_SizeInBytes)(benchmark::State& state) {
    gef::SplitPointStrategy strategy = static_cast<gef::SplitPointStrategy>(state.range(1));
    size_t partition_size = state.range(2);
    state.SetLabel(current_basename + "/" + g_factory_name + "/" + strategyToString(strategy) + "/" + std::to_string(partition_size));
    SizeInBytesBenchmark<B_GEF_NO_RLE_Wrapper>(state, input_data, partition_size, g_factory, strategy);
}

BENCHMARK_DEFINE_F(UniformedPartitionerBenchmark, U_GEF_SizeInBytes)(benchmark::State& state) {
    gef::SplitPointStrategy strategy = static_cast<gef::SplitPointStrategy>(state.range(1));
    size_t partition_size = state.range(2);
    state.SetLabel(current_basename + "/" + g_factory_name + "/" + strategyToString(strategy) + "/" + std::to_string(partition_size));
    SizeInBytesBenchmark<U_GEF_Wrapper>(state, input_data, partition_size, g_factory, strategy);
}

BENCHMARK_DEFINE_F(UniformedPartitionerBenchmark, RLE_GEF_SizeInBytes)(benchmark::State& state) {
    size_t partition_size = state.range(1);
    state.SetLabel(current_basename + "/" + g_factory_name + "/" + std::to_string(partition_size));
    SizeInBytesBenchmark<RLE_GEF_Wrapper>(state, input_data, partition_size, g_factory);
}

// Decompression Throughput Benchmark Definitions
BENCHMARK_DEFINE_F(UniformedPartitionerBenchmark, B_GEF_Decompression)(benchmark::State& state) {
    gef::SplitPointStrategy strategy = static_cast<gef::SplitPointStrategy>(state.range(1));
    size_t partition_size = state.range(2);
    state.SetLabel(current_basename + "/" + g_factory_name + "/" + strategyToString(strategy) + "/" + std::to_string(partition_size));
    DecompressionThroughputBenchmark<B_GEF_Wrapper>(state, input_data, partition_size, g_factory, strategy);
}

BENCHMARK_DEFINE_F(UniformedPartitionerBenchmark, B_GEF_NO_RLE_Decompression)(benchmark::State& state) {
    gef::SplitPointStrategy strategy = static_cast<gef::SplitPointStrategy>(state.range(1));
    size_t partition_size = state.range(2);
    state.SetLabel(current_basename + "/" + g_factory_name + "/" + strategyToString(strategy) + "/" + std::to_string(partition_size));
    DecompressionThroughputBenchmark<B_GEF_NO_RLE_Wrapper>(state, input_data, partition_size, g_factory, strategy);
}

BENCHMARK_DEFINE_F(UniformedPartitionerBenchmark, U_GEF_Decompression)(benchmark::State& state) {
    gef::SplitPointStrategy strategy = static_cast<gef::SplitPointStrategy>(state.range(1));
    size_t partition_size = state.range(2);
    state.SetLabel(current_basename + "/" + g_factory_name + "/" + strategyToString(strategy) + "/" + std::to_string(partition_size));
    DecompressionThroughputBenchmark<U_GEF_Wrapper>(state, input_data, partition_size, g_factory, strategy);
}

BENCHMARK_DEFINE_F(UniformedPartitionerBenchmark, RLE_GEF_Decompression)(benchmark::State& state) {
    size_t partition_size = state.range(1);
    state.SetLabel(current_basename + "/" + g_factory_name + "/" + std::to_string(partition_size));
    DecompressionThroughputBenchmark<RLE_GEF_Wrapper>(state, input_data, partition_size, g_factory);
}

// Partial get_elements throughput
BENCHMARK_DEFINE_F(UniformedPartitionerBenchmark, B_GEF_GetElements)(benchmark::State& state) {
    gef::SplitPointStrategy strategy = static_cast<gef::SplitPointStrategy>(state.range(1));
    size_t partition_size = state.range(2);
    size_t range = state.range(3);
    state.SetLabel(current_basename + "/" + g_factory_name + "/" + strategyToString(strategy) + "/" +
                   std::to_string(partition_size) + "/range=" + std::to_string(range));
    GetElementsRangeBenchmark<B_GEF_Wrapper>(state, input_data, partition_size, range, g_factory, strategy);
}

BENCHMARK_DEFINE_F(UniformedPartitionerBenchmark, B_GEF_NO_RLE_GetElements)(benchmark::State& state) {
    gef::SplitPointStrategy strategy = static_cast<gef::SplitPointStrategy>(state.range(1));
    size_t partition_size = state.range(2);
    size_t range = state.range(3);
    state.SetLabel(current_basename + "/" + g_factory_name + "/" + strategyToString(strategy) + "/" +
                   std::to_string(partition_size) + "/range=" + std::to_string(range));
    GetElementsRangeBenchmark<B_GEF_NO_RLE_Wrapper>(state, input_data, partition_size, range, g_factory, strategy);
}

BENCHMARK_DEFINE_F(UniformedPartitionerBenchmark, U_GEF_GetElements)(benchmark::State& state) {
    gef::SplitPointStrategy strategy = static_cast<gef::SplitPointStrategy>(state.range(1));
    size_t partition_size = state.range(2);
    size_t range = state.range(3);
    state.SetLabel(current_basename + "/" + g_factory_name + "/" + strategyToString(strategy) + "/" +
                   std::to_string(partition_size) + "/range=" + std::to_string(range));
    GetElementsRangeBenchmark<U_GEF_Wrapper>(state, input_data, partition_size, range, g_factory, strategy);
}

BENCHMARK_DEFINE_F(UniformedPartitionerBenchmark, RLE_GEF_GetElements)(benchmark::State& state) {
    size_t partition_size = state.range(1);
    size_t range = state.range(2);
    state.SetLabel(current_basename + "/" + g_factory_name + "/" + std::to_string(partition_size) +
                   "/range=" + std::to_string(range));
    GetElementsRangeBenchmark<RLE_GEF_Wrapper>(state, input_data, partition_size, range, g_factory);
}

void RegisterBenchmarksForFile(size_t file_idx) {
    const std::vector<int64_t> file_idx_args = {static_cast<int64_t>(file_idx)};
    const std::vector<int64_t> strategy_args = {
        static_cast<int64_t>(gef::SplitPointStrategy::APPROXIMATE_SPLIT_POINT),
        static_cast<int64_t>(gef::SplitPointStrategy::OPTIMAL_SPLIT_POINT)
    };
    
    // Args for all partition sizes (Compression benchmarks)
    std::vector<int64_t> partition_args;
    partition_args.reserve(PARTITION_SIZES.size());
    for (auto partition_size : PARTITION_SIZES) {
        partition_args.push_back(static_cast<int64_t>(partition_size));
    }
    
    // Args for fixed partition size (Lookup, Decompression, GetElements, SizeInBytes)
    const std::vector<int64_t> fixed_partition_args = {static_cast<int64_t>(FIXED_PARTITION_SIZE)};

    std::vector<int64_t> range_args;
    range_args.reserve(GET_ELEMENTS_RANGES.size());
    for (auto range : GET_ELEMENTS_RANGES) {
        range_args.push_back(static_cast<int64_t>(range));
    }

    // Argument lists for all partitions (Compression)
    const std::vector<std::vector<int64_t>> throughput_strategy_lists = {
        file_idx_args,
        strategy_args,
        partition_args
    };
    const std::vector<std::vector<int64_t>> partition_arg_lists = {
        file_idx_args,
        partition_args
    };

    // Argument lists for fixed partition (Decompression, GetElements, Lookup)
    const std::vector<std::vector<int64_t>> fixed_throughput_strategy_lists = {
        file_idx_args,
        strategy_args,
        fixed_partition_args
    };
    const std::vector<std::vector<int64_t>> fixed_partition_arg_lists = {
        file_idx_args,
        fixed_partition_args
    };
    const std::vector<std::vector<int64_t>> fixed_get_elements_strategy_lists = {
        file_idx_args,
        strategy_args,
        fixed_partition_args,
        range_args
    };
    const std::vector<std::vector<int64_t>> fixed_get_elements_partition_lists = {
        file_idx_args,
        fixed_partition_args,
        range_args
    };

    const std::vector<int64_t> default_strategy_args = {
        static_cast<int64_t>(file_idx),
        static_cast<int64_t>(DEFAULT_LOOKUP_STRATEGY),
        static_cast<int64_t>(DEFAULT_PARTITION_SIZE)
    };
    const std::vector<int64_t> default_partition_args = {
        static_cast<int64_t>(file_idx),
        static_cast<int64_t>(DEFAULT_PARTITION_SIZE)
    };

    // Compression - Run on ALL partition sizes
    BENCHMARK_REGISTER_F(UniformedPartitionerBenchmark, B_GEF_Compression)
        ->ArgsProduct(throughput_strategy_lists)
        ->ArgNames({"file_idx", "strategy", "partition_size"});

    BENCHMARK_REGISTER_F(UniformedPartitionerBenchmark, B_GEF_NO_RLE_Compression)
        ->ArgsProduct(throughput_strategy_lists)
        ->ArgNames({"file_idx", "strategy", "partition_size"});

    BENCHMARK_REGISTER_F(UniformedPartitionerBenchmark, U_GEF_Compression)
        ->ArgsProduct(throughput_strategy_lists)
        ->ArgNames({"file_idx", "strategy", "partition_size"});

    BENCHMARK_REGISTER_F(UniformedPartitionerBenchmark, RLE_GEF_Compression)
        ->ArgsProduct(partition_arg_lists)
        ->ArgNames({"file_idx", "partition_size"});

    // Lookup - Run only on FIXED partition size (2^20) and only if NOT OpenMP
    #ifndef _OPENMP
    BENCHMARK_REGISTER_F(UniformedPartitionerBenchmark, B_GEF_Lookup)
        ->ArgsProduct(fixed_throughput_strategy_lists)
        ->ArgNames({"file_idx", "strategy", "partition_size"})
        ->Iterations(1);

    BENCHMARK_REGISTER_F(UniformedPartitionerBenchmark, B_GEF_NO_RLE_Lookup)
        ->ArgsProduct(fixed_throughput_strategy_lists)
        ->ArgNames({"file_idx", "strategy", "partition_size"})
        ->Iterations(1);

    BENCHMARK_REGISTER_F(UniformedPartitionerBenchmark, U_GEF_Lookup)
        ->ArgsProduct(fixed_throughput_strategy_lists)
        ->ArgNames({"file_idx", "strategy", "partition_size"})
        ->Iterations(1);

    BENCHMARK_REGISTER_F(UniformedPartitionerBenchmark, RLE_GEF_Lookup)
        ->ArgsProduct(fixed_partition_arg_lists)
        ->ArgNames({"file_idx", "partition_size"})
        ->Iterations(1);
    #endif

    // Size benchmarks - Run only on FIXED partition size (DEFAULT_PARTITION_SIZE is updated)
    BENCHMARK_REGISTER_F(UniformedPartitionerBenchmark, B_GEF_SizeInBytes)
        ->ArgsProduct({
            file_idx_args,
            {static_cast<int64_t>(gef::SplitPointStrategy::APPROXIMATE_SPLIT_POINT),
             static_cast<int64_t>(gef::SplitPointStrategy::OPTIMAL_SPLIT_POINT)},
            fixed_partition_args
        })
        ->ArgNames({"file_idx", "strategy", "partition_size"})
        ->Iterations(1);
    
    BENCHMARK_REGISTER_F(UniformedPartitionerBenchmark, B_GEF_NO_RLE_SizeInBytes)
        ->ArgsProduct({
            file_idx_args,
            {static_cast<int64_t>(gef::SplitPointStrategy::APPROXIMATE_SPLIT_POINT),
             static_cast<int64_t>(gef::SplitPointStrategy::OPTIMAL_SPLIT_POINT)},
            fixed_partition_args
        })
        ->ArgNames({"file_idx", "strategy", "partition_size"})
        ->Iterations(1);
    
    BENCHMARK_REGISTER_F(UniformedPartitionerBenchmark, U_GEF_SizeInBytes)
        ->ArgsProduct({
            file_idx_args,
            {static_cast<int64_t>(gef::SplitPointStrategy::APPROXIMATE_SPLIT_POINT),
             static_cast<int64_t>(gef::SplitPointStrategy::OPTIMAL_SPLIT_POINT)},
            fixed_partition_args
        })
        ->ArgNames({"file_idx", "strategy", "partition_size"})
        ->Iterations(1);

    BENCHMARK_REGISTER_F(UniformedPartitionerBenchmark, RLE_GEF_SizeInBytes)
        ->ArgsProduct(fixed_partition_arg_lists)
        ->ArgNames({"file_idx", "partition_size"})
        ->Iterations(1);

    // Decompression Throughput - Run only on FIXED partition size (2^20)
    BENCHMARK_REGISTER_F(UniformedPartitionerBenchmark, B_GEF_Decompression)
        ->ArgsProduct(fixed_throughput_strategy_lists)
        ->ArgNames({"file_idx", "strategy", "partition_size"});

    BENCHMARK_REGISTER_F(UniformedPartitionerBenchmark, B_GEF_NO_RLE_Decompression)
        ->ArgsProduct(fixed_throughput_strategy_lists)
        ->ArgNames({"file_idx", "strategy", "partition_size"});

    BENCHMARK_REGISTER_F(UniformedPartitionerBenchmark, U_GEF_Decompression)
        ->ArgsProduct(fixed_throughput_strategy_lists)
        ->ArgNames({"file_idx", "strategy", "partition_size"});

    BENCHMARK_REGISTER_F(UniformedPartitionerBenchmark, RLE_GEF_Decompression)
        ->ArgsProduct(fixed_partition_arg_lists)
        ->ArgNames({"file_idx", "partition_size"});

    // Partial get_elements throughput - Run only on FIXED partition size (2^20)
    BENCHMARK_REGISTER_F(UniformedPartitionerBenchmark, B_GEF_GetElements)
        ->ArgsProduct(fixed_get_elements_strategy_lists)
        ->ArgNames({"file_idx", "strategy", "partition_size", "range"});

    BENCHMARK_REGISTER_F(UniformedPartitionerBenchmark, B_GEF_NO_RLE_GetElements)
        ->ArgsProduct(fixed_get_elements_strategy_lists)
        ->ArgNames({"file_idx", "strategy", "partition_size", "range"});

    BENCHMARK_REGISTER_F(UniformedPartitionerBenchmark, U_GEF_GetElements)
        ->ArgsProduct(fixed_get_elements_strategy_lists)
        ->ArgNames({"file_idx", "strategy", "partition_size", "range"});

    BENCHMARK_REGISTER_F(UniformedPartitionerBenchmark, RLE_GEF_GetElements)
        ->ArgsProduct(fixed_get_elements_partition_lists)
        ->ArgNames({"file_idx", "partition_size", "range"});
}
#pragma endregion


#pragma region Main
int main(int argc, char** argv) {
    benchmark::Initialize(&argc, argv);

    // Parse custom arguments
    BitVectorImplementation selected_impl = BitVectorImplementation::PASTA;
    
    for (int i = 1; i < argc; ++i) {
        std::string arg = argv[i];
        if (arg == "--bitvector=sux") {
            selected_impl = BitVectorImplementation::SUX;
        } else if (arg == "--bitvector=sdsl") {
            selected_impl = BitVectorImplementation::SDSL;
        } else if (arg == "--bitvector=pasta") {
            selected_impl = BitVectorImplementation::PASTA;
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
        std::cerr << "Usage: " << argv[0] << " [--bitvector=pasta|sdsl|sux] <input_file1> [input_file2 ...] [benchmark_flags]" << std::endl;
        std::cerr << "Example: " << argv[0] << " --bitvector=sux data/file1.bin --benchmark_format=json" << std::endl;
        std::cerr << "\nBitVector Implementation Options:" << std::endl;
        std::cerr << "  --bitvector=pasta  Use Pasta bitvectors (default)" << std::endl;
        std::cerr << "  --bitvector=sdsl   Use SDSL bitvectors" << std::endl;
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
