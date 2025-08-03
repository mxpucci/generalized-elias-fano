#include <benchmark/benchmark.h>
#include "gef/B_GEF.hpp"
#include "gef/U_GEF.hpp"
#include "gef/RLE_GEF.hpp"
#include "gef/B_GEF_NO_RLE.hpp"
#include "gef/UniformedPartitioner.hpp"
#include "datastructures/SDSLBitVectorFactory.hpp"
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
    U_GEF_Wrapper(const std::vector<T>& data, std::shared_ptr<IBitVectorFactory> factory)
        : gef::U_GEF<T>(factory, data) {}
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
struct B_GEF_NO_RLE_Wrapper : public gef::B_GEF_NO_RLE<T> {
    B_GEF_NO_RLE_Wrapper(const std::vector<T>& data, std::shared_ptr<IBitVectorFactory> factory, gef::SplitPointStrategy strategy)
        : gef::B_GEF_NO_RLE<T>(factory, data, strategy) {}
    B_GEF_NO_RLE_Wrapper() : gef::B_GEF_NO_RLE<T>() {}
};

std::vector<std::string> g_input_files;
std::shared_ptr<IBitVectorFactory> g_factory = std::make_shared<SDSLBitVectorFactory>();

std::string strategyToString(gef::SplitPointStrategy strategy) {
    switch (strategy) {
        case gef::SplitPointStrategy::APPROXIMATE_SPLIT_POINT: return "APPROXIMATE";
        case gef::SplitPointStrategy::BRUTE_FORCE_SPLIT_POINT: return "BRUTE_FORCE";
        case gef::SplitPointStrategy::BINARY_SEARCH_SPLIT_POINT: return "BINARY_SEARCH";
        default: return "UNKNOWN";
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

const size_t FIXED_PARTITION_SIZE = 8192;

BENCHMARK_DEFINE_F(FileBasedCompressionBenchmark, B_GEF_Compression)(benchmark::State& state) {
    gef::SplitPointStrategy strategy = static_cast<gef::SplitPointStrategy>(state.range(1));
    state.SetLabel(current_basename + "/" + strategyToString(strategy));

    for (auto _ : state) {
        state.PauseTiming();
        std::vector<int64_t> data_copy = input_data;
        state.ResumeTiming();
        gef::UniformedPartitioner<int64_t, B_GEF_Wrapper<int64_t>, std::shared_ptr<IBitVectorFactory>, gef::SplitPointStrategy> compressor(
            data_copy, FIXED_PARTITION_SIZE, g_factory, strategy);
        benchmark::DoNotOptimize(compressor);
    }

    gef::UniformedPartitioner<int64_t, B_GEF_Wrapper<int64_t>, std::shared_ptr<IBitVectorFactory>, gef::SplitPointStrategy> final_compressor(
        input_data, FIXED_PARTITION_SIZE, g_factory, strategy);
    state.counters["size_in_bytes"] = final_compressor.size_in_bytes();
    state.counters["bpi"] = static_cast<double>(final_compressor.size_in_bytes() * 8) / input_data.size();

    double bytes_processed = static_cast<double>(state.iterations() * input_data.size() * sizeof(int64_t));
    state.counters["compression_throughput_MBs"] = benchmark::Counter(bytes_processed, benchmark::Counter::kIsRate, benchmark::Counter::kIs1024);
}

BENCHMARK_DEFINE_F(FileBasedCompressionBenchmark, B_GEF_NO_RLE_Compression)(benchmark::State& state) {
    gef::SplitPointStrategy strategy = static_cast<gef::SplitPointStrategy>(state.range(1));
    state.SetLabel(current_basename + "/" + strategyToString(strategy));

    for (auto _ : state) {
        state.PauseTiming();
        std::vector<int64_t> data_copy = input_data;
        state.ResumeTiming();
        gef::UniformedPartitioner<int64_t, B_GEF_NO_RLE_Wrapper<int64_t>, std::shared_ptr<IBitVectorFactory>, gef::SplitPointStrategy> compressor(
            data_copy, FIXED_PARTITION_SIZE, g_factory, strategy);
        benchmark::DoNotOptimize(compressor);
    }

    gef::UniformedPartitioner<int64_t, B_GEF_NO_RLE_Wrapper<int64_t>, std::shared_ptr<IBitVectorFactory>, gef::SplitPointStrategy> final_compressor(
        input_data, FIXED_PARTITION_SIZE, g_factory, strategy);
    state.counters["size_in_bytes"] = final_compressor.size_in_bytes();
    state.counters["bpi"] = static_cast<double>(final_compressor.size_in_bytes() * 8) / input_data.size();


    double bytes_processed = static_cast<double>(state.iterations() * input_data.size() * sizeof(int64_t));
    state.counters["compression_throughput_MBs"] = benchmark::Counter(bytes_processed, benchmark::Counter::kIsRate, benchmark::Counter::kIs1024);
}

BENCHMARK_DEFINE_F(FileBasedCompressionBenchmark, U_GEF_Compression)(benchmark::State& state) {
    state.SetLabel(current_basename);
    for (auto _ : state) {
        state.PauseTiming();
        std::vector<int64_t> data_copy = input_data;
        state.ResumeTiming();
        gef::UniformedPartitioner<int64_t, U_GEF_Wrapper<int64_t>, std::shared_ptr<IBitVectorFactory>> compressor(
            data_copy, FIXED_PARTITION_SIZE, g_factory);
        benchmark::DoNotOptimize(compressor);
    }
    gef::UniformedPartitioner<int64_t, U_GEF_Wrapper<int64_t>, std::shared_ptr<IBitVectorFactory>> final_compressor(
        input_data, FIXED_PARTITION_SIZE, g_factory);
    state.counters["size_in_bytes"] = final_compressor.size_in_bytes();
    state.counters["bpi"] = static_cast<double>(final_compressor.size_in_bytes() * 8) / input_data.size();


    double bytes_processed = static_cast<double>(state.iterations() * input_data.size() * sizeof(int64_t));
    state.counters["compression_throughput_MBs"] = benchmark::Counter(bytes_processed, benchmark::Counter::kIsRate, benchmark::Counter::kIs1024);
}

BENCHMARK_DEFINE_F(FileBasedCompressionBenchmark, RLE_GEF_Compression)(benchmark::State& state) {
    state.SetLabel(current_basename);
    for (auto _ : state) {
        state.PauseTiming();
        std::vector<int64_t> data_copy = input_data;
        state.ResumeTiming();
        gef::UniformedPartitioner<int64_t, RLE_GEF_Wrapper<int64_t>, std::shared_ptr<IBitVectorFactory>> compressor(
            data_copy, FIXED_PARTITION_SIZE, g_factory);
        benchmark::DoNotOptimize(compressor);
    }
    gef::UniformedPartitioner<int64_t, RLE_GEF_Wrapper<int64_t>, std::shared_ptr<IBitVectorFactory>> final_compressor(
        input_data, FIXED_PARTITION_SIZE, g_factory);
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


#pragma region Lookup Definitions and Registration
BENCHMARK_DEFINE_F(FileBasedCompressionBenchmark, B_GEF_Lookup)(benchmark::State& state) {
    gef::SplitPointStrategy strategy = static_cast<gef::SplitPointStrategy>(state.range(1));
    state.SetLabel(current_basename + "/" + strategyToString(strategy));
    LookupBenchmark<B_GEF_Wrapper>(state, input_data, FIXED_PARTITION_SIZE, g_factory, strategy);
}

BENCHMARK_DEFINE_F(FileBasedCompressionBenchmark, B_GEF_NO_RLE_Lookup)(benchmark::State& state) {
    gef::SplitPointStrategy strategy = static_cast<gef::SplitPointStrategy>(state.range(1));
    state.SetLabel(current_basename + "/" + strategyToString(strategy));
    LookupBenchmark<B_GEF_NO_RLE_Wrapper>(state, input_data, FIXED_PARTITION_SIZE, g_factory, strategy);
}

BENCHMARK_DEFINE_F(FileBasedCompressionBenchmark, U_GEF_Lookup)(benchmark::State& state) {
    state.SetLabel(current_basename);
    LookupBenchmark<U_GEF_Wrapper>(state, input_data, FIXED_PARTITION_SIZE, g_factory);
}

BENCHMARK_DEFINE_F(FileBasedCompressionBenchmark, RLE_GEF_Lookup)(benchmark::State& state) {
    state.SetLabel(current_basename);
    LookupBenchmark<RLE_GEF_Wrapper>(state, input_data, FIXED_PARTITION_SIZE, g_factory);
}

void RegisterBenchmarksForFile(size_t file_idx) {
    // Compression
    BENCHMARK_REGISTER_F(FileBasedCompressionBenchmark, B_GEF_Compression)
        ->Args({(long)file_idx, (long)gef::SplitPointStrategy::APPROXIMATE_SPLIT_POINT})
        ->Args({(long)file_idx, (long)gef::SplitPointStrategy::BRUTE_FORCE_SPLIT_POINT})
        ->Args({(long)file_idx, (long)gef::SplitPointStrategy::BINARY_SEARCH_SPLIT_POINT})
        ->ArgNames({"file_idx", "strategy"});
    BENCHMARK_REGISTER_F(FileBasedCompressionBenchmark, B_GEF_NO_RLE_Compression)
        ->Args({(long)file_idx, (long)gef::SplitPointStrategy::APPROXIMATE_SPLIT_POINT})
        ->Args({(long)file_idx, (long)gef::SplitPointStrategy::BRUTE_FORCE_SPLIT_POINT})
        ->Args({(long)file_idx, (long)gef::SplitPointStrategy::BINARY_SEARCH_SPLIT_POINT})
        ->ArgNames({"file_idx", "strategy"});
    BENCHMARK_REGISTER_F(FileBasedCompressionBenchmark, U_GEF_Compression)
        ->Arg(file_idx)->ArgNames({"file_idx"});
    BENCHMARK_REGISTER_F(FileBasedCompressionBenchmark, RLE_GEF_Compression)
        ->Arg(file_idx)->ArgNames({"file_idx"});

    // Lookup
    BENCHMARK_REGISTER_F(FileBasedCompressionBenchmark, B_GEF_Lookup)
        ->Args({(long)file_idx, (long)gef::SplitPointStrategy::APPROXIMATE_SPLIT_POINT})
        ->Args({(long)file_idx, (long)gef::SplitPointStrategy::BRUTE_FORCE_SPLIT_POINT})
        ->Args({(long)file_idx, (long)gef::SplitPointStrategy::BINARY_SEARCH_SPLIT_POINT})
        ->ArgNames({"file_idx", "strategy"});
    BENCHMARK_REGISTER_F(FileBasedCompressionBenchmark, B_GEF_NO_RLE_Lookup)
        ->Args({(long)file_idx, (long)gef::SplitPointStrategy::APPROXIMATE_SPLIT_POINT})
        ->Args({(long)file_idx, (long)gef::SplitPointStrategy::BRUTE_FORCE_SPLIT_POINT})
        ->Args({(long)file_idx, (long)gef::SplitPointStrategy::BINARY_SEARCH_SPLIT_POINT})
        ->ArgNames({"file_idx", "strategy"});
    BENCHMARK_REGISTER_F(FileBasedCompressionBenchmark, U_GEF_Lookup)
        ->Arg(file_idx)->ArgNames({"file_idx"});
    BENCHMARK_REGISTER_F(FileBasedCompressionBenchmark, RLE_GEF_Lookup)
        ->Arg(file_idx)->ArgNames({"file_idx"});
}
#pragma endregion


#pragma region Main
int main(int argc, char** argv) {
    benchmark::Initialize(&argc, argv);

    for (int i = 1; i < argc; ++i) {
        g_input_files.push_back(argv[i]);
    }

    if (g_input_files.empty()) {
        std::cerr << "Usage: " << argv[0] << " <input_file1> [input_file2 ...] [benchmark_flags]" << std::endl;
        std::cerr << "Example: " << argv[0] << " data/file1.bin data/file2.bin --benchmark_format=json" << std::endl;
        return 1;
    }

    for (size_t i = 0; i < g_input_files.size(); ++i) {
        RegisterBenchmarksForFile(i);
    }

    benchmark::RunSpecifiedBenchmarks();

    return 0;
}
#pragma endregion