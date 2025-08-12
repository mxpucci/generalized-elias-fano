#include <benchmark/benchmark.h>
#include "gef/B_GEF.hpp"
#include "gef/U_GEF.hpp"
#include "gef/RLE_GEF.hpp"
#include "gef/B_GEF_NO_RLE.hpp"
#include "gef/UniformedPartitioner.hpp"
#include "datastructures/SDSLBitVectorFactory.hpp"
#include "gef/utils.hpp"
#include <vector>
#include <string>
#include <iostream>
#include <filesystem>

#pragma region Compressor Wrappers and Globals
template<typename T>
struct U_GEF_Wrapper : public gef::U_GEF<T> {
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
struct B_GEF_NO_RLE_Wrapper : public gef::B_GEF_NO_RLE<T> {
    B_GEF_NO_RLE_Wrapper(const std::vector<T>& data, std::shared_ptr<IBitVectorFactory> factory, gef::SplitPointStrategy strategy)
        : gef::B_GEF_NO_RLE<T>(factory, data, strategy) {}
    B_GEF_NO_RLE_Wrapper() : gef::B_GEF_NO_RLE<T>() {}
};

std::vector<std::string> g_input_files;
std::shared_ptr<IBitVectorFactory> g_factory = std::make_shared<SDSLBitVectorFactory>();
size_t g_block_size = 0;
#pragma endregion

class BlockSizeBenchmark : public benchmark::Fixture {
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
    }

    void TearDown(::benchmark::State& state) override {}
};

BENCHMARK_DEFINE_F(BlockSizeBenchmark, B_GEF_BlockSize)(benchmark::State& state) {
    size_t block_size = state.range(1);
    state.SetLabel(current_basename + "/B_GEF/block_size:" + std::to_string(block_size));

    gef::UniformedPartitioner<int64_t, B_GEF_Wrapper<int64_t>, std::shared_ptr<IBitVectorFactory>, gef::SplitPointStrategy> compressor(
        input_data, block_size, g_factory, gef::SplitPointStrategy::BRUTE_FORCE_SPLIT_POINT);
    state.counters["size_in_bytes"] = compressor.size_in_bytes();

    for (auto _ : state) {
        benchmark::DoNotOptimize(compressor);
    }
}

BENCHMARK_DEFINE_F(BlockSizeBenchmark, B_GEF_NO_RLE_BlockSize)(benchmark::State& state) {
    size_t block_size = state.range(1);
    state.SetLabel(current_basename + "/B_GEF_NO_RLE/block_size:" + std::to_string(block_size));

    gef::UniformedPartitioner<int64_t, B_GEF_NO_RLE_Wrapper<int64_t>, std::shared_ptr<IBitVectorFactory>, gef::SplitPointStrategy> compressor(
        input_data, block_size, g_factory, gef::SplitPointStrategy::BRUTE_FORCE_SPLIT_POINT);
    state.counters["size_in_bytes"] = compressor.size_in_bytes();

    for (auto _ : state) {
        benchmark::DoNotOptimize(compressor);
    }
}

BENCHMARK_DEFINE_F(BlockSizeBenchmark, U_GEF_BlockSize)(benchmark::State& state) {
    size_t block_size = state.range(1);
    state.SetLabel(current_basename + "/U_GEF/block_size:" + std::to_string(block_size));

    gef::UniformedPartitioner<int64_t, U_GEF_Wrapper<int64_t>, std::shared_ptr<IBitVectorFactory>, gef::SplitPointStrategy> compressor(
        input_data, block_size, g_factory, gef::SplitPointStrategy::BRUTE_FORCE_SPLIT_POINT);
    state.counters["size_in_bytes"] = compressor.size_in_bytes();

    for (auto _ : state) {
        benchmark::DoNotOptimize(compressor);
    }
}

BENCHMARK_DEFINE_F(BlockSizeBenchmark, RLE_GEF_BlockSize)(benchmark::State& state) {
    size_t block_size = state.range(1);
    state.SetLabel(current_basename + "/RLE_GEF/block_size:" + std::to_string(block_size));

    gef::UniformedPartitioner<int64_t, RLE_GEF_Wrapper<int64_t>, std::shared_ptr<IBitVectorFactory>> compressor(
        input_data, block_size, g_factory);
    state.counters["size_in_bytes"] = compressor.size_in_bytes();

    for (auto _ : state) {
        benchmark::DoNotOptimize(compressor);
    }
}

void RegisterBlockSizeBenchmarks(size_t file_idx) {
    if (g_block_size > 0) {
        BENCHMARK_REGISTER_F(BlockSizeBenchmark, B_GEF_BlockSize)->Args({(long)file_idx, (long)g_block_size});
        BENCHMARK_REGISTER_F(BlockSizeBenchmark, B_GEF_NO_RLE_BlockSize)->Args({(long)file_idx, (long)g_block_size});
        BENCHMARK_REGISTER_F(BlockSizeBenchmark, U_GEF_BlockSize)->Args({(long)file_idx, (long)g_block_size});
        BENCHMARK_REGISTER_F(BlockSizeBenchmark, RLE_GEF_BlockSize)->Args({(long)file_idx, (long)g_block_size});
        return;
    }

    const long start_block_size = 2048;
    const long end_block_size = 512 * 1024;
    const long step = 2048;

    for (long block_size = start_block_size; block_size <= end_block_size; block_size += step) {
        BENCHMARK_REGISTER_F(BlockSizeBenchmark, B_GEF_BlockSize)->Args({(long)file_idx, block_size});
        BENCHMARK_REGISTER_F(BlockSizeBenchmark, B_GEF_NO_RLE_BlockSize)->Args({(long)file_idx, block_size});
        BENCHMARK_REGISTER_F(BlockSizeBenchmark, U_GEF_BlockSize)->Args({(long)file_idx, block_size});
        BENCHMARK_REGISTER_F(BlockSizeBenchmark, RLE_GEF_BlockSize)->Args({(long)file_idx, block_size});
    }
}

int main(int argc, char** argv) {
    std::vector<char*> new_argv;
    new_argv.push_back(argv[0]);
    for (int i = 1; i < argc; ++i) {
        if (std::string(argv[i]) == "--block_size" && i + 1 < argc) {
            g_block_size = std::stoul(argv[i + 1]);
            i++; // skip next argument
        } else {
            new_argv.push_back(argv[i]);
        }
    }
    int new_argc = new_argv.size();
    char** new_argv_ptr = new_argv.data();

    benchmark::Initialize(&new_argc, new_argv_ptr);

    for (int i = 1; i < new_argc; ++i) {
        g_input_files.push_back(new_argv_ptr[i]);
    }

    if (g_input_files.empty()) {
        std::cerr << "Usage: " << argv[0] << " <input_file1> [input_file2 ...] [--block_size <size>] [benchmark_flags]" << std::endl;
        return 1;
    }

    for (size_t i = 0; i < g_input_files.size(); ++i) {
        RegisterBlockSizeBenchmarks(i);
    }

    benchmark::RunSpecifiedBenchmarks();

    return 0;
}