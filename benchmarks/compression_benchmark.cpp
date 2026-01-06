// Minimal benchmark translation unit (factory-free, uses gef::internal).
#include <benchmark/benchmark.h>

#include "gef/B_GEF.hpp"
#include "gef/B_STAR_GEF.hpp"
#include "gef/U_GEF.hpp"
#include "gef/RLE_GEF.hpp"
#include "gef/UniformPartitioning.hpp"
#include "gef/utils.hpp"

#include <algorithm>
#include <filesystem>
#include <string>
#include <vector>

namespace {

constexpr size_t K = 32000;

std::vector<std::string> g_input_files;

std::string strategyToString(gef::SplitPointStrategy strategy) {
    switch (strategy) {
        case gef::SplitPointStrategy::APPROXIMATE_SPLIT_POINT: return "APPROXIMATE";
        case gef::SplitPointStrategy::OPTIMAL_SPLIT_POINT: return "OPTIMAL";
        default: return "UNKNOWN";
    }
}

class UniformPartitioningBenchmark : public benchmark::Fixture {
public:
    std::vector<int64_t> input_data;
    std::string current_basename;

    void SetUp(::benchmark::State& state) override {
        const size_t file_idx = static_cast<size_t>(state.range(0));
        if (file_idx >= g_input_files.size()) {
            state.SkipWithError("File index out of bounds.");
            return;
        }
        const auto& path = g_input_files[file_idx];
        current_basename = std::filesystem::path(path).filename().string();
        input_data = read_data_binary<int64_t, int64_t>(path, true);
        if (input_data.empty()) {
            state.SkipWithError("Input data is empty.");
        }
        state.counters["num_integers"] = input_data.size();
    }
};

} // namespace

BENCHMARK_DEFINE_F(UniformPartitioningBenchmark, B_GEF_Compression)(benchmark::State& state) {
    const auto strategy = static_cast<gef::SplitPointStrategy>(state.range(1));
    state.SetLabel(current_basename + "/" + strategyToString(strategy) + "/K=" + std::to_string(K));

    for (auto _ : state) {
        state.PauseTiming();
        std::vector<int64_t> data_copy = input_data;
        state.ResumeTiming();
        gef::UniformPartitioning<int64_t, gef::internal::B_GEF<int64_t>, K, gef::SplitPointStrategy> compressor(
            data_copy, strategy);
        benchmark::DoNotOptimize(compressor.size_in_bytes());
    }
}

BENCHMARK_DEFINE_F(UniformPartitioningBenchmark, B_STAR_GEF_Compression)(benchmark::State& state) {
    const auto strategy = static_cast<gef::SplitPointStrategy>(state.range(1));
    state.SetLabel(current_basename + "/" + strategyToString(strategy) + "/K=" + std::to_string(K));

    for (auto _ : state) {
        state.PauseTiming();
        std::vector<int64_t> data_copy = input_data;
        state.ResumeTiming();
        gef::UniformPartitioning<int64_t, gef::internal::B_STAR_GEF<int64_t>, K, gef::SplitPointStrategy> compressor(
            data_copy, strategy);
        benchmark::DoNotOptimize(compressor.size_in_bytes());
    }
}

BENCHMARK_DEFINE_F(UniformPartitioningBenchmark, U_GEF_Compression)(benchmark::State& state) {
    const auto strategy = static_cast<gef::SplitPointStrategy>(state.range(1));
    state.SetLabel(current_basename + "/" + strategyToString(strategy) + "/K=" + std::to_string(K));

    for (auto _ : state) {
        state.PauseTiming();
        std::vector<int64_t> data_copy = input_data;
        state.ResumeTiming();
        gef::UniformPartitioning<int64_t, gef::internal::U_GEF<int64_t>, K, gef::SplitPointStrategy> compressor(
            data_copy, strategy);
        benchmark::DoNotOptimize(compressor.size_in_bytes());
    }
}

BENCHMARK_DEFINE_F(UniformPartitioningBenchmark, RLE_GEF_Compression)(benchmark::State& state) {
    state.SetLabel(current_basename + "/K=" + std::to_string(K));

    for (auto _ : state) {
        state.PauseTiming();
        std::vector<int64_t> data_copy = input_data;
        state.ResumeTiming();
        gef::UniformPartitioning<int64_t, gef::internal::RLE_GEF<int64_t>, K> compressor(data_copy);
        benchmark::DoNotOptimize(compressor.size_in_bytes());
    }
}

static void RegisterBenchmarksForFile(size_t file_idx) {
    const std::vector<int64_t> file_idx_args = {static_cast<int64_t>(file_idx)};
    const std::vector<int64_t> strategy_args = {
        static_cast<int64_t>(gef::SplitPointStrategy::APPROXIMATE_SPLIT_POINT),
        static_cast<int64_t>(gef::SplitPointStrategy::OPTIMAL_SPLIT_POINT)
    };

    BENCHMARK_REGISTER_F(UniformPartitioningBenchmark, B_GEF_Compression)->ArgsProduct({file_idx_args, strategy_args});
    BENCHMARK_REGISTER_F(UniformPartitioningBenchmark, B_STAR_GEF_Compression)->ArgsProduct({file_idx_args, strategy_args});
    BENCHMARK_REGISTER_F(UniformPartitioningBenchmark, U_GEF_Compression)->ArgsProduct({file_idx_args, strategy_args});
    BENCHMARK_REGISTER_F(UniformPartitioningBenchmark, RLE_GEF_Compression)->ArgsProduct({file_idx_args});
}

int main(int argc, char** argv) {
    benchmark::Initialize(&argc, argv);

    for (int i = 1; i < argc; ++i) {
        const std::string arg = argv[i];
        if (!arg.empty() && arg.rfind("--", 0) == 0) continue;
        g_input_files.push_back(arg);
    }

    if (g_input_files.empty()) {
        std::cerr << "Usage: " << argv[0] << " <input_file1> [input_file2 ...] [benchmark_flags]\n";
        return 1;
    }

    for (size_t i = 0; i < g_input_files.size(); ++i) {
        RegisterBenchmarksForFile(i);
    }

    benchmark::RunSpecifiedBenchmarks();
    return 0;
}
