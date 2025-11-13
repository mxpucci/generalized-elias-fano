#include "gef/B_GEF_STAR.hpp"
#include "gef/UniformedPartitioner.hpp"
#include "gef/CompressionProfile.hpp"
#include "gef/utils.hpp"
#include "datastructures/IBitVectorFactory.hpp"
#include "datastructures/SDSLBitVectorFactory.hpp"
#include "datastructures/SUXBitVectorFactory.hpp"

#include <algorithm>
#include <chrono>
#include <cctype>
#include <cmath>
#include <cstdint>
#include <exception>
#include <filesystem>
#include <iomanip>
#include <iostream>
#include <memory>
#include <numeric>
#include <optional>
#include <stdexcept>
#include <string>
#include <string_view>
#include <vector>

using namespace std::chrono;

namespace {

struct ProgramOptions {
    std::filesystem::path dataset_path;
    size_t partition_size = 0; // 0 means auto (entire dataset)
    size_t iterations = 5;
    gef::SplitPointStrategy strategy = gef::SplitPointStrategy::OPTIMAL_SPLIT_POINT;
    std::string bitvector = "sdsl";
    bool verbose = false;
};

template<typename T>
struct BGEFStarWrapper : public gef::B_GEF_STAR<T> {
    BGEFStarWrapper(gef::Span<const T> data,
                    const std::shared_ptr<IBitVectorFactory>& factory,
                    gef::SplitPointStrategy strategy,
                    gef::CompressionBuildMetrics* metrics = nullptr)
        : BGEFStarWrapper(std::vector<T>(data.data(), data.data() + data.size()), factory, strategy, metrics) {}

    BGEFStarWrapper(const std::vector<T>& data,
                    const std::shared_ptr<IBitVectorFactory>& factory,
                    gef::SplitPointStrategy strategy,
                    gef::CompressionBuildMetrics* metrics = nullptr)
        : gef::B_GEF_STAR<T>(factory, data, strategy, metrics) {}

    BGEFStarWrapper() : gef::B_GEF_STAR<T>() {}
};

std::optional<ProgramOptions> parse_arguments(int argc, char** argv) {
    if (argc < 2) {
        return std::nullopt;
    }

    ProgramOptions opts{};
    std::vector<std::string> positional;

    for (int i = 1; i < argc; ++i) {
        std::string_view arg(argv[i]);
        if (arg == "--help" || arg == "-h") {
            return std::nullopt;
        } else if (arg == "--verbose") {
            opts.verbose = true;
        } else if (arg.rfind("--partition-size=", 0) == 0) {
            opts.partition_size = static_cast<size_t>(std::stoull(std::string(arg.substr(17))));
        } else if (arg.rfind("--iterations=", 0) == 0) {
            opts.iterations = static_cast<size_t>(std::stoull(std::string(arg.substr(13))));
        } else if (arg.rfind("--strategy=", 0) == 0) {
            std::string value(arg.substr(11));
            std::transform(value.begin(),
                           value.end(),
                           value.begin(),
                           [](unsigned char c) { return static_cast<char>(std::tolower(c)); });
            if (value == "approx" || value == "approximate") {
                opts.strategy = gef::SplitPointStrategy::APPROXIMATE_SPLIT_POINT;
            } else if (value == "optimal" || value == "bruteforce" || value == "brute") {
                opts.strategy = gef::SplitPointStrategy::OPTIMAL_SPLIT_POINT;
            } else {
                throw std::invalid_argument("Unknown strategy: " + value);
            }
        } else if (arg.rfind("--bitvector=", 0) == 0) {
            std::string value(arg.substr(12));
            std::transform(value.begin(),
                           value.end(),
                           value.begin(),
                           [](unsigned char c) { return static_cast<char>(std::tolower(c)); });
            if (value != "sdsl" && value != "sux") {
                throw std::invalid_argument("Unsupported bitvector implementation: " + value);
            }
            opts.bitvector = std::move(value);
        } else if (!arg.empty() && arg[0] == '-') {
            throw std::invalid_argument("Unknown option: " + std::string(arg));
        } else {
            positional.emplace_back(arg);
        }
    }

    if (positional.empty()) {
        throw std::invalid_argument("Missing dataset path argument");
    }

    opts.dataset_path = std::filesystem::path(positional.front());
    if (!std::filesystem::exists(opts.dataset_path)) {
        throw std::invalid_argument("Dataset does not exist: " + opts.dataset_path.string());
    }

    if (opts.iterations == 0) {
        throw std::invalid_argument("Iterations must be greater than zero");
    }

    return opts;
}

std::shared_ptr<IBitVectorFactory> make_factory(const std::string& name) {
    if (name == "sdsl") {
        return std::make_shared<SDSLBitVectorFactory>();
    }
    if (name == "sux") {
        return std::make_shared<SUXBitVectorFactory>();
    }
    throw std::invalid_argument("Unsupported bitvector factory: " + name);
}

std::string to_string(gef::SplitPointStrategy strategy) {
    switch (strategy) {
        case gef::SplitPointStrategy::OPTIMAL_SPLIT_POINT:
            return "OPTIMAL";
        case gef::SplitPointStrategy::APPROXIMATE_SPLIT_POINT:
            return "APPROXIMATE";
        default:
            return "UNKNOWN";
    }
}

struct MeasurementResult {
    double average_seconds = 0.0;
    double min_seconds = 0.0;
    double max_seconds = 0.0;
    double throughput_mb_s = 0.0;
    double size_in_bytes = 0.0;
    double bits_per_int = 0.0;
    gef::CompressionBuildMetrics build_metrics{};
};

MeasurementResult measure(const std::vector<int64_t>& data,
                          size_t iterations,
                          size_t partition_size,
                          const std::shared_ptr<IBitVectorFactory>& factory,
                          gef::SplitPointStrategy strategy,
                          bool verbose) {
    if (data.empty()) {
        throw std::runtime_error("Dataset is empty");
    }

    std::vector<double> elapsed_seconds;
    elapsed_seconds.reserve(iterations);

    for (size_t i = 0; i < iterations; ++i) {
        std::vector<int64_t> data_copy = data;
        auto t0 = steady_clock::now();
        gef::UniformedPartitioner<int64_t,
                                  BGEFStarWrapper<int64_t>,
                                  std::shared_ptr<IBitVectorFactory>,
                                  gef::SplitPointStrategy> compressor(
            data_copy, partition_size, factory, strategy);
        auto t1 = steady_clock::now();
        double seconds = duration<double>(t1 - t0).count();
        elapsed_seconds.push_back(seconds);
        if (verbose) {
            std::cerr << "Iteration " << (i + 1) << "/" << iterations << ": "
                      << seconds << " s" << std::endl;
        }
    }

    std::sort(elapsed_seconds.begin(), elapsed_seconds.end());
    double avg = std::accumulate(elapsed_seconds.begin(), elapsed_seconds.end(), 0.0) / iterations;
    double min = elapsed_seconds.front();
    double max = elapsed_seconds.back();

    const double bytes_processed = static_cast<double>(data.size()) * sizeof(int64_t);
    const double throughput_mb_s = (bytes_processed / avg) / (1024.0 * 1024.0);

    gef::CompressionBuildMetrics build_metrics;
    gef::UniformedPartitioner<int64_t,
                              BGEFStarWrapper<int64_t>,
                              std::shared_ptr<IBitVectorFactory>,
                              gef::SplitPointStrategy,
                              gef::CompressionBuildMetrics*> final_compressor(
        data, partition_size, factory, strategy, &build_metrics);
    const double size_in_bytes = static_cast<double>(final_compressor.size_in_bytes());
    const double bits_per_int = (size_in_bytes * 8.0) / static_cast<double>(data.size());

    return MeasurementResult{
        .average_seconds = avg,
        .min_seconds = min,
        .max_seconds = max,
        .throughput_mb_s = throughput_mb_s,
        .size_in_bytes = size_in_bytes,
        .bits_per_int = bits_per_int,
        .build_metrics = build_metrics
    };
}

void print_options(const ProgramOptions& opts, size_t effective_partition_size) {
    std::cout << "Dataset:             " << opts.dataset_path << "\n"
              << "Partition size:      " << effective_partition_size
              << (opts.partition_size == 0 ? " (auto)" : "") << "\n"
              << "Iterations:          " << opts.iterations << "\n"
              << "Strategy:            " << to_string(opts.strategy) << "\n"
              << "Bitvector factory:   " << opts.bitvector << "\n"
#ifdef GEF_DISABLE_SIMD
              << "SIMD intrinsics:     disabled\n";
#else
              << "SIMD intrinsics:     enabled\n";
#endif
}

} // namespace

int main(int argc, char** argv) {
    try {
        auto maybe_opts = parse_arguments(argc, argv);
        if (!maybe_opts.has_value()) {
            std::cout << "Usage: " << argv[0]
                      << " [options] <dataset.bin>\n\n"
                      << "Options:\n"
                      << "  --partition-size=<N>   Partition size (default: auto - entire dataset)\n"
                      << "  --iterations=<N>       Number of measurement iterations (default: 5)\n"
                      << "  --strategy=<approx|optimal>  Split point strategy (default: optimal)\n"
                      << "  --bitvector=<sdsl|sux> Bitvector implementation (default: sdsl)\n"
                      << "  --verbose              Print per-iteration timings\n"
                      << std::endl;
            return maybe_opts.has_value() ? 0 : 1;
        }

        const ProgramOptions& opts = *maybe_opts;
        auto data = read_data_binary<int64_t, int64_t>(opts.dataset_path.string(), true);

        if (data.empty()) {
            throw std::runtime_error("Dataset is empty");
        }

        size_t effective_partition_size = (opts.partition_size == 0) ? data.size() : opts.partition_size;
        if (effective_partition_size == 0) {
            effective_partition_size = data.size();
        }
        effective_partition_size = std::min(effective_partition_size, data.size());
        effective_partition_size = std::max<size_t>(effective_partition_size, 1);

        print_options(opts, effective_partition_size);
        std::cout << "Number of integers:  " << data.size() << "\n";

        auto factory = make_factory(opts.bitvector);
        auto result = measure(data,
                              opts.iterations,
                              effective_partition_size,
                              factory,
                              opts.strategy,
                              opts.verbose);

        std::cout << std::fixed << std::setprecision(3);
        std::cout << "Average time:        " << result.average_seconds << " s\n"
                  << "Min / Max time:      " << result.min_seconds << " s / "
                  << result.max_seconds << " s\n"
                  << "Throughput:          " << result.throughput_mb_s << " MB/s\n"
                  << "Compressed size:     " << (result.size_in_bytes / (1024.0 * 1024.0)) << " MB\n"
                  << "Bits per integer:    " << result.bits_per_int << "\n";

        const auto& metrics = result.build_metrics;
        std::cout << "\nBuild phase timings ("
                  << metrics.partitions << " partition"
                  << (metrics.partitions == 1 ? "" : "s") << "):\n";
        if (metrics.partitions == 0) {
            std::cout << "  No partitions were constructed.\n";
        } else {
            const double split_avg = metrics.split_point_seconds / static_cast<double>(metrics.partitions);
            const double allocation_avg = metrics.allocation_seconds / static_cast<double>(metrics.partitions);
            const double population_avg = metrics.population_seconds / static_cast<double>(metrics.partitions);

            std::cout << "  Split-point estimation: "
                      << metrics.split_point_seconds << " s total ("
                      << split_avg << " s / partition)\n";
            std::cout << "  Structure allocation:   "
                      << metrics.allocation_seconds << " s total ("
                      << allocation_avg << " s / partition)\n";
            std::cout << "  Structure population:   "
                      << metrics.population_seconds << " s total ("
                      << population_avg << " s / partition)\n";
            std::cout << "  Elements processed:     "
                      << metrics.elements_processed << "\n";
            std::cout << "  Exceptions encountered: "
                      << metrics.total_exceptions << "\n";
            std::cout << "  Split point (avg/min/max): "
                      << metrics.average_split_point() << " / "
                      << static_cast<int>(metrics.min_split_point) << " / "
                      << static_cast<int>(metrics.max_split_point) << "\n";
        }

        std::cout << std::endl;

        return 0;
    } catch (const std::exception& ex) {
        std::cerr << "Error: " << ex.what() << std::endl;
        return 1;
    }
}

