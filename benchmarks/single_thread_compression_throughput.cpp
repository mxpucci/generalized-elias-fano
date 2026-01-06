#include "gef/B_STAR_GEF.hpp"
#include "gef/UniformPartitioning.hpp"
#include "gef/utils.hpp"

#include <algorithm>
#include <chrono>
#include <cctype>
#include <filesystem>
#include <iomanip>
#include <iostream>
#include <numeric>
#include <optional>
#include <stdexcept>
#include <string>
#include <string_view>
#include <vector>

using namespace std::chrono;

namespace {

constexpr size_t K = 32000;

struct ProgramOptions {
    std::filesystem::path dataset_path;
    size_t iterations = 5;
    gef::SplitPointStrategy strategy = gef::SplitPointStrategy::OPTIMAL_SPLIT_POINT;
    bool verbose = false;
};

std::optional<ProgramOptions> parse_arguments(int argc, char** argv) {
    if (argc < 2) return std::nullopt;

    ProgramOptions opts{};
    std::vector<std::string> positional;

    for (int i = 1; i < argc; ++i) {
        std::string_view arg(argv[i]);
        if (arg == "--help" || arg == "-h") return std::nullopt;
        if (arg == "--verbose") { opts.verbose = true; continue; }
        if (arg.rfind("--iterations=", 0) == 0) {
            opts.iterations = static_cast<size_t>(std::stoull(std::string(arg.substr(13))));
            continue;
        }
        if (arg.rfind("--strategy=", 0) == 0) {
            std::string value(arg.substr(11));
            std::transform(value.begin(), value.end(), value.begin(),
                           [](unsigned char c) { return static_cast<char>(std::tolower(c)); });
            if (value == "approx" || value == "approximate") {
                opts.strategy = gef::SplitPointStrategy::APPROXIMATE_SPLIT_POINT;
            } else if (value == "optimal" || value == "bruteforce" || value == "brute") {
                opts.strategy = gef::SplitPointStrategy::OPTIMAL_SPLIT_POINT;
            } else {
                throw std::invalid_argument("Unknown strategy: " + value);
            }
            continue;
        }
        if (!arg.empty() && arg[0] == '-') {
            throw std::invalid_argument("Unknown option: " + std::string(arg));
        }
        positional.emplace_back(arg);
    }

    if (positional.empty()) throw std::invalid_argument("Missing dataset path argument");
    opts.dataset_path = std::filesystem::path(positional.front());
    if (!std::filesystem::exists(opts.dataset_path)) {
        throw std::invalid_argument("Dataset does not exist: " + opts.dataset_path.string());
    }
    if (opts.iterations == 0) throw std::invalid_argument("Iterations must be greater than zero");
    return opts;
}

std::string to_string(gef::SplitPointStrategy strategy) {
    switch (strategy) {
        case gef::SplitPointStrategy::OPTIMAL_SPLIT_POINT: return "OPTIMAL";
        case gef::SplitPointStrategy::APPROXIMATE_SPLIT_POINT: return "APPROXIMATE";
        default: return "UNKNOWN";
    }
}

} // namespace

int main(int argc, char** argv) {
    try {
        auto maybe_opts = parse_arguments(argc, argv);
        if (!maybe_opts.has_value()) {
            std::cout << "Usage: " << argv[0] << " [options] <dataset.bin>\n\n"
                      << "Options:\n"
                      << "  --iterations=<N>\n"
                      << "  --strategy=<approx|optimal>\n"
                      << "  --verbose\n";
            return 1;
        }

        const ProgramOptions& opts = *maybe_opts;
        auto data = read_data_binary<int64_t, int64_t>(opts.dataset_path.string(), true);
        if (data.empty()) throw std::runtime_error("Dataset is empty");

        std::vector<double> times;
        times.reserve(opts.iterations);

        for (size_t i = 0; i < opts.iterations; ++i) {
            std::vector<int64_t> data_copy = data;
            auto t0 = steady_clock::now();
            gef::UniformPartitioning<int64_t, gef::internal::B_STAR_GEF<int64_t>, K, gef::SplitPointStrategy> compressor(
                data_copy, opts.strategy);
            auto t1 = steady_clock::now();
            const double seconds = duration<double>(t1 - t0).count();
            times.push_back(seconds);
            if (opts.verbose) {
                std::cerr << "Iteration " << (i + 1) << ": " << seconds << " s\n";
            }
        }

        std::sort(times.begin(), times.end());
        const double avg = std::accumulate(times.begin(), times.end(), 0.0) / static_cast<double>(times.size());
        const double bytes_processed = static_cast<double>(data.size()) * sizeof(int64_t);
        const double throughput_mb_s = (bytes_processed / avg) / (1024.0 * 1024.0);

        std::cout << "Dataset:            " << opts.dataset_path << "\n"
                  << "K:                  " << K << "\n"
                  << "Iterations:         " << opts.iterations << "\n"
                  << "Strategy:           " << to_string(opts.strategy) << "\n"
                  << "Num integers:       " << data.size() << "\n"
                  << std::fixed << std::setprecision(3)
                  << "Avg time:           " << avg << " s\n"
                  << "Throughput:         " << throughput_mb_s << " MB/s\n";
        return 0;
    } catch (const std::exception& ex) {
        std::cerr << "Error: " << ex.what() << "\n";
        return 1;
    }
}

