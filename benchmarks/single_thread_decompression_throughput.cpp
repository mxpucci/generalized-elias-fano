#include "gef/B_STAR_GEF.hpp"
#include "gef/B_GEF.hpp"
#include "gef/U_GEF.hpp"
#include "gef/utils.hpp"
#include <algorithm>
#include <chrono>
#include <cctype>
#include <iostream>
#include <vector>
#include <filesystem>
#include <numeric>
#include <iomanip>
#include <random>
#include <optional>
#include <stdexcept>
#include <string>
#include <string_view>

template <class T>
void do_not_optimize(T const& value) {
    asm volatile("" : : "r,m"(value) : "memory");
}

struct ProgramOptions {
    std::filesystem::path dataset_path;
    size_t iterations = 5;
    gef::SplitPointStrategy strategy = gef::SplitPointStrategy::OPTIMAL_SPLIT_POINT;
    std::string compressor = "bgef_star"; // Default compressor
    bool verbose = false;
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
        } else if (arg.rfind("--iterations=", 0) == 0) {
            opts.iterations = static_cast<size_t>(std::stoull(std::string(arg.substr(13))));
        } else if (arg.rfind("--strategy=", 0) == 0) {
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
        } else if (arg.rfind("--compressor=", 0) == 0) {
            std::string value(arg.substr(13));
            std::transform(value.begin(), value.end(), value.begin(),
                           [](unsigned char c) { return static_cast<char>(std::tolower(c)); });
            if (value != "bgef" && value != "bgef_star" && value != "ugef") {
                throw std::invalid_argument("Unsupported compressor: " + value);
            }
            opts.compressor = std::move(value);
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

template<typename CompressorType>
void run_benchmark(const std::vector<int64_t>& data,
                   gef::SplitPointStrategy strategy,
                   size_t iterations,
                   bool verbose) {
    
    std::cout << "Building compressor..." << std::endl;
    CompressorType compressor(data, strategy);
    
    std::cout << "Compressed size: " << (compressor.size_in_bytes() / (1024.0 * 1024.0)) << " MB" << std::endl;
    std::cout << "Bits per integer: " << (compressor.size_in_bytes() * 8.0) / data.size() << std::endl;

    std::vector<int64_t> decompressed(data.size());
    std::vector<double> timings;
    timings.reserve(iterations);

    for(size_t i=0; i<iterations; ++i) {
        // Clear cache effects if possible? (hard to do portably without root/huge overhead)
        // We just measure hot loop
        
        auto t1 = std::chrono::high_resolution_clock::now();
        compressor.get_elements(0, data.size(), decompressed);
        auto t2 = std::chrono::high_resolution_clock::now();
        
        double ns = std::chrono::duration_cast<std::chrono::nanoseconds>(t2 - t1).count();
        timings.push_back(ns);
        do_not_optimize(decompressed);
        
        if (verbose) {
            std::cout << "Iteration " << (i+1) << ": " << (ns/1e6) << " ms" << std::endl;
        }
    }

    // Verification (run once)
    for (size_t i = 0; i < data.size(); ++i) {
        if (data[i] != decompressed[i]) {
            std::cerr << "Decompression error at " << i << ": expected " << data[i] << ", got " << decompressed[i] << std::endl;
            break;
        }
    }

    double avg_ns = std::accumulate(timings.begin(), timings.end(), 0.0) / iterations;
    double min_ns = *std::min_element(timings.begin(), timings.end());
    double max_ns = *std::max_element(timings.begin(), timings.end());

    double throughput_mb_s = (data.size() * sizeof(int64_t) / 1e6) / (avg_ns / 1e9);

    std::cout << std::fixed << std::setprecision(3);
    std::cout << "Average decompression time: " << (avg_ns / 1e6) << " ms\n";
    std::cout << "Min / Max time:           " << (min_ns / 1e6) << " ms / " << (max_ns / 1e6) << " ms\n";
    std::cout << "Decompression Throughput: " << throughput_mb_s << " MB/s\n";
}

int main(int argc, char** argv) {
    try {
        auto maybe_opts = parse_arguments(argc, argv);
        if (!maybe_opts.has_value()) {
            std::cout << "Usage: " << argv[0]
                      << " [options] <dataset.bin>\n\n"
                      << "Options:\n"
                      << "  --iterations=<N>       Number of measurement iterations (default: 5)\n"
                      << "  --strategy=<approx|optimal>  Split point strategy (default: optimal)\n"
                      << "  --compressor=<bgef|bgef_star|ugef> Compressor implementation (default: bgef_star)\n"
                      << "  --verbose              Print per-iteration timings\n"
                      << std::endl;
            return maybe_opts.has_value() ? 0 : 1;
        }

        const ProgramOptions& opts = *maybe_opts;
        auto data = read_data_binary<int64_t, int64_t>(opts.dataset_path.string(), true);

        if (data.empty()) {
            throw std::runtime_error("Dataset is empty");
        }

        std::cout << "Dataset:             " << opts.dataset_path << "\n"
                  << "Iterations:          " << opts.iterations << "\n"
                  << "Strategy:            " << to_string(opts.strategy) << "\n"
                  << "Compressor:          " << opts.compressor << "\n"
                  << "Number of integers:  " << data.size() << "\n";

        if (opts.compressor == "bgef_star") {
            run_benchmark<gef::internal::B_STAR_GEF<int64_t>>(data, opts.strategy, opts.iterations, opts.verbose);
        } else if (opts.compressor == "ugef") {
            run_benchmark<gef::internal::U_GEF<int64_t>>(data, opts.strategy, opts.iterations, opts.verbose);
        } else { // bgef
            run_benchmark<gef::internal::B_GEF<int64_t>>(data, opts.strategy, opts.iterations, opts.verbose);
        }

        return 0;
    } catch (const std::exception& ex) {
        std::cerr << "Error: " << ex.what() << std::endl;
        return 1;
    }
}

