#include "gef/B_GEF_STAR.hpp"
#include "gef/B_GEF.hpp"
#include "gef/utils.hpp"
#include "datastructures/IBitVectorFactory.hpp"
#include "datastructures/SDSLBitVectorFactory.hpp"
#include "datastructures/SUXBitVectorFactory.hpp"
#include "datastructures/PastaBitVectorFactory.hpp"
#include "datastructures/SDSLBitVector.hpp"
#include "datastructures/SUXBitVector.hpp"
#include "datastructures/PastaBitVector.hpp"
#include <algorithm>
#include <chrono>
#include <cctype>
#include <iostream>
#include <vector>
#include <memory>
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
    size_t num_queries = 1000000;
    size_t iterations = 5;
    gef::SplitPointStrategy strategy = gef::SplitPointStrategy::OPTIMAL_SPLIT_POINT;
    std::string bitvector = "sdsl";
    std::string compressor = "bgef_star";
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
        } else if (arg.rfind("--queries=", 0) == 0) {
            opts.num_queries = static_cast<size_t>(std::stoull(std::string(arg.substr(10))));
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
        } else if (arg.rfind("--bitvector=", 0) == 0) {
            std::string value(arg.substr(12));
            std::transform(value.begin(), value.end(), value.begin(),
                           [](unsigned char c) { return static_cast<char>(std::tolower(c)); });
            if (value != "sdsl" && value != "sux" && value != "pasta") {
                throw std::invalid_argument("Unsupported bitvector implementation: " + value);
            }
            opts.bitvector = std::move(value);
        } else if (arg.rfind("--compressor=", 0) == 0) {
            std::string value(arg.substr(13));
            std::transform(value.begin(), value.end(), value.begin(),
                           [](unsigned char c) { return static_cast<char>(std::tolower(c)); });
            if (value != "bgef" && value != "bgef_star") {
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

std::shared_ptr<IBitVectorFactory> make_factory(const std::string& name) {
    if (name == "sdsl") {
        return std::make_shared<SDSLBitVectorFactory>();
    }
    if (name == "sux") {
        return std::make_shared<SUXBitVectorFactory>();
    }
    if (name == "pasta") {
        return std::make_shared<PastaBitVectorFactory>();
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

std::vector<size_t> generate_random_indices(size_t n, size_t num_queries, uint32_t seed = 1234) {
    std::mt19937 mt(seed);
    std::uniform_int_distribution<size_t> dist(0, n - 1);
    std::vector<size_t> indices(num_queries);
    for (auto &idx : indices) {
        idx = dist(mt);
    }
    return indices;
}

template<typename CompressorType>
void run_benchmark(const std::vector<int64_t>& data,
                   const std::shared_ptr<IBitVectorFactory>& factory,
                   gef::SplitPointStrategy strategy,
                   size_t num_queries,
                   size_t iterations,
                   bool verbose) {
    
    std::cout << "Building compressor..." << std::endl;
    CompressorType compressor(factory, data, strategy);
    
    std::cout << "Compressed size: " << (compressor.size_in_bytes() / (1024.0 * 1024.0)) << " MB" << std::endl;
    std::cout << "Bits per integer: " << (compressor.size_in_bytes() * 8.0) / data.size() << std::endl;

    // Check max queries
    if (num_queries > data.size()) {
         // Allow more queries than data size (just random sampling)
    }

    std::vector<double> timings;
    timings.reserve(iterations);
    
    auto indices = generate_random_indices(data.size(), num_queries);

    for(size_t i=0; i<iterations; ++i) {
        auto t1 = std::chrono::high_resolution_clock::now();
        int64_t sum = 0;
        for (auto idx : indices) {
            sum += compressor[idx];
        }
        auto t2 = std::chrono::high_resolution_clock::now();
        
        double ns = std::chrono::duration_cast<std::chrono::nanoseconds>(t2 - t1).count();
        timings.push_back(ns);
        do_not_optimize(sum);
        
        if (verbose) {
            std::cout << "Iteration " << (i+1) << ": " << (ns/1e6) << " ms" << std::endl;
        }
    }

    // Random verification (check first 100 random indices or so to be sure)
    for (size_t k = 0; k < std::min<size_t>(100, num_queries); ++k) {
        size_t idx = indices[k];
        if (data[idx] != compressor[idx]) {
             std::cerr << "Random access error at " << idx << ": expected " << data[idx] << ", got " << compressor[idx] << std::endl;
             exit(1);
        }
    }

    double avg_ns = std::accumulate(timings.begin(), timings.end(), 0.0) / iterations;
    double min_ns = *std::min_element(timings.begin(), timings.end());
    double max_ns = *std::max_element(timings.begin(), timings.end());

    double avg_latency_ns = avg_ns / num_queries;

    std::cout << std::fixed << std::setprecision(3);
    std::cout << "Average total time:     " << (avg_ns / 1e6) << " ms\n";
    std::cout << "Min / Max total time:   " << (min_ns / 1e6) << " ms / " << (max_ns / 1e6) << " ms\n";
    std::cout << "Average Latency:        " << avg_latency_ns << " ns/query\n";
}

int main(int argc, char** argv) {
    try {
        auto maybe_opts = parse_arguments(argc, argv);
        if (!maybe_opts.has_value()) {
            std::cout << "Usage: " << argv[0]
                      << " [options] <dataset.bin>\n\n"
                      << "Options:\n"
                      << "  --iterations=<N>       Number of measurement iterations (default: 5)\n"
                      << "  --queries=<N>          Number of random queries (default: 1000000)\n"
                      << "  --strategy=<approx|optimal>  Split point strategy (default: optimal)\n"
                      << "  --bitvector=<sdsl|sux|pasta> Bitvector implementation (default: sdsl)\n"
                      << "  --compressor=<bgef|bgef_star> Compressor implementation (default: bgef_star)\n"
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
                  << "Queries:             " << opts.num_queries << "\n"
                  << "Strategy:            " << to_string(opts.strategy) << "\n"
                  << "Bitvector factory:   " << opts.bitvector << "\n"
                  << "Compressor:          " << opts.compressor << "\n"
                  << "Number of integers:  " << data.size() << "\n";

        auto factory = make_factory(opts.bitvector);

        if (opts.compressor == "bgef_star") {
            if (opts.bitvector == "sux") {
                run_benchmark<gef::B_GEF_STAR<int64_t, SUXBitVector>>(data, factory, opts.strategy, opts.num_queries, opts.iterations, opts.verbose);
            } else if (opts.bitvector == "pasta") {
                run_benchmark<gef::B_GEF_STAR<int64_t, PastaBitVector>>(data, factory, opts.strategy, opts.num_queries, opts.iterations, opts.verbose);
            } else {
                run_benchmark<gef::B_GEF_STAR<int64_t, SDSLBitVector>>(data, factory, opts.strategy, opts.num_queries, opts.iterations, opts.verbose);
            }
        } else { // bgef
            if (opts.bitvector == "sux") {
                run_benchmark<gef::B_GEF<int64_t, SUXBitVector, SUXBitVector>>(data, factory, opts.strategy, opts.num_queries, opts.iterations, opts.verbose);
            } else if (opts.bitvector == "pasta") {
                run_benchmark<gef::B_GEF<int64_t, PastaBitVector, PastaBitVector>>(data, factory, opts.strategy, opts.num_queries, opts.iterations, opts.verbose);
            } else {
                run_benchmark<gef::B_GEF<int64_t, SDSLBitVector, SDSLBitVector>>(data, factory, opts.strategy, opts.num_queries, opts.iterations, opts.verbose);
            }
        }

        return 0;
    } catch (const std::exception& ex) {
        std::cerr << "Error: " << ex.what() << std::endl;
        return 1;
    }
}

