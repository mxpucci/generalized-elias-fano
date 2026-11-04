#include <bit>
#include <chrono>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

#ifdef _OPENMP
#include <omp.h>
#endif

#include "gef/gef.hpp"

using Clock = std::chrono::steady_clock;

std::vector<int64_t> read_series(const std::filesystem::path& path) {
    static_assert(std::endian::native == std::endian::little);
    std::ifstream input(path, std::ios::binary);
    uint64_t count = 0;
    if (!input.read(reinterpret_cast<char*>(&count), sizeof(count))) {
        throw std::runtime_error("cannot read time-series header");
    }
    const uint64_t bytes = std::filesystem::file_size(path);
    const unsigned __int128 payload = static_cast<unsigned __int128>(count) * sizeof(int64_t);
    const size_t header = bytes == payload + 16 ? 16 : 8;
    if (!count || bytes != payload + header) throw std::runtime_error("invalid time-series length");
    input.seekg(header);
    std::vector<int64_t> values(count);
    for (size_t begin = 0; begin < values.size();) {
        const size_t length = std::min<size_t>(values.size() - begin, 1 << 20);
        if (!input.read(reinterpret_cast<char*>(values.data() + begin), length * sizeof(int64_t))) {
            throw std::runtime_error("truncated time-series data");
        }
        begin += length;
    }
    return values;
}

template<class Index, class Oracle>
Index build_partitioned(const std::vector<int64_t>& values, gef::PartitionOptions options,
                        double& preprocessing_seconds) {
    const auto begin = Clock::now();
    Oracle oracle(values, options);
    preprocessing_seconds = std::chrono::duration<double>(Clock::now() - begin).count();
    return Index(values, oracle, options);
}

template<class Builder>
void measure(const std::string& dataset, const std::string& codec,
             const std::vector<int64_t>& values, gef::PartitionOptions options, Builder build) {
    double preprocessing_seconds = 0;
    const auto begin = Clock::now();
    auto index = build(preprocessing_seconds);
    const double build_seconds = std::chrono::duration<double>(Clock::now() - begin).count();
    std::vector<int64_t> decoded(values.size());
    if (index.get_elements(0, values.size(), decoded) != values.size() || decoded != values) {
        throw std::runtime_error(codec + " failed full decoding verification");
    }
    std::cout << dataset << ',' << codec << ',' << values.size() << ',' << options.grid << ','
              << options.epsilon << ',' << options.block_overhead_bits << ',' << preprocessing_seconds
              << ',' << build_seconds << ',' << index.size_in_bytes() << ','
              << index.theoretical_size_in_bytes() << ',';
    if constexpr (requires { index.estimated_bits(); }) {
        std::cout << index.estimated_bits() << ',' << index.partition_ends().size();
    } else {
        std::cout << ',' << (values.size() / 32000 + (values.size() % 32000 != 0));
    }
    std::cout << ",1" << std::endl;
}

int main(int argc, char** argv) {
    try {
        if (argc < 2 || argc > 5) {
            throw std::invalid_argument("usage: partition_oracle_comparison dataset.bin [grid=256] [epsilon=0.1] [block_overhead_bits=256]");
        }
#ifdef _OPENMP
        omp_set_dynamic(0);
        omp_set_num_threads(1);
#endif
        gef::PartitionOptions options;
        if (argc > 2) options.grid = std::stoull(argv[2]);
        if (argc > 3) options.epsilon = std::stod(argv[3]);
        if (argc > 4) options.block_overhead_bits = std::stoull(argv[4]);
        const std::filesystem::path path(argv[1]);
        const auto values = read_series(path);
        const std::string dataset = path.stem().string();
        std::cout << std::setprecision(12)
                  << "dataset,codec,n,grid,epsilon,block_overhead_bits,preprocessing_seconds,build_seconds,size_in_bytes,theoretical_bytes,estimated_bits,partitions,verified\n";
        measure(dataset, "bstar_uniform32000", values, options, [&](double&) {
            return gef::B_STAR_GEF<int64_t>(values);
        });
        measure(dataset, "bstar_partitioned_exact", values, options, [&](double& preprocessing) {
            return build_partitioned<gef::PARTITIONED_B_STAR_GEF<int64_t>, gef::ExactBStarOracle<int64_t>>(
                values, options, preprocessing);
        });
        measure(dataset, "bstar_partitioned_bound", values, options, [&](double& preprocessing) {
            using Oracle = gef::BoundBStarOracle<int64_t>;
            return build_partitioned<gef::PARTITIONED_B_STAR_GEF<int64_t, Oracle>, Oracle>(
                values, options, preprocessing);
        });
        measure(dataset, "rle_uniform32000", values, options, [&](double&) {
            return gef::RLE_GEF<int64_t>(values);
        });
        measure(dataset, "rle_partitioned", values, options, [&](double& preprocessing) {
            return build_partitioned<gef::PARTITIONED_RLE_GEF<int64_t>, gef::RLEPartitionOracle<int64_t>>(
                values, options, preprocessing);
        });
    } catch (const std::exception& error) {
        std::cerr << error.what() << '\n';
        return 1;
    }
}
