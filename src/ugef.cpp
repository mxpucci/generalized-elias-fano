#include <iostream>
#include <vector>
#include <string>
#include <fstream>
#include <limits>
#include <stdexcept>
#include <cstring> // For std::strerror
#include <cerrno>  // For errno
#include <cstdint> // For uint32_t
#include "gef/U_GEF.hpp"
#include "gef/UniformPartitioning.hpp"
#include "gef/utils.hpp"


// Wrapper to adapt U_GEF constructor for UniformPartitioning
template<typename T>
struct B_GEF_Wrapper : public gef::internal::U_GEF<T> {
    // Constructor for compression
    B_GEF_Wrapper(gef::Span<const T> data,
                  gef::SplitPointStrategy strategy = gef::SplitPointStrategy::APPROXIMATE_SPLIT_POINT)
            : gef::internal::U_GEF<T>(data, strategy) {}

    // Default constructor for loading from stream
    B_GEF_Wrapper() : gef::internal::U_GEF<T>() {}
};

/**
 * @brief The main entry point of the program.
 *
 * It expects one command-line argument: the path to the binary file to read.
 */
int main(const int argc, char* argv[]) {
    if (argc < 2) {
        std::cerr << "Usage: " << argv[0] << " <filename>" << std::endl;
        return 1;
    }


    for (size_t arg_index = 1; arg_index < argc; ++arg_index) {
        std::string input_filename = argv[arg_index];
        std::vector<int64_t> input_data = read_data_binary<int64_t, int64_t>(input_filename, true);

        double input_size_mb = static_cast<double>(input_data.size() * sizeof(int64_t)) / (1024.0 * 1024.0);

        constexpr size_t partition_size = static_cast<size_t>(1ULL << 20);
        const auto split_strategy = gef::SplitPointStrategy::APPROXIMATE_SPLIT_POINT;
        std::vector<size_t> k_values = {partition_size};
        double best_compression = 100;
        for (size_t k : k_values) {
            if (input_data.empty()) continue;
            gef::UniformPartitioning<int64_t,
                                      B_GEF_Wrapper<int64_t>,
                                      partition_size,
                                      gef::SplitPointStrategy>
                partitioned_gef(input_data, split_strategy);
            // double partitioned_size_mb = partitioned_gef.size_in_megabytes();
            double partitioned_size_mb = static_cast<double>(partitioned_gef.theoretical_size_in_bytes()) / (1024.0 * 1024.0);
            best_compression = std::min(best_compression, (100 * partitioned_size_mb) / input_size_mb);
        }

        std::cout << argv[arg_index] << " - Compression ratio: " << best_compression << std::endl;
    }
    return 0;
}