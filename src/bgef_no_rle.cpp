#include <iostream>
#include <vector>
#include <string>
#include <fstream>
#include <cstdint>
#include "gef/B_STAR_GEF.hpp"
#include "gef/UniformPartitioning.hpp"
#include "gef/utils.hpp"
#include "gef/IGEF.hpp"

// Wrapper to adapt B_GEF_NO_RLE constructor for UniformPartitioning
template<typename T>
struct B_GEF_NO_RLE_Wrapper : public gef::internal::B_STAR_GEF<T> {
    // Constructor for compression with explicit strategy
    B_GEF_NO_RLE_Wrapper(gef::Span<const T> data,
                         gef::SplitPointStrategy strategy)
            : gef::internal::B_STAR_GEF<T>(data, strategy) {}

    // Convenience constructor using the default split-point strategy
    B_GEF_NO_RLE_Wrapper(gef::Span<const T> data)
            : B_GEF_NO_RLE_Wrapper(data, gef::SplitPointStrategy::OPTIMAL_SPLIT_POINT) {}

    // Default constructor for loading from stream
    B_GEF_NO_RLE_Wrapper() : gef::internal::B_STAR_GEF<T>() {}
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

        std::cout << input_filename << std::endl;


        double input_size_mb = static_cast<double>(input_data.size() * sizeof(int64_t)) / (1024.0 * 1024.0);


        std::vector<size_t> k_values = {8192};
        double best_compression = 100;
        for (size_t k : k_values) {
            if (input_data.empty()) continue;
            // Since k is fixed at 8192 in the vector, we use it as template argument
            gef::UniformPartitioning<int64_t, B_GEF_NO_RLE_Wrapper<int64_t>, 8192> partitioned_gef(input_data);
            double partitioned_size_mb = partitioned_gef.size_in_megabytes();
            best_compression = std::min(best_compression, (100 * partitioned_size_mb) / input_size_mb);
        }

        std::cout << argv[arg_index] << " - Compression ratio: " << best_compression << std::endl;
    }
    return 0;
}