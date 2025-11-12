#include <iostream>
#include <vector>
#include <string>
#include <fstream>
#include <cstdint>
#include "gef/B_GEF_STAR.hpp"
#include "datastructures/SDSLBitVectorFactory.hpp"
#include "gef/UniformedPartitioner.hpp"
#include "gef/utils.hpp"
#include "datastructures/IBitVectorFactory.hpp"
#include "gef/IGEF.hpp"

// Wrapper to adapt B_GEF_NO_RLE constructor for UniformedPartitioner
template<typename T>
struct B_GEF_NO_RLE_Wrapper : public gef::B_GEF_STAR<T> {
    // Constructor for compression with explicit strategy
    B_GEF_NO_RLE_Wrapper(gef::Span<const T> data,
                         const std::shared_ptr<IBitVectorFactory>& factory,
                         gef::SplitPointStrategy strategy)
            : gef::B_GEF_STAR<T>(factory, std::vector<T>(data.data(), data.data() + data.size()), strategy) {}

    // Convenience constructor using the default split-point strategy
    B_GEF_NO_RLE_Wrapper(gef::Span<const T> data,
                         const std::shared_ptr<IBitVectorFactory>& factory)
            : B_GEF_NO_RLE_Wrapper(data, factory, gef::SplitPointStrategy::OPTIMAL_SPLIT_POINT) {}

    // Default constructor for loading from stream
    B_GEF_NO_RLE_Wrapper() : gef::B_GEF_STAR<T>() {}
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


        auto factory = std::make_shared<SDSLBitVectorFactory>();
        double input_size_mb = static_cast<double>(input_data.size() * sizeof(int64_t)) / (1024.0 * 1024.0);


        std::vector<size_t> k_values = {8192};
        double best_compression = 100;
        for (size_t k : k_values) {
            if (input_data.empty()) continue;
            gef::UniformedPartitioner<int64_t, B_GEF_NO_RLE_Wrapper<int64_t>, std::shared_ptr<IBitVectorFactory>> partitioned_gef(input_data, k, factory);
            double partitioned_size_mb = partitioned_gef.size_in_megabytes();
            best_compression = std::min(best_compression, (100 * partitioned_size_mb) / input_size_mb);
        }

        std::cout << argv[arg_index] << " - Compression ratio: " << best_compression << std::endl;
    }
    return 0;
}