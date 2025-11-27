#include <iostream>
#include <vector>
#include <random>
#include <memory>
#include "gef/U_GEF.hpp"
#include "datastructures/SDSLBitVectorFactory.hpp"

int main() {
    // Create random data in range [0, 65535]
    // This corresponds to total_bits = 16.
    // h=0 cost should be ~16 bits per element (plus metadata).
    // Ratio 16/64 = 25%.
    
    const size_t N = 100000;
    std::vector<uint64_t> data(N);
    std::mt19937 gen(42);
    std::uniform_int_distribution<uint64_t> dist(0, 65535);
    
    for (size_t i = 0; i < N; ++i) {
        data[i] = dist(gen);
    }
    
    auto factory = std::make_shared<SDSLBitVectorFactory>();
    
    // Compress
    gef::U_GEF<uint64_t> ugef(factory, data, gef::SplitPointStrategy::APPROXIMATE_SPLIT_POINT);
    
    size_t size_bytes = ugef.size_in_bytes();
    double ratio = (double)size_bytes * 100.0 / (N * 8.0);
    
    std::cout << "Total bits: " << (int)gef::U_GEF<uint64_t>::bits_for_range(0, 65535) << std::endl;
    std::cout << "Chosen split point b: " << (int)ugef.split_point() << std::endl;
    // We can't access h directly but we can infer it from b and total_bits
    // actually we can't easily get h without being inside or friend, but b tells us.
    // If b < total_bits, h > 0.
    
    std::cout << "Compression Ratio: " << ratio << "%" << std::endl;
    
    if (ratio > 30.0) {
        std::cout << "FAIL: Ratio too high. Expected ~25%." << std::endl;
        return 1;
    }
    
    std::cout << "SUCCESS: Ratio is reasonable." << std::endl;
    return 0;
}



