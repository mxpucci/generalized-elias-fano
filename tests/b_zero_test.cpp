#include <gtest/gtest.h>
#include <vector>
#include <numeric>
#include <iostream>
#include "gef/U_GEF.hpp"
#include "gef/B_GEF.hpp"
#include "gef/B_GEF_STAR.hpp"
#include "gef/RLE_GEF.hpp"
#include "../include/datastructures/SDSLBitVectorFactory.hpp"

using namespace gef;

template <typename GEF_TYPE>
void check_b_zero_behavior(const std::string& name) {
    // Create a sequence that is highly predictable (linear), which should favor b=0 for delta-based compressors
    // 0, 1, 2, ... 999.
    // total_bits will be ~10.
    // Deltas are 1.
    // If b=0, h=10. Deltas of 1 fit easily.
    size_t N = 1000;
    std::vector<uint64_t> data(N);
    std::iota(data.begin(), data.end(), 0);

    auto factory = std::make_shared<SDSLBitVectorFactory>();
    GEF_TYPE compressor(factory, data);

    uint8_t b = compressor.split_point();
    std::cout << name << " chosen b: " << (int)b << std::endl;
    
    if (b == 0) {
        size_t theo_size = compressor.theoretical_size_in_bytes();
        std::cout << name << " theoretical size: " << theo_size << std::endl;
        // If buggy, L is 1000 * 64 bits = 64000 bits = 8000 bytes.
        // If correct, L is 0 bytes.
        // Total size should be dominated by B (1000 bits = 125 bytes) + G + metadata.
        // Should be well under 2500 bytes.
        EXPECT_LT(theo_size, 2500) << name << " theoretical size is suspiciously large (buggy L?)";
    } else {
        std::cout << name << " did not choose b=0, skipping size check." << std::endl;
    }
    
    // Verify correctness
    std::vector<uint64_t> decoded(N);
    compressor.get_elements(0, N, decoded);
    EXPECT_EQ(data, decoded) << name << " decoding failed";
}

template <typename GEF_TYPE>
void check_constant_behavior(const std::string& name) {
    // Constant data favors RLE with b=0 (all high parts same)
    size_t N = 1000;
    std::vector<uint64_t> data(N, 123);

    auto factory = std::make_shared<SDSLBitVectorFactory>();
    GEF_TYPE compressor(factory, data);

    uint8_t b = compressor.split_point();
    std::cout << name << " (const) chosen b: " << (int)b << std::endl;
    
    if (b == 0) {
        size_t theo_size = compressor.theoretical_size_in_bytes();
        std::cout << name << " (const) theoretical size: " << theo_size << std::endl;
        // If buggy, L is 8000 bytes.
        EXPECT_LT(theo_size, 2500);
    }
    
    std::vector<uint64_t> decoded(N);
    compressor.get_elements(0, N, decoded);
    EXPECT_EQ(data, decoded) << name << " decoding failed";
}

TEST(BZeroTest, U_GEF_Linear) {
    check_b_zero_behavior<U_GEF<uint64_t>>("U_GEF");
}

TEST(BZeroTest, B_GEF_Linear) {
    check_b_zero_behavior<B_GEF<uint64_t>>("B_GEF");
}

TEST(BZeroTest, B_GEF_STAR_Linear) {
    check_b_zero_behavior<B_GEF_STAR<uint64_t>>("B_GEF_STAR");
}

TEST(BZeroTest, RLE_GEF_Linear) {
    check_b_zero_behavior<RLE_GEF<uint64_t>>("RLE_GEF");
}

TEST(BZeroTest, RLE_GEF_Constant) {
    check_constant_behavior<RLE_GEF<uint64_t>>("RLE_GEF");
}

TEST(BZeroTest, U_GEF_Constant) {
    check_constant_behavior<U_GEF<uint64_t>>("U_GEF");
}
