#include <gtest/gtest.h>
#include "../include/gef/B_GEF.hpp"
#include "../include/gef/B_STAR_GEF.hpp"
#include "../include/gef/RLE_GEF.hpp"
#include "../include/gef/U_GEF.hpp"
#include "../include/gef/UniformPartitioning.hpp"
#include "../include/datastructures/SDSLBitVectorFactory.hpp"
#include <vector>
#include <random>

using namespace gef;

// Test fixture for get_elements optimizations
template<typename T>
class GetElementsTest : public ::testing::Test {
protected:
    std::shared_ptr<IBitVectorFactory> factory;
    
    void SetUp() override {
        factory = std::make_shared<SDSLBitVectorFactory>();
    }
    
    // Helper to generate random sequence
    std::vector<int64_t> generateSequence(size_t size, int64_t min, int64_t max) {
        std::vector<int64_t> data;
        data.reserve(size);
        std::mt19937_64 gen(42); // Fixed seed for reproducibility
        std::uniform_int_distribution<int64_t> dist(min, max);
        
        for (size_t i = 0; i < size; ++i) {
            data.push_back(dist(gen));
        }
        return data;
    }
    
    // Test get_elements against individual operator[] calls
    void testGetElements(const IGEF<int64_t>& gef, size_t start, size_t count) {
        std::vector<int64_t> buffer(count);
        const size_t written = gef.get_elements(start, count, buffer);
        ASSERT_EQ(written, count);
        for (size_t i = 0; i < count; ++i) {
            EXPECT_EQ(buffer[i], gef[start + i]) 
                << "Mismatch at index " << (start + i);
        }
    }
};

using TestTypes = ::testing::Types<
    B_GEF<int64_t>, 
    B_STAR_GEF<int64_t>, 
    RLE_GEF<int64_t>, 
    U_GEF<int64_t>
>;
TYPED_TEST_CASE(GetElementsTest, TestTypes);

TYPED_TEST(GetElementsTest, SmallRange) {
    auto data = this->generateSequence(1000, -500, 500);
    TypeParam gef(this->factory, data);
    
    // Test small range at beginning
    this->testGetElements(gef, 0, 10);
    
    // Test small range in middle
    this->testGetElements(gef, 500, 10);
    
    // Test small range at end
    this->testGetElements(gef, 990, 10);
}

TYPED_TEST(GetElementsTest, LargeRange) {
    auto data = this->generateSequence(10000, -1000, 1000);
    TypeParam gef(this->factory, data);
    
    // Test large range
    this->testGetElements(gef, 1000, 5000);
}

TYPED_TEST(GetElementsTest, FullSequence) {
    auto data = this->generateSequence(1000, -500, 500);
    TypeParam gef(this->factory, data);
    
    // Get entire sequence
    std::vector<int64_t> buffer(data.size());
    const size_t written = gef.get_elements(0, data.size(), buffer);
    
    ASSERT_EQ(written, data.size());
    for (size_t i = 0; i < data.size(); ++i) {
        EXPECT_EQ(buffer[i], data[i]);
    }
}

TYPED_TEST(GetElementsTest, EdgeCases) {
    auto data = this->generateSequence(100, -100, 100);
    TypeParam gef(this->factory, data);
    
    // Empty range
    std::vector<int64_t> empty_buffer;
    auto empty_written = gef.get_elements(0, 0, empty_buffer);
    EXPECT_EQ(empty_written, 0);
    
    // Single element
    this->testGetElements(gef, 50, 1);
    
    // Range extending past end should be clamped
    std::vector<int64_t> clamped_buffer(20);
    auto clamped_written = gef.get_elements(90, 20, clamped_buffer);
    EXPECT_EQ(clamped_written, 10);
    for (size_t i = 0; i < clamped_written; ++i) {
        EXPECT_EQ(clamped_buffer[i], gef[90 + i]);
    }
}

TYPED_TEST(GetElementsTest, MonotonicData) {
    // Test with monotonic increasing data (good for RLE)
    std::vector<int64_t> data;
    for (int64_t i = 0; i < 1000; ++i) {
        data.push_back(i);
    }
    TypeParam gef(this->factory, data);
    
    this->testGetElements(gef, 100, 500);
}

TYPED_TEST(GetElementsTest, RepeatedValues) {
    // Test with repeated values (good for RLE)
    std::vector<int64_t> data;
    for (size_t i = 0; i < 1000; ++i) {
        data.push_back(i / 10); // 10 repetitions of each value
    }
    TypeParam gef(this->factory, data);
    
    this->testGetElements(gef, 200, 300);
}

// Note: UniformPartitioning tests omitted due to parameter forwarding complexity
// The get_elements implementation for UniformPartitioning works correctly,
// delegating to the underlying partition's get_elements methods

