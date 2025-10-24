#include <gtest/gtest.h>
#include "gef/U_GEF.hpp"
#include "gef/B_GEF.hpp"
#include "gef/B_GEF_STAR.hpp"
#include "gef/RLE_GEF.hpp"
#include "datastructures/SDSLBitVectorFactory.hpp"
#include "gef_test_utils.hpp"
#include <vector>
#include <memory>
#include <type_traits>

// Helper trait to extract the underlying value_type from a GEF implementation class.
template<typename T>
struct get_value_type;

template<template<typename> class C, typename T>
struct get_value_type<C<T>> {
    using type = T;
};

// Test fixture for typed tests.
template <typename T>
class GEF_Overhead_TypedTest : public ::testing::Test {
protected:
    std::shared_ptr<IBitVectorFactory> factory;

    void SetUp() override {
        factory = std::make_shared<SDSLBitVectorFactory>();
    }
};

// Define the list of implementations and types to test with.
using Implementations = ::testing::Types<
    gef::RLE_GEF<int8_t>, gef::U_GEF<int8_t>, gef::B_GEF<int8_t>, gef::B_GEF_STAR<int8_t>,
    gef::RLE_GEF<uint8_t>, gef::U_GEF<uint8_t>, gef::B_GEF<uint8_t>, gef::B_GEF_STAR<uint8_t>,
    gef::RLE_GEF<int16_t>, gef::U_GEF<int16_t>, gef::B_GEF<int16_t>, gef::B_GEF_STAR<int16_t>,
    gef::RLE_GEF<uint16_t>, gef::U_GEF<uint16_t>, gef::B_GEF<uint16_t>, gef::B_GEF_STAR<uint16_t>,
    gef::RLE_GEF<int32_t>, gef::U_GEF<int32_t>, gef::B_GEF<int32_t>, gef::B_GEF_STAR<int32_t>,
    gef::RLE_GEF<uint32_t>, gef::U_GEF<uint32_t>, gef::B_GEF<uint32_t>, gef::B_GEF_STAR<uint32_t>,
    gef::RLE_GEF<int64_t>, gef::U_GEF<int64_t>, gef::B_GEF<int64_t>, gef::B_GEF_STAR<int64_t>,
    gef::RLE_GEF<uint64_t>, gef::U_GEF<uint64_t>, gef::B_GEF<uint64_t>, gef::B_GEF_STAR<uint64_t>
>;

TYPED_TEST_CASE(GEF_Overhead_TypedTest, Implementations);

TYPED_TEST(GEF_Overhead_TypedTest, SDSLOverheadIsReasonable) {
    using GEF_Class = TypeParam;
    using value_type = typename get_value_type<GEF_Class>::type;

    struct TestCase {
        std::string name;
        std::vector<value_type> sequence;
    };

    std::vector<TestCase> test_cases;

    // Highly compressible sequence
    test_cases.push_back({
        "Highly Compressible",
        gef::test::generate_random_sequence<value_type>(2000, 
            std::is_signed_v<value_type> ? -100 : 0, 100, 0.8, 10)
    });

    // Medium compressibility
    test_cases.push_back({
        "Medium Compressibility",
        gef::test::generate_random_sequence<value_type>(2000, 
            std::is_signed_v<value_type> ? -1000 : 0, 1000, 0.3, 3)
    });

    // Poorly compressible
    test_cases.push_back({
        "Poorly Compressible",
        gef::test::generate_random_sequence<value_type>(2000, 
            std::is_signed_v<value_type> ? -10000 : 0, 10000, 0.0, 1)
    });

    // All elements identical
    test_cases.push_back({
        "All Identical",
        std::vector<value_type>(1000, 42)
    });

    // Long sequence
    test_cases.push_back({
        "Long Sequence",
        gef::test::generate_random_sequence<value_type>(10000, 
            std::is_signed_v<value_type> ? -500 : 0, 500, 0.5, 4)
    });

    for (const auto& test : test_cases) {
        SCOPED_TRACE("Test Case: " + test.name);

        GEF_Class gef(this->factory, test.sequence);
        
        size_t theoretical = gef.theoretical_size_in_bytes();
        size_t actual_without_supports = gef.size_in_bytes_without_supports();
        
        // SDSL has a fixed overhead (~64-128 bytes per int_vector for headers/alignment)
        // Since theoretical now uses width*size (matching SDSL's data storage exactly),
        // the only difference should be SDSL's fixed header/alignment overhead.
        
        size_t allowed_max = theoretical + 512;  // Allow 512 bytes for SDSL overhead
        
        EXPECT_LE(actual_without_supports, allowed_max)
            << "[" << test.name << "] " 
            << "SDSL overhead too large: theoretical=" << theoretical 
            << " bytes, actual_without_supports=" << actual_without_supports 
            << " bytes, allowed_max=" << allowed_max;
        
        // Sanity check: actual should always be >= theoretical
        EXPECT_GE(actual_without_supports, theoretical)
            << "[" << test.name << "] "
            << "Actual size cannot be less than theoretical";
    }
}

