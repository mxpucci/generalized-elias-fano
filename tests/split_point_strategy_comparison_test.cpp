#include <gtest/gtest.h>
#include "gef/U_GEF.hpp"
#include "gef/B_GEF.hpp"
#include "gef/B_GEF_NO_RLE.hpp"
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

// Helper trait to check if a type is a specialization of B_GEF_NO_RLE.
template <typename T>
struct is_b_gef_no_rle : std::false_type {};

template <typename ValType>
struct is_b_gef_no_rle<gef::B_GEF_NO_RLE<ValType>> : std::true_type {};


// Test fixture for typed tests.
template <typename T>
class GEF_SplitPointStrategyComparison_TypedTest : public ::testing::Test {
protected:
    std::shared_ptr<IBitVectorFactory> factory;

    void SetUp() override {
        factory = std::make_shared<SDSLBitVectorFactory>();
    }
};

// Define the list of implementations and types to test with.
using Implementations = ::testing::Types<
    gef::U_GEF<int16_t>, gef::B_GEF<int16_t>, gef::B_GEF_NO_RLE<int16_t>,
    gef::U_GEF<uint16_t>, gef::B_GEF<uint16_t>, gef::B_GEF_NO_RLE<uint16_t>,
    gef::U_GEF<int32_t>, gef::B_GEF<int32_t>, gef::B_GEF_NO_RLE<int32_t>,
    gef::U_GEF<uint32_t>, gef::B_GEF<uint32_t>, gef::B_GEF_NO_RLE<uint32_t>
>;

TYPED_TEST_CASE(GEF_SplitPointStrategyComparison_TypedTest, Implementations);

TYPED_TEST(GEF_SplitPointStrategyComparison_TypedTest, BruteForceIsOptimal) {
    using GEF_Class = TypeParam;
    using value_type = typename get_value_type<GEF_Class>::type;

    struct TestCase {
        std::string name;
        std::vector<value_type> sequence;
    };

    std::vector<TestCase> test_cases;

    // Highly compressible sequence: small gaps, mostly increasing
    test_cases.push_back({
        "Highly Compressible",
        gef::test::generate_random_sequence<value_type>(2000, 0, 100, 0.8, 10)
    });

    // Slightly compressible sequence: medium gaps, some noise
    test_cases.push_back({
        "Slightly Compressible",
        gef::test::generate_random_sequence<value_type>(2000, 0, 1000, 0.3, 3)
    });

    // Poorly compressible sequence: large gaps, random data
    test_cases.push_back({
        "Poorly Compressible",
        gef::test::generate_random_sequence<value_type>(2000, 0, 50000, 0.0, 1)
    });

    // All elements are the same
    test_cases.push_back({
        "All Elements Identical",
        std::vector<value_type>(1000, 42)
    });

    // Alternating values
    std::vector<value_type> alternating_seq;
    alternating_seq.reserve(1000);
    for(int i = 0; i < 1000; ++i) {
        alternating_seq.push_back(i % 2 == 0 ? 10 : 1000);
    }
    test_cases.push_back({ "Alternating Values", alternating_seq });


    for (const auto& test : test_cases) {
        SCOPED_TRACE("Test Case: " + test.name);

        // Instantiate with Brute Force strategy
        GEF_Class gef_brute_force(this->factory, test.sequence, gef::BRUTE_FORCE_SPLIT_POINT);
        size_t brute_force_size = gef_brute_force.size_in_bytes();

        // Instantiate with Binary Search strategy
        GEF_Class gef_binary_search(this->factory, test.sequence, gef::BINARY_SEARCH_SPLIT_POINT);
        size_t binary_search_size = gef_binary_search.size_in_bytes();

        // Instantiate with Approximate strategy
        GEF_Class gef_approximate(this->factory, test.sequence, gef::APPROXIMATE_SPLIT_POINT);
        size_t approximate_size = gef_approximate.size_in_bytes();

        const auto [min_it, max_it] = std::minmax_element(test.sequence.begin(), test.sequence.end());
        const value_type min_val = *min_it;
        const value_type max_val = *max_it;
        const uint64_t u = max_val - min_val + 1;
        const uint8_t total_bits = (u > 1) ? static_cast<uint8_t>(floor(log2(u)) + 1) : 1;

        // For B_GEF_NO_RLE, the cost function is convex, so binary search is guaranteed
        // to find the optimal split point, which is the same one brute force finds.
        // For other implementations (like B_GEF with RLE), the cost function might have
        // local minima, so binary search is only a heuristic.
        if constexpr (is_b_gef_no_rle<GEF_Class>::value) {
            EXPECT_EQ(brute_force_size, binary_search_size)
                    << "For B_GEF_NO_RLE, brute force size (" << brute_force_size
                    << ") should be EQUAL to binary search size (" << binary_search_size << ")";
        } else {
            EXPECT_LE(brute_force_size, binary_search_size)
                << "Brute force size (" << brute_force_size
                << ") should be <= binary search size (" << binary_search_size << ")";
        }

        // The approximate strategy should always be larger or equal in size than brute force.
        EXPECT_LE(brute_force_size, approximate_size)
            << "Brute force size (" << brute_force_size
            << ") should be <= approximate size (" << approximate_size << ")";

        // Also verify correctness of the data
        for(size_t i = 0; i < test.sequence.size(); ++i) {
            ASSERT_EQ(gef_brute_force.at(i), test.sequence[i]);
            ASSERT_EQ(gef_binary_search.at(i), test.sequence[i]);
            ASSERT_EQ(gef_approximate.at(i), test.sequence[i]);
        }
    }
}