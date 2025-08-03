#include <gtest/gtest.h>
#include "gef/RLE_GEF.hpp"
#include "datastructures/SDSLBitVectorFactory.hpp"
#include "gef_test_utils.hpp"
#include <vector>
#include <memory>
#include <filesystem>
#include <fstream>
#include <type_traits>
#include "gef/U_GEF.hpp"
#include "gef/B_GEF.hpp"
#include "gef/B_GEF_NO_RLE.hpp"

// Helper trait to extract the underlying value_type from a GEF implementation class.
// e.g., get_value_type<gef::RLE_GEF<int32_t>>::type will be int32_t.
template<typename T>
struct get_value_type;

template<template<typename> class C, typename T>
struct get_value_type<C<T>> {
    using type = T;
};


// Test fixture for typed tests. This allows us to run the same tests
// for different GEF implementations and integral types.
template <typename T>
class GEF_SplitPointStrategy_TypedTest : public ::testing::Test {
protected:
    // A shared factory for creating bit vectors in tests.
    std::shared_ptr<IBitVectorFactory> factory;

    void SetUp() override {
        // Initialize the factory before each test.
        factory = std::make_shared<SDSLBitVectorFactory>();
    }
};

// Define the list of all implementations and types we want to test with.
using Implementations = ::testing::Types<
    gef::U_GEF<int8_t>, gef::B_GEF<int8_t>, gef::B_GEF_NO_RLE<int8_t>,
    gef::U_GEF<uint8_t>, gef::B_GEF<uint8_t>, gef::B_GEF_NO_RLE<uint8_t>,
    gef::U_GEF<int16_t>, gef::B_GEF<int16_t>, gef::B_GEF_NO_RLE<int16_t>,
    gef::U_GEF<uint16_t>, gef::B_GEF<uint16_t>, gef::B_GEF_NO_RLE<uint16_t>,
    gef::U_GEF<int32_t>, gef::B_GEF<int32_t>, gef::B_GEF_NO_RLE<int32_t>,
    gef::U_GEF<uint32_t>, gef::B_GEF<uint32_t>, gef::B_GEF_NO_RLE<uint32_t>,
    gef::U_GEF<int64_t>, gef::B_GEF<int64_t>, gef::B_GEF_NO_RLE<int64_t>,
    gef::U_GEF<uint64_t>, gef::B_GEF<uint64_t>, gef::B_GEF_NO_RLE<uint64_t>
>;

TYPED_TEST_CASE(GEF_SplitPointStrategy_TypedTest, Implementations);

TYPED_TEST(GEF_SplitPointStrategy_TypedTest, ApproximateSplitPoint) {
    using GEF_Class = TypeParam;
    using value_type = typename get_value_type<GEF_Class>::type;

    const std::vector<value_type> sequence = gef::test::generate_random_sequence<value_type>(1000, 0, 1000);

    GEF_Class gef_impl(this->factory, sequence, gef::APPROXIMATE_SPLIT_POINT);

    for(size_t i = 0; i < sequence.size(); ++i) {
        ASSERT_EQ(gef_impl.at(i), sequence[i]);
    }
}

TYPED_TEST(GEF_SplitPointStrategy_TypedTest, BruteForceSplitPoint) {
    using GEF_Class = TypeParam;
    using value_type = typename get_value_type<GEF_Class>::type;

    const std::vector<value_type> sequence = gef::test::generate_random_sequence<value_type>(1000, 0, 1000);

    GEF_Class gef_impl(this->factory, sequence, gef::BRUTE_FORCE_SPLIT_POINT);

    for(size_t i = 0; i < sequence.size(); ++i) {
        ASSERT_EQ(gef_impl.at(i), sequence[i]);
    }
}

TYPED_TEST(GEF_SplitPointStrategy_TypedTest, BinarySearchSplitPoint) {
    using GEF_Class = TypeParam;
    using value_type = typename get_value_type<GEF_Class>::type;

    const std::vector<value_type> sequence = gef::test::generate_random_sequence<value_type>(1000, 0, 1000);

    GEF_Class gef_impl(this->factory, sequence, gef::BINARY_SEARCH_SPLIT_POINT);

    for(size_t i = 0; i < sequence.size(); ++i) {
        ASSERT_EQ(gef_impl.at(i), sequence[i]);
    }
}
