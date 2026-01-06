#include <gtest/gtest.h>
#include "gef/RLE_GEF.hpp"
#include "gef_test_utils.hpp"
#include <vector>
#include <filesystem>
#include <fstream>
#include <type_traits>
#include "gef/U_GEF.hpp"
#include "gef/B_GEF.hpp"
#include "gef/B_STAR_GEF.hpp"

// Helper trait to extract the underlying value_type from a GEF implementation class.
// (Moved to gef_test_utils.hpp)

// Test fixture for typed tests. This allows us to run the same tests
// for different GEF implementations and integral types.
template <typename T>
class GEF_SplitPointStrategy_TypedTest : public ::testing::Test {
protected:
    void SetUp() override {
        // no setup needed
    }
};

namespace {
// Define the list of all implementations and types we want to test with.
using GEF_SplitStrategy_Implementations = ::testing::Types<
    gef::internal::U_GEF<int8_t>, gef::internal::B_GEF<int8_t>, gef::internal::B_STAR_GEF<int8_t>,
    gef::internal::U_GEF<uint8_t>, gef::internal::B_GEF<uint8_t>, gef::internal::B_STAR_GEF<uint8_t>,
    gef::internal::U_GEF<int16_t>, gef::internal::B_GEF<int16_t>, gef::internal::B_STAR_GEF<int16_t>,
    gef::internal::U_GEF<uint16_t>, gef::internal::B_GEF<uint16_t>, gef::internal::B_STAR_GEF<uint16_t>,
    gef::internal::U_GEF<int32_t>, gef::internal::B_GEF<int32_t>, gef::internal::B_STAR_GEF<int32_t>,
    gef::internal::U_GEF<uint32_t>, gef::internal::B_GEF<uint32_t>, gef::internal::B_STAR_GEF<uint32_t>,
    gef::internal::U_GEF<int64_t>, gef::internal::B_GEF<int64_t>, gef::internal::B_STAR_GEF<int64_t>,
    gef::internal::U_GEF<uint64_t>, gef::internal::B_GEF<uint64_t>, gef::internal::B_STAR_GEF<uint64_t>
>;
}

TYPED_TEST_CASE(GEF_SplitPointStrategy_TypedTest, GEF_SplitStrategy_Implementations);

TYPED_TEST(GEF_SplitPointStrategy_TypedTest, ApproximateSplitPoint) {
    using GEF_Class = TypeParam;
    using value_type = typename get_value_type<GEF_Class>::type;

    value_type max_val = 100;
    if constexpr (sizeof(value_type) > 1) {
        max_val = 1000;
    }
    const std::vector<value_type> sequence = gef::test::generate_random_sequence<value_type>(1000, 0, max_val);

    GEF_Class gef_impl(sequence, gef::APPROXIMATE_SPLIT_POINT);

    for(size_t i = 0; i < sequence.size(); ++i) {
        ASSERT_EQ(gef_impl.at(i), sequence[i]);
    }
}

TYPED_TEST(GEF_SplitPointStrategy_TypedTest, BruteForceSplitPoint) {
    using GEF_Class = TypeParam;
    using value_type = typename get_value_type<GEF_Class>::type;

    value_type max_val = 100;
    if constexpr (sizeof(value_type) > 1) {
        max_val = 1000;
    }
    const std::vector<value_type> sequence = gef::test::generate_random_sequence<value_type>(1000, 0, max_val);

    GEF_Class gef_impl(sequence, gef::OPTIMAL_SPLIT_POINT);

    for(size_t i = 0; i < sequence.size(); ++i) {
        ASSERT_EQ(gef_impl.at(i), sequence[i]);
    }
}
