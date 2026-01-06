#include <gtest/gtest.h>
#include "gef/U_GEF.hpp"
#include "gef/B_GEF.hpp"
#include "gef/B_STAR_GEF.hpp"
#include "gef/RLE_GEF.hpp"
#include "gef_test_utils.hpp"
#include <vector>
#include <type_traits>

// Helper trait to extract the underlying value_type from a GEF implementation class.
// (Moved to gef_test_utils.hpp)

// Test fixture for typed tests.
template <typename T>
class GEF_Overhead_TypedTest : public ::testing::Test {
protected:
    void SetUp() override {
        // no setup needed
    }
};

namespace {
// Define the list of implementations and types to test with.
using GEF_Overhead_Implementations = ::testing::Types<
    gef::internal::RLE_GEF<int8_t>, gef::internal::U_GEF<int8_t>, gef::internal::B_GEF<int8_t>, gef::internal::B_STAR_GEF<int8_t>,
    gef::internal::RLE_GEF<uint8_t>, gef::internal::U_GEF<uint8_t>, gef::internal::B_GEF<uint8_t>, gef::internal::B_STAR_GEF<uint8_t>,
    gef::internal::RLE_GEF<int16_t>, gef::internal::U_GEF<int16_t>, gef::internal::B_GEF<int16_t>, gef::internal::B_STAR_GEF<int16_t>,
    gef::internal::RLE_GEF<uint16_t>, gef::internal::U_GEF<uint16_t>, gef::internal::B_GEF<uint16_t>, gef::internal::B_STAR_GEF<uint16_t>,
    gef::internal::RLE_GEF<int32_t>, gef::internal::U_GEF<int32_t>, gef::internal::B_GEF<int32_t>, gef::internal::B_STAR_GEF<int32_t>,
    gef::internal::RLE_GEF<uint32_t>, gef::internal::U_GEF<uint32_t>, gef::internal::B_GEF<uint32_t>, gef::internal::B_STAR_GEF<uint32_t>,
    gef::internal::RLE_GEF<int64_t>, gef::internal::U_GEF<int64_t>, gef::internal::B_GEF<int64_t>, gef::internal::B_STAR_GEF<int64_t>,
    gef::internal::RLE_GEF<uint64_t>, gef::internal::U_GEF<uint64_t>, gef::internal::B_GEF<uint64_t>, gef::internal::B_STAR_GEF<uint64_t>
>;
}

TYPED_TEST_CASE(GEF_Overhead_TypedTest, GEF_Overhead_Implementations);

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
    value_type mid_max;
    if constexpr (sizeof(value_type) == 1) mid_max = 100;
    else mid_max = 1000;
    test_cases.push_back({
        "Medium Compressibility",
        gef::test::generate_random_sequence<value_type>(2000, 
            std::is_signed_v<value_type> ? static_cast<value_type>(-mid_max) : static_cast<value_type>(0), mid_max, 0.3, 3)
    });

    // Poorly compressible
    value_type poor_max;
    if constexpr (sizeof(value_type) == 1) poor_max = 100;
    else poor_max = 10000;
    test_cases.push_back({
        "Poorly Compressible",
        gef::test::generate_random_sequence<value_type>(2000, 
            std::is_signed_v<value_type> ? static_cast<value_type>(-poor_max) : static_cast<value_type>(0), poor_max, 0.0, 1)
    });

    // All elements identical
    test_cases.push_back({
        "All Identical",
        std::vector<value_type>(1000, 42)
    });

    // Long Sequence
    value_type long_max;
    if constexpr (sizeof(value_type) == 1) long_max = 100;
    else long_max = 500;
    test_cases.push_back({
        "Long Sequence",
        gef::test::generate_random_sequence<value_type>(10000, 
            std::is_signed_v<value_type> ? static_cast<value_type>(-long_max) : static_cast<value_type>(0), long_max, 0.5, 4)
    });

    for (const auto& test : test_cases) {
        SCOPED_TRACE("Test Case: " + test.name);

        GEF_Class gef(test.sequence);
        
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

